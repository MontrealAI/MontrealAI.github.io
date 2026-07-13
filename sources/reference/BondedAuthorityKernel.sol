// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IActiveAGINodeRegistry {
    function isActiveOperator(address operator) external view returns (bool);
}

interface IERC20Minimal {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

/// @notice Reference enforcement kernel. It has not been audited and must not be used with live value as-is.
contract BondedAuthorityKernel {
    enum BondKind { WorkerProof, ValidatorAccuracy, ClaimAuthority, Challenger, CapabilityReuse }
    enum BondStatus { None, Locked, Released, Slashed }
    enum Transition { Paid, Trusted, ChallengeSafe, Reusable }

    struct BondReceipt {
        bytes32 bondId;
        bytes32 missionId;
        BondKind kind;
        BondStatus status;
        address actor;
        uint256 amount;
        bytes32 proofRoot;
        bytes32 policyHash;
        uint64 expiry;
        uint256 returnedAmount;
        uint256 slashedAmount;
    }

    struct TransitionReceipt {
        bool granted;
        bytes32 proofRoot;
        bytes32 policyHash;
        uint64 grantedAt;
    }

    IERC20Minimal public immutable agiAlpha;
    IActiveAGINodeRegistry public immutable agiNodeRegistry;
    address public owner;
    address public resolver;
    address public treasury;
    uint256 private entered;

    mapping(bytes32 => BondReceipt) private bonds;
    mapping(bytes32 => mapping(Transition => TransitionReceipt)) public transitions;
    mapping(bytes32 => bool) public challengeFiled;

    event BondLocked(bytes32 indexed bondId, bytes32 indexed missionId, BondKind indexed kind, address actor, uint256 amount, bytes32 proofRoot, bytes32 policyHash);
    event BondResolved(bytes32 indexed bondId, BondStatus status, uint256 returnedAmount, uint256 slashedAmount, address slashRecipient);
    event TransitionGranted(bytes32 indexed missionId, Transition indexed transition, bytes32 proofRoot, bytes32 policyHash);
    event ChallengeStateChanged(bytes32 indexed missionId, bool filed);
    event ResolverChanged(address indexed resolver);
    event TreasuryChanged(address indexed treasury);

    modifier onlyOwner() { require(msg.sender == owner, "NOT_OWNER"); _; }
    modifier onlyResolver() { require(msg.sender == resolver, "NOT_RESOLVER"); _; }
    modifier nonReentrant() { require(entered == 0, "REENTRANCY"); entered = 1; _; entered = 0; }

    constructor(address token, address nodeRegistry, address initialResolver, address initialTreasury) {
        require(token != address(0) && nodeRegistry != address(0) && initialResolver != address(0) && initialTreasury != address(0), "ZERO_ADDRESS");
        agiAlpha = IERC20Minimal(token);
        agiNodeRegistry = IActiveAGINodeRegistry(nodeRegistry);
        owner = msg.sender;
        resolver = initialResolver;
        treasury = initialTreasury;
    }

    function setResolver(address nextResolver) external onlyOwner {
        require(nextResolver != address(0), "ZERO_ADDRESS");
        resolver = nextResolver;
        emit ResolverChanged(nextResolver);
    }

    function setTreasury(address nextTreasury) external onlyOwner {
        require(nextTreasury != address(0), "ZERO_ADDRESS");
        treasury = nextTreasury;
        emit TreasuryChanged(nextTreasury);
    }

    function transferOwnership(address nextOwner) external onlyOwner {
        require(nextOwner != address(0), "ZERO_ADDRESS");
        owner = nextOwner;
    }

    function lockBond(
        bytes32 bondId,
        bytes32 missionId,
        BondKind kind,
        uint256 amount,
        bytes32 proofRoot,
        bytes32 policyHash,
        uint64 expiry
    ) external nonReentrant {
        require(bonds[bondId].status == BondStatus.None, "BOND_EXISTS");
        require(amount > 0, "ZERO_AMOUNT");
        require(missionId != bytes32(0) && policyHash != bytes32(0), "BAD_BINDING");
        if (kind == BondKind.ValidatorAccuracy) require(agiNodeRegistry.isActiveOperator(msg.sender), "VALIDATOR_NOT_ACTIVE_AGI_NODE");
        _safeTransferFrom(address(agiAlpha), msg.sender, address(this), amount);
        bonds[bondId] = BondReceipt({
            bondId: bondId,
            missionId: missionId,
            kind: kind,
            status: BondStatus.Locked,
            actor: msg.sender,
            amount: amount,
            proofRoot: proofRoot,
            policyHash: policyHash,
            expiry: expiry,
            returnedAmount: 0,
            slashedAmount: 0
        });
        emit BondLocked(bondId, missionId, kind, msg.sender, amount, proofRoot, policyHash);
    }

    function bindProofRoot(bytes32 bondId, bytes32 proofRoot) external onlyResolver {
        BondReceipt storage bond = bonds[bondId];
        require(bond.status == BondStatus.Locked, "NOT_LOCKED");
        require(bond.proofRoot == bytes32(0) || bond.proofRoot == proofRoot, "ROOT_ALREADY_BOUND");
        bond.proofRoot = proofRoot;
    }

    function markChallengeFiled(bytes32 missionId, bool filed) external onlyResolver {
        challengeFiled[missionId] = filed;
        emit ChallengeStateChanged(missionId, filed);
    }

    function grantTransition(
        bytes32 missionId,
        Transition transition,
        bytes32 proofRoot,
        bytes32 policyHash,
        bytes32[] calldata bondIds
    ) external onlyResolver {
        require(!transitions[missionId][transition].granted, "ALREADY_GRANTED");
        uint256 observedMask;
        for (uint256 i = 0; i < bondIds.length; i++) {
            BondReceipt storage bond = bonds[bondIds[i]];
            require(bond.status == BondStatus.Locked, "BOND_NOT_LOCKED");
            require(bond.missionId == missionId, "MISSION_MISMATCH");
            require(bond.policyHash == policyHash, "POLICY_MISMATCH");
            require(bond.proofRoot == bytes32(0) || bond.proofRoot == proofRoot, "PROOF_MISMATCH");
            if (bond.expiry != 0) require(block.timestamp <= bond.expiry, "BOND_EXPIRED");
            observedMask |= (uint256(1) << uint256(bond.kind));
        }
        uint256 required = _requiredMask(transition, challengeFiled[missionId]);
        require((observedMask & required) == required, "MISSING_REQUIRED_BOND");
        transitions[missionId][transition] = TransitionReceipt(true, proofRoot, policyHash, uint64(block.timestamp));
        emit TransitionGranted(missionId, transition, proofRoot, policyHash);
    }

    function resolveBond(bytes32 bondId, uint16 slashBps, address slashRecipient) external onlyResolver nonReentrant {
        BondReceipt storage bond = bonds[bondId];
        require(bond.status == BondStatus.Locked, "NOT_LOCKED");
        require(slashBps <= 10_000, "BAD_BPS");
        uint256 slashed = (bond.amount * slashBps) / 10_000;
        uint256 returnedAmount = bond.amount - slashed;
        bond.slashedAmount = slashed;
        bond.returnedAmount = returnedAmount;
        bond.status = slashed == 0 ? BondStatus.Released : BondStatus.Slashed;
        if (returnedAmount != 0) _safeTransfer(address(agiAlpha), bond.actor, returnedAmount);
        if (slashed != 0) _safeTransfer(address(agiAlpha), slashRecipient == address(0) ? treasury : slashRecipient, slashed);
        emit BondResolved(bondId, bond.status, returnedAmount, slashed, slashRecipient == address(0) ? treasury : slashRecipient);
    }

    function getBond(bytes32 bondId) external view returns (BondReceipt memory) {
        return bonds[bondId];
    }

    function _requiredMask(Transition transition, bool hasChallenge) internal pure returns (uint256 mask) {
        if (transition == Transition.Paid) {
            mask = (uint256(1) << uint256(BondKind.WorkerProof)) | (uint256(1) << uint256(BondKind.ValidatorAccuracy));
        } else if (transition == Transition.Trusted) {
            mask = uint256(1) << uint256(BondKind.ClaimAuthority);
        } else if (transition == Transition.ChallengeSafe) {
            mask = uint256(1) << uint256(BondKind.ClaimAuthority);
            if (hasChallenge) mask |= uint256(1) << uint256(BondKind.Challenger);
        } else {
            mask = uint256(1) << uint256(BondKind.CapabilityReuse);
        }
    }

    function _safeTransfer(address token, address to, uint256 amount) internal {
        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(IERC20Minimal.transfer.selector, to, amount));
        require(success && (data.length == 0 || abi.decode(data, (bool))), "TRANSFER_FAILED");
    }

    function _safeTransferFrom(address token, address from, address to, uint256 amount) internal {
        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(IERC20Minimal.transferFrom.selector, from, to, amount));
        require(success && (data.length == 0 || abi.decode(data, (bool))), "TRANSFER_FROM_FAILED");
    }
}
