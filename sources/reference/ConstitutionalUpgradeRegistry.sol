// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Reference registry for constitutionally gated GoalOS upgrades.
/// @dev Stores compact commitments only. It is unaudited and must not hold live value as-is.
contract ConstitutionalUpgradeRegistry {
    enum Decision { None, CanaryOnly, Promote, Rollback }

    struct UpgradeEpoch {
        uint64 epoch;
        bytes32 previousEpochRoot;
        bytes32 proposalHash;
        bytes32 candidateArtifactRoot;
        bytes32 invariantReportRoot;
        bytes32 councilRoot;
        bytes32 canaryRoot;
        bytes32 delayedOutcomeRoot;
        bytes32 rollbackRoot;
        bytes32 bondPlanHash;
        bytes32 constitutionalEpochRoot;
        Decision decision;
        uint64 createdAt;
        address submitter;
    }

    address public owner;
    uint64 public lastEpoch;
    bytes32 public lastEpochRoot;
    mapping(address => bool) public authorizedCouncilSubmitter;
    mapping(uint64 => UpgradeEpoch) public upgrades;

    event CouncilSubmitterSet(address indexed submitter, bool authorized);
    event UpgradeRecorded(uint64 indexed epoch, bytes32 indexed constitutionalEpochRoot, bytes32 indexed previousEpochRoot, Decision decision, bytes32 proposalHash, address submitter);

    error NotOwner();
    error NotAuthorized();
    error InvalidEpoch();
    error InvalidPreviousRoot();
    error InvalidDecision();
    error MissingCommitment();

    constructor(address initialOwner) {
        owner = initialOwner == address(0) ? msg.sender : initialOwner;
        authorizedCouncilSubmitter[owner] = true;
    }

    modifier onlyOwner() { if (msg.sender != owner) revert NotOwner(); _; }
    modifier onlyCouncilSubmitter() { if (!authorizedCouncilSubmitter[msg.sender]) revert NotAuthorized(); _; }

    function setCouncilSubmitter(address submitter, bool authorized) external onlyOwner {
        authorizedCouncilSubmitter[submitter] = authorized;
        emit CouncilSubmitterSet(submitter, authorized);
    }

    function transferOwnership(address nextOwner) external onlyOwner {
        require(nextOwner != address(0), "zero owner");
        owner = nextOwner;
        authorizedCouncilSubmitter[nextOwner] = true;
    }

    function recordUpgrade(
        uint64 epoch,
        bytes32 previousEpochRoot,
        bytes32 proposalHash,
        bytes32 candidateArtifactRoot,
        bytes32 invariantReportRoot,
        bytes32 councilRoot,
        bytes32 canaryRoot,
        bytes32 delayedOutcomeRoot,
        bytes32 rollbackRoot,
        bytes32 bondPlanHash,
        bytes32 constitutionalEpochRoot,
        Decision decision
    ) external onlyCouncilSubmitter {
        if (epoch != lastEpoch + 1) revert InvalidEpoch();
        if (lastEpoch == 0) {
            if (previousEpochRoot != bytes32(0)) revert InvalidPreviousRoot();
        } else if (previousEpochRoot != lastEpochRoot) {
            revert InvalidPreviousRoot();
        }
        if (decision == Decision.None) revert InvalidDecision();
        if (
            proposalHash == bytes32(0) || candidateArtifactRoot == bytes32(0) ||
            invariantReportRoot == bytes32(0) || councilRoot == bytes32(0) ||
            canaryRoot == bytes32(0) || delayedOutcomeRoot == bytes32(0) ||
            rollbackRoot == bytes32(0) || bondPlanHash == bytes32(0) ||
            constitutionalEpochRoot == bytes32(0)
        ) revert MissingCommitment();

        upgrades[epoch] = UpgradeEpoch({
            epoch: epoch,
            previousEpochRoot: previousEpochRoot,
            proposalHash: proposalHash,
            candidateArtifactRoot: candidateArtifactRoot,
            invariantReportRoot: invariantReportRoot,
            councilRoot: councilRoot,
            canaryRoot: canaryRoot,
            delayedOutcomeRoot: delayedOutcomeRoot,
            rollbackRoot: rollbackRoot,
            bondPlanHash: bondPlanHash,
            constitutionalEpochRoot: constitutionalEpochRoot,
            decision: decision,
            createdAt: uint64(block.timestamp),
            submitter: msg.sender
        });
        lastEpoch = epoch;
        lastEpochRoot = constitutionalEpochRoot;
        emit UpgradeRecorded(epoch, constitutionalEpochRoot, previousEpochRoot, decision, proposalHash, msg.sender);
    }
}
