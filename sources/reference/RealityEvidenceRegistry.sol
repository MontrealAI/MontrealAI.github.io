// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Reference registry for public commitments produced by the GoalOS Independent Reality Layer.
/// @dev This contract stores commitments only. It does not authenticate underlying field evidence,
///      perform token settlement, or grant professional, legal, engineering, or operational authority.
contract RealityEvidenceRegistry {
    enum Status { None, Candidate, Admitted, Repaired, Rejected, Revoked }

    struct CampaignEpoch {
        bytes32 campaignId;
        uint64 epoch;
        bytes32 previousRoot;
        bytes32 preregistrationHash;
        bytes32 evidenceRoot;
        bytes32 institutionalProofRoot;
        bytes32 chronicleRoot;
        bytes32 skillRoot;
        bytes32 graphRoot;
        Status status;
        uint64 committedAt;
    }

    address public immutable administrator;
    mapping(bytes32 => CampaignEpoch[]) private _epochs;
    mapping(bytes32 => mapping(address => bytes32)) public institutionAttestations;

    event RealityEpochCommitted(bytes32 indexed campaignId, uint64 indexed epoch, bytes32 indexed graphRoot, Status status);
    event InstitutionAttested(bytes32 indexed campaignId, address indexed institution, bytes32 attestationHash);
    event RealityEpochRevoked(bytes32 indexed campaignId, uint64 indexed epoch, bytes32 reasonHash);

    error NotAdministrator();
    error InvalidEpoch();
    error InvalidPreviousRoot();
    error UnknownEpoch();

    modifier onlyAdministrator() {
        if (msg.sender != administrator) revert NotAdministrator();
        _;
    }

    constructor(address admin) {
        administrator = admin == address(0) ? msg.sender : admin;
    }

    function commitEpoch(
        bytes32 campaignId,
        uint64 epoch,
        bytes32 previousRoot,
        bytes32 preregistrationHash,
        bytes32 evidenceRoot,
        bytes32 institutionalProofRoot,
        bytes32 chronicleRoot,
        bytes32 skillRoot,
        bytes32 graphRoot,
        Status status
    ) external onlyAdministrator {
        CampaignEpoch[] storage list = _epochs[campaignId];
        if (epoch != list.length + 1) revert InvalidEpoch();
        if (epoch == 1) {
            if (previousRoot != bytes32(0)) revert InvalidPreviousRoot();
        } else if (previousRoot != list[list.length - 1].graphRoot) {
            revert InvalidPreviousRoot();
        }
        list.push(CampaignEpoch({
            campaignId: campaignId,
            epoch: epoch,
            previousRoot: previousRoot,
            preregistrationHash: preregistrationHash,
            evidenceRoot: evidenceRoot,
            institutionalProofRoot: institutionalProofRoot,
            chronicleRoot: chronicleRoot,
            skillRoot: skillRoot,
            graphRoot: graphRoot,
            status: status,
            committedAt: uint64(block.timestamp)
        }));
        emit RealityEpochCommitted(campaignId, epoch, graphRoot, status);
    }

    function attest(bytes32 campaignId, bytes32 attestationHash) external {
        institutionAttestations[campaignId][msg.sender] = attestationHash;
        emit InstitutionAttested(campaignId, msg.sender, attestationHash);
    }

    function revoke(bytes32 campaignId, uint64 epoch, bytes32 reasonHash) external onlyAdministrator {
        CampaignEpoch[] storage list = _epochs[campaignId];
        if (epoch == 0 || epoch > list.length) revert UnknownEpoch();
        list[epoch - 1].status = Status.Revoked;
        emit RealityEpochRevoked(campaignId, epoch, reasonHash);
    }

    function epochCount(bytes32 campaignId) external view returns (uint256) {
        return _epochs[campaignId].length;
    }

    function getEpoch(bytes32 campaignId, uint64 epoch) external view returns (CampaignEpoch memory) {
        CampaignEpoch[] storage list = _epochs[campaignId];
        if (epoch == 0 || epoch > list.length) revert UnknownEpoch();
        return list[epoch - 1];
    }
}
