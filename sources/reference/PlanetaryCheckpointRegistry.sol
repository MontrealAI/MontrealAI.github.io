// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Reference registry for batched planetary proof-fabric checkpoints.
/// @dev Stores only compact commitments. Raw evidence and private intelligence remain off-chain.
contract PlanetaryCheckpointRegistry {
    struct Checkpoint {
        uint64 epoch;
        bytes32 previousRoot;
        bytes32 zonesRoot;
        bytes32 topologyHash;
        bytes32 policyHash;
        bytes32 planetaryRoot;
        uint64 createdAt;
        address submitter;
    }

    address public owner;
    uint64 public lastEpoch;
    bytes32 public lastRoot;
    mapping(address => bool) public authorizedSubmitter;
    mapping(uint64 => Checkpoint) public checkpoints;

    event SubmitterAuthorized(address indexed submitter, bool authorized);
    event PlanetaryCheckpointSubmitted(uint64 indexed epoch, bytes32 indexed planetaryRoot, bytes32 previousRoot, bytes32 zonesRoot, bytes32 topologyHash, bytes32 policyHash, address submitter);

    error NotOwner();
    error NotAuthorized();
    error InvalidEpoch();
    error InvalidPreviousRoot();
    error EmptyRoot();

    constructor(address initialOwner) {
        owner = initialOwner == address(0) ? msg.sender : initialOwner;
        authorizedSubmitter[owner] = true;
    }

    modifier onlyOwner() { if (msg.sender != owner) revert NotOwner(); _; }
    modifier onlySubmitter() { if (!authorizedSubmitter[msg.sender]) revert NotAuthorized(); _; }

    function setSubmitter(address submitter, bool authorized) external onlyOwner {
        authorizedSubmitter[submitter] = authorized;
        emit SubmitterAuthorized(submitter, authorized);
    }

    function transferOwnership(address nextOwner) external onlyOwner {
        require(nextOwner != address(0), "zero owner");
        owner = nextOwner;
        authorizedSubmitter[nextOwner] = true;
    }

    function submitCheckpoint(
        uint64 epoch,
        bytes32 previousRoot,
        bytes32 zonesRoot,
        bytes32 topologyHash,
        bytes32 policyHash,
        bytes32 planetaryRoot
    ) external onlySubmitter {
        if (planetaryRoot == bytes32(0) || zonesRoot == bytes32(0)) revert EmptyRoot();
        if (epoch != lastEpoch + 1) revert InvalidEpoch();
        if (lastEpoch == 0) {
            if (previousRoot != bytes32(0)) revert InvalidPreviousRoot();
        } else if (previousRoot != lastRoot) {
            revert InvalidPreviousRoot();
        }
        checkpoints[epoch] = Checkpoint({
            epoch: epoch,
            previousRoot: previousRoot,
            zonesRoot: zonesRoot,
            topologyHash: topologyHash,
            policyHash: policyHash,
            planetaryRoot: planetaryRoot,
            createdAt: uint64(block.timestamp),
            submitter: msg.sender
        });
        lastEpoch = epoch;
        lastRoot = planetaryRoot;
        emit PlanetaryCheckpointSubmitted(epoch, planetaryRoot, previousRoot, zonesRoot, topologyHash, policyHash, msg.sender);
    }
}
