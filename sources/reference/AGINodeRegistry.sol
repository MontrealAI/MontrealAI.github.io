// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IENSRegistryNode {
    function owner(bytes32 node) external view returns (address);
}

interface INameWrapperNode {
    function getData(uint256 id) external view returns (address owner, uint32 fuses, uint64 expiry);
}

/// @notice Reference registry that accepts only effective owners of direct children of node.agi.eth.
/// @dev This contract is not audited. ENSv2 or custom registries should be connected through a reviewed adapter.
contract AGINodeRegistry {
    uint32 internal constant PARENT_CANNOT_CONTROL = 1 << 16;

    struct NodeRecord {
        bytes32 node;
        bytes32 labelhash;
        address operator;
        bytes32 publicKeyHash;
        bytes32 regionHash;
        bytes32 capabilitiesHash;
        uint64 registeredAt;
        uint64 updatedAt;
        uint64 ensExpiry;
        uint32 ensFuses;
        uint32 reputationPpm;
        bool wrapped;
        bool active;
    }

    IENSRegistryNode public immutable ens;
    INameWrapperNode public immutable wrapper;
    bytes32 public immutable nodeAgiEthParent;
    address public owner;
    address public resolver;
    bool public requireEmancipated;

    mapping(bytes32 => NodeRecord) private nodes;
    mapping(address => bytes32) public nodeByOperator;

    event NodeRegistered(bytes32 indexed node, bytes32 indexed labelhash, address indexed operator, bytes32 publicKeyHash, bytes32 regionHash, bool wrapped);
    event NodeStatusChanged(bytes32 indexed node, bool active);
    event ReputationUpdated(bytes32 indexed node, uint32 reputationPpm);

    modifier onlyOwner() { require(msg.sender == owner, "NOT_OWNER"); _; }
    modifier onlyResolver() { require(msg.sender == resolver, "NOT_RESOLVER"); _; }

    constructor(address ensRegistry, address nameWrapper, bytes32 parentNode, address initialResolver, bool requirePcc) {
        require(ensRegistry != address(0) && nameWrapper != address(0) && parentNode != bytes32(0) && initialResolver != address(0), "ZERO_ADDRESS");
        ens = IENSRegistryNode(ensRegistry);
        wrapper = INameWrapperNode(nameWrapper);
        nodeAgiEthParent = parentNode;
        owner = msg.sender;
        resolver = initialResolver;
        requireEmancipated = requirePcc;
    }

    function setResolver(address nextResolver) external onlyOwner {
        require(nextResolver != address(0), "ZERO_ADDRESS");
        resolver = nextResolver;
    }

    function setRequireEmancipated(bool required) external onlyOwner {
        requireEmancipated = required;
    }

    function registerDirectNode(bytes32 labelhash, bytes32 publicKeyHash, bytes32 regionHash, bytes32 capabilitiesHash) external returns (bytes32 node) {
        require(labelhash != bytes32(0) && publicKeyHash != bytes32(0), "BAD_NODE");
        node = keccak256(abi.encodePacked(nodeAgiEthParent, labelhash));
        (address effectiveOwner, bool wrapped, uint32 fuses, uint64 expiry) = effectiveOwnerOf(node);
        require(effectiveOwner == msg.sender, "NOT_EFFECTIVE_ENS_OWNER");
        if (expiry != 0) require(block.timestamp < expiry, "ENS_NAME_EXPIRED");
        if (requireEmancipated) require((fuses & PARENT_CANNOT_CONTROL) != 0, "DIRECT_NAME_NOT_EMANCIPATED");
        require(nodeByOperator[msg.sender] == bytes32(0) || nodeByOperator[msg.sender] == node, "OPERATOR_ALREADY_BOUND");
        NodeRecord storage record = nodes[node];
        require(record.operator == address(0) || record.operator == msg.sender, "NODE_ALREADY_BOUND");
        if (record.registeredAt == 0) record.registeredAt = uint64(block.timestamp);
        record.node = node;
        record.labelhash = labelhash;
        record.operator = msg.sender;
        record.publicKeyHash = publicKeyHash;
        record.regionHash = regionHash;
        record.capabilitiesHash = capabilitiesHash;
        record.updatedAt = uint64(block.timestamp);
        record.ensExpiry = expiry;
        record.ensFuses = fuses;
        record.wrapped = wrapped;
        record.active = true;
        nodeByOperator[msg.sender] = node;
        emit NodeRegistered(node, labelhash, msg.sender, publicKeyHash, regionHash, wrapped);
    }

    function refreshENSState(bytes32 node) external {
        NodeRecord storage record = nodes[node];
        require(record.operator != address(0), "UNKNOWN_NODE");
        (address effectiveOwner, bool wrapped, uint32 fuses, uint64 expiry) = effectiveOwnerOf(node);
        record.active = effectiveOwner == record.operator && (expiry == 0 || block.timestamp < expiry) && (!requireEmancipated || (fuses & PARENT_CANNOT_CONTROL) != 0);
        record.wrapped = wrapped;
        record.ensFuses = fuses;
        record.ensExpiry = expiry;
        record.updatedAt = uint64(block.timestamp);
        emit NodeStatusChanged(node, record.active);
    }

    function setNodeActive(bytes32 node, bool active) external onlyResolver {
        require(nodes[node].operator != address(0), "UNKNOWN_NODE");
        nodes[node].active = active;
        nodes[node].updatedAt = uint64(block.timestamp);
        emit NodeStatusChanged(node, active);
    }

    function updateReputation(bytes32 node, uint32 reputationPpm) external onlyResolver {
        require(nodes[node].operator != address(0), "UNKNOWN_NODE");
        require(reputationPpm <= 1_000_000, "BAD_REPUTATION");
        nodes[node].reputationPpm = reputationPpm;
        nodes[node].updatedAt = uint64(block.timestamp);
        emit ReputationUpdated(node, reputationPpm);
    }

    function effectiveOwnerOf(bytes32 node) public view returns (address effectiveOwner, bool wrapped, uint32 fuses, uint64 expiry) {
        address registryOwner = ens.owner(node);
        if (registryOwner == address(wrapper)) {
            (effectiveOwner, fuses, expiry) = wrapper.getData(uint256(node));
            wrapped = true;
        } else {
            effectiveOwner = registryOwner;
            wrapped = false;
            fuses = 0;
            expiry = 0;
        }
    }

    function isActiveOperator(address operator) external view returns (bool) {
        bytes32 node = nodeByOperator[operator];
        if (node == bytes32(0)) return false;
        NodeRecord memory record = nodes[node];
        if (!record.active) return false;
        (address effectiveOwner,, uint32 fuses, uint64 expiry) = effectiveOwnerOf(node);
        if (effectiveOwner != operator) return false;
        if (expiry != 0 && block.timestamp >= expiry) return false;
        if (requireEmancipated && (fuses & PARENT_CANNOT_CONTROL) == 0) return false;
        return true;
    }

    function getNode(bytes32 node) external view returns (NodeRecord memory) {
        return nodes[node];
    }
}
