// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/// @title UserResponsibilityReceiptRegistryV4
/// @notice Stores minimum commitments proving that an account accepted a specific terms/policy version.
/// @dev A receipt is evidence only. It does not transfer mandatory liability, provide legal clearance,
///      create tax treatment, or make software/$AGIALPHA a legal person.
contract UserResponsibilityReceiptRegistryV4 {
    struct Receipt {
        bytes32 termsHash;
        bytes32 policyHash;
        bytes32 roleHash;
        bytes32 receiptHash;
        uint64 acceptedAt;
        uint64 expiresAt;
        bool revoked;
    }
    mapping(address => Receipt) public receipts;
    event ResponsibilityReceiptCommitted(address indexed account, bytes32 indexed termsHash, bytes32 indexed roleHash, bytes32 policyHash, bytes32 receiptHash, uint64 acceptedAt, uint64 expiresAt);
    event ResponsibilityReceiptRevoked(address indexed account, bytes32 indexed receiptHash);

    function commit(bytes32 termsHash, bytes32 policyHash, bytes32 roleHash, bytes32 receiptHash, uint64 expiresAt) external {
        require(termsHash != bytes32(0) && policyHash != bytes32(0) && roleHash != bytes32(0) && receiptHash != bytes32(0), "zero commitment");
        require(expiresAt > block.timestamp, "expired");
        receipts[msg.sender] = Receipt(termsHash, policyHash, roleHash, receiptHash, uint64(block.timestamp), expiresAt, false);
        emit ResponsibilityReceiptCommitted(msg.sender, termsHash, roleHash, policyHash, receiptHash, uint64(block.timestamp), expiresAt);
    }
    function revoke() external {
        Receipt storage r = receipts[msg.sender];
        require(r.receiptHash != bytes32(0), "missing");
        r.revoked = true;
        emit ResponsibilityReceiptRevoked(msg.sender, r.receiptHash);
    }
    function valid(address account, bytes32 termsHash, bytes32 policyHash, bytes32 roleHash) external view returns (bool) {
        Receipt memory r = receipts[account];
        return !r.revoked && r.expiresAt >= block.timestamp && r.termsHash == termsHash && r.policyHash == policyHash && r.roleHash == roleHash;
    }
}
