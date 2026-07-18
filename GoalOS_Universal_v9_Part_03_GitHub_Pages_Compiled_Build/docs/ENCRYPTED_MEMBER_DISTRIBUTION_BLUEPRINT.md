# Encrypted Member Distribution Blueprint

The GitHub Pages and ordinary IPFS edition verifies Web3 entitlement and activates the intended member interface. It does not make publicly delivered static bytes secret.

For cryptographic confidentiality, deploy member-only payloads as encrypted objects and release the content key only after current direct `*.club.agi.eth` ownership is proved. A production implementation should include:

1. Client-side encryption before publication.
2. Immutable encrypted objects on IPFS or approved storage.
3. An independently reviewed key-release service or decentralized threshold network.
4. A live Ethereum Mainnet access condition that checks the exact namehash, Registry/Name Wrapper ownership, and wrapped expiry.
5. Key revocation or short-lived decryption grants after transfer or expiry.
6. No plaintext secrets, master keys, or privileged API credentials in GitHub Pages or IPFS files.
7. Independent smart-contract, cryptographic, privacy, and incident-response review.

This blueprint is not marked active in the release. Public static Web3 mode remains transparent by design. Enterprise confidential use should employ private hosting or the downloaded local workspace until an encrypted distribution layer is separately implemented and approved.
