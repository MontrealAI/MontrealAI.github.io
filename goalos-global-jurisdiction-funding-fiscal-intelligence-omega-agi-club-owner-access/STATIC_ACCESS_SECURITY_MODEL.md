# Static owner-access security model

Release: v10.0.0-GHP1 (2026-07-24)

- Exact one-label `label.club.agi.eth` only.
- Ethereum Mainnet only.
- ENS Registry direct owner accepted.
- Only the official Ethereum Mainnet ENS Name Wrapper (`0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401`) is recognized; direct ERC-1155 ownership is accepted only when `ownerOf` and `getData.owner` agree and any reported expiry is current.
- Approvals, delegates, resolvers, managers, operators and nested names rejected.
- Domain-bound signed statement; no transaction.
- Ownership rechecked after signature, every five minutes, on focus, on session restoration and before owner-center exports.
- Account, chain or provider disconnect relocks immediately.
- Receipt stored in `sessionStorage`; no server collection.
- CSP uses `connect-src 'none'`; wallet-provider RPC remains extension-mediated.
- Static source is public and not confidential.
