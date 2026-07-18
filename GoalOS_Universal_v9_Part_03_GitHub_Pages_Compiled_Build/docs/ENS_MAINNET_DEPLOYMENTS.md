# ENS Ethereum Mainnet Deployment Policy

GoalOS verifies direct `*.club.agi.eth` ownership against Ethereum Mainnet only.

## Default production addresses

- ENS Registry: `0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e`
- Official ENS Ethereum Mainnet Name Wrapper: `0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401`

The Sepolia Name Wrapper (`0x0635513f179D50A207757E05759CbD106d7dFcE8`) is deliberately rejected by production preflight and must never be used in the Mainnet configuration.

## Optional additional Mainnet wrappers

No historical or custom wrapper is trusted by default. An operator may add another wrapper only after independently confirming:

1. It is deployed on Ethereum Mainnet.
2. The ENS Registry owner of the direct name points to that exact wrapper.
3. Its `ownerOf` and `getData(...).owner` semantics are compatible.
4. Its expiry semantics are understood and fail closed.
5. A controlled owner, transfer, wrong-account, and expiry acceptance matrix passes.
6. The exact address and rationale are recorded in the production acceptance report.

## Fail-closed rule

A configured wrapped membership activates access only when all of the following hold at one coherent Ethereum block:

1. `ENSRegistry.owner(namehash)` equals that exact configured wrapper address.
2. `NameWrapper.ownerOf(tokenId)` equals the selected MetaMask account.
3. `NameWrapper.getData(tokenId).owner` equals the same account.
4. `ownerOf` and `getData.owner` agree.
5. The expiry is non-zero and either equals the explicit uint64 no-expiry sentinel or is later than the current time.

Resolver records, reverse records, approvals, operators, text records, and self-asserted metadata never activate membership.
