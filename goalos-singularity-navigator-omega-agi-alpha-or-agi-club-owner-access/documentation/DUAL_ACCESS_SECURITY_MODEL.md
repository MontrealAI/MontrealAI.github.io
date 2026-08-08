# GoalOS Dual-Access Security Model — v3.3.0-SN5-BI2

## Canonical rule

**Access = Current AGI Club direct owner OR Current direct holder of at least 1,000,000 official AGIALPHA on Ethereum Mainnet.**

Qualifying contract: `0xA61a3B3a130a9c20768EEBF97E21515A6046a1fA`  
Chain: Ethereum Mainnet (`1`)  
Raw minimum: `1000000000000000000000000` base units

## AGIALPHA route

The browser requests accounts, verifies Ethereum Mainnet, reads contract bytecode, calls ERC-20 `balanceOf(address)`, reads the verification block, requests a domain-bound temporary signature, then repeats account, network and balance checks before unlocking. It never calls `approve`, `transfer`, `transferFrom`, `permit`, staking, deposit, burn, bridge, swap, or transaction-sending methods.

The route is fail-closed. It relocks on a balance below the threshold, account or network change, wallet disconnect, signature failure, verification uncertainty, a 30-minute absolute expiry, ten minutes of inactivity, or failed pre-export revalidation. Revalidation occurs every five minutes, when the page regains focus or visibility, and before protected evidence exports.

## Boundary

The AGIALPHA receipt proves only that the connected address met the stated balance condition at recorded checks and signed a temporary local receipt. It creates no AGI Club membership, ENS identity, validator or governance status, professional or organizational authority, financial right, redemption claim, permanent licence, or assurance about token value or liquidity.

Because the release is public client-side software, this gate is not unbypassable DRM and does not make public source confidential. Any downstream service relying on a receipt must independently verify the signature, current chain state, release, origin, legal digest, session validity and applicable authority.
