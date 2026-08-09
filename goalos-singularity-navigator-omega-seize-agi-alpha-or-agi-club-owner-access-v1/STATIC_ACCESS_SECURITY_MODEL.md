# Static Access and Succession Security Model
## GoalOS Singularity Navigator Ω + SEIZE — v3.4.0-SN5-BI2-VS1

## Qualifying access routes

1. **AGI Club Owner Access:** current direct ownership of one exact single-label `label.club.agi.eth` name on Ethereum Mainnet.
2. **AGIALPHA Balance-Qualified Access:** current direct balance of at least `1,000,000` official AGIALPHA at `0xA61a3B3a130a9c20768EEBF97E21515A6046a1fA` on Ethereum Mainnet.

The connected wallet must satisfy one route. Balances are not aggregated across wallets. Delegation, resolver control, approvals, managers, beneficial-ownership assertions, former ownership, other networks, unrelated wrappers, liquidity positions and exchange account statements do not qualify in the public release.

## Verification controls

- Ethereum network: Mainnet only, chain ID `1`.
- ENS Registry: `0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e`.
- Official Mainnet Name Wrapper: `0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401`.
- AGIALPHA raw minimum: `1000000000000000000000000` base units.
- Token route: read-only `eth_getCode` and `balanceOf`; no approval or transaction method.
- Signature: EIP-712 preferred; domain-bound `personal_sign` fallback.
- Session: 30-minute absolute limit; 10-minute inactivity lock; five-minute revalidation; focus, account, network, disconnect and pre-export rechecks.
- Failure mode: uncertainty fails closed and relocks the institution.

## Public-static boundary

GitHub Pages serves client-side source publicly. The gate provides current eligibility verification, signed provenance, conditional licensing and session controls—not confidential server-side DRM or regulated identity certification. A client-generated receipt must not be treated as authoritative outside the public application without independent chain validation.

## SEIZE authority boundary

SEIZE may generate a Successor Book, underwrite proof, freeze a challenger, compile bounded formation jobs and produce a fresh-work scorecard. It must not self-admit, self-install, expand its own authority, execute an external transaction or allocate capital. The public release keeps `production_release_enabled: false`. Accountable human admission and deployment-specific authority remain separate.
