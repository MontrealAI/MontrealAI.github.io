# Security and Access

## Qualifying routes

- Current direct owner of exactly one single-label `*.club.agi.eth` name.
- Current direct holder of at least 1,000,000 official AGIALPHA in the connected Ethereum Mainnet wallet.

## Explicit exclusions

The gate does not count:

- resolver control;
- delegated control;
- operator approval;
- former ownership;
- beneficial or custodial claims;
- balances in another wallet;
- balances held through staking, escrow, lending, bridges or contracts;
- screenshots or signed statements about ownership.

## Session

The wallet signs an origin-bound receipt with a nonce, block number, issue time, expiry, route and `authorityCreated: NONE`. The client revalidates on focus, account changes, network changes and every two minutes.

## No access transaction

Access never asks for token approval, transfer, payment, staking, locking, burning, deposit or custody.

## GitHub Pages limitation

A static deployment cannot make its source files confidential. To make v3.0.0 genuinely exclusive:

1. host only the access shell publicly;
2. place the application on a private origin;
3. deploy `access-gateway/`;
4. configure `SESSION_SECRET`, `PRIVATE_ORIGIN_SECRET`, `PRIVATE_APP_ORIGIN` and the allowed public origin;
5. proxy `/app/*` only after server-side signature and current-chain revalidation.

## Access is not authority

Holding AGIALPHA or an AGI Club name qualifies a user to open the product. It grants no job role, validator status, mission capability, proof standing, admission power or Authority Envelope.
