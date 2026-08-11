# Security and access model

## Exact eligibility

`Access = CurrentDirectAGIClubOwner OR CurrentDirectAGIALPHAHolder>=1,000,000`

### AGI Club

The user enters one ASCII label. The app computes `namehash(label.club.agi.eth)`, reads the ENS Registry and, when wrapped, reads the effective owner and expiry from the NameWrapper. The connected wallet must be the current direct owner.

### AGIALPHA

The app performs a read-only ERC-20 `balanceOf(connectedWallet)` call on Ethereum Mainnet. Only the connected wallet’s direct current balance counts.

## Excluded claims

The gate does not aggregate wallets and does not accept delegated control, resolver control, allowances, former balances, exchange statements, wrapped assets, liquidity positions, other networks or beneficial claims.

## No transaction authority

The app never requests approval, transfer, deposit, payment, staking, locking, burning or custody. The access signature is a personal message and records `authorityCreated = NONE`.

## Fail-closed session

The session is bound to origin, account, chain, route, issue time, expiry and nonce. The app revalidates on focus, account change, chain change and a fixed interval. Eligibility loss, inactivity, uncertainty or expiry relocks the app.

## Static-gate boundary

A GitHub Pages gate controls browser presentation and creates a signed eligibility receipt. It is not confidential server-side DRM. Protect costly APIs and confidential evidence with authenticated server infrastructure.


## Secure AI backend revalidation

The Cloudflare Worker does not rely only on the browser gate. For every AI request it verifies the signed, origin-bound receipt, confirms the current expiry and Ethereum Mainnet context, and rechecks the qualifying on-chain route. Local-demo receipts are rejected by the remote Worker.
