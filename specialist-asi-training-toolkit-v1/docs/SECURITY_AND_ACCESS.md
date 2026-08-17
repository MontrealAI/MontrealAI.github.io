# Security and Access Boundary

## Public access gate

The official static interface reads current Ethereum Mainnet state and verifies either exact direct AGI Club ownership or the connected wallet's direct AGIALPHA balance. The wallet signs a domain-bound, expiring access receipt that explicitly creates **no authority**.

The interface never requests approval, transfer, payment, staking, locking, burning, deposit, custody or transaction execution.

## Relocking

Access is revalidated on focus, account change, network change and a five-minute cadence. It expires after 30 minutes and locks after 20 minutes of inactivity. Any uncertain state fails closed.

## Static-host boundary

GitHub Pages serves public client code. The gate is an official eligibility interface, not confidential DRM. Confidential customer evidence, protected proof, secrets, release credentials and operational authority require private server-side infrastructure and independent custody.

## Local backend

The reference service binds to loopback by default. CORS is restricted to declared origins. Request bodies are bounded. The service emits security headers and creates no production authority.

## Independent proof

Formation and proof are separate. The exact release is frozen before proof; protected proof executes without learning; signed adjudication reports `authority_created: NONE`. Real admission remains a separate accountable decision.
