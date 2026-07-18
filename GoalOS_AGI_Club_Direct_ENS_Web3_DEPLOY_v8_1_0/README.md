# GoalOS Autonomous Money Machine Ω — AGI Club Direct-ENS Web3 Edition v8.1.0

This release replaces the registry-blocked v7 activation path with the direct ownership method already proven in the AGIJobManager interface:

```text
Connect MetaMask
→ enter one direct club label (for example: elite)
→ verify elite.club.agi.eth against Ethereum Mainnet
→ support Registry or Name Wrapper ownership and expiry
→ sign one readable gasless message
→ activate the complete Club Edition
```

## What a member needs

- MetaMask
- Ethereum Mainnet
- Current ownership of one direct `*.club.agi.eth` name

No GoalOS account, password, access-registry contract, activation transaction, gas, token approval, payment, or asset transfer is required.

## Start

Open `index.html` on an HTTPS GitHub Pages or IPFS gateway deployment. For the complete operator and deployment guide, open `START_HERE.html`.

## Important static-Web3 boundary

This edition gates the normal member experience and revalidates current ENS ownership. Publicly hosted JavaScript remains inspectable and can be saved after legitimate delivery. Static access control is entitlement gating, not confidential DRM.

## Production gates

Before announcing production activation, complete:

1. independent code and dependency review;
2. counsel/privacy review;
3. one controlled unwrapped direct name test;
4. one controlled wrapped direct name test, including expiry;
5. transfer-to-new-wallet test;
6. account/network/session-change test;
7. GitHub Pages and independent IPFS-gateway retrieval;
8. legal/version acceptance test;
9. incident and rollback rehearsal.

See `deploy/LIVE_MAINNET_ACCEPTANCE_CHECKLIST.md`.

## Release status

Automated packaging, CSP reconciliation, browser QA, mocked Registry/Name Wrapper tests, link validation, deterministic IPFS CAR generation, and manifest/checksum generation are included in v8.1.0.

The final earned designation remains **deployment-ready production release candidate** until the operator completes controlled-wallet Ethereum Mainnet acceptance and obtains independent counsel and security review. See `EXTERNAL_ACCEPTANCE_STATUS.md`.
