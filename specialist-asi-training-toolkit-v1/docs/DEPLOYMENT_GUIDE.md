# GoalOS v9.0.0-UVSI3 — Production Deployment Guide

## A. GitHub Pages public institution

1. Extract the `GoalOS_UVSI3_v9_0_0_GITHUB_PAGES.zip` package.
2. Upload its **contents** to the repository root, including `.github/` and `.nojekyll`.
3. Confirm that every extracted file is below 25 MiB.
4. In **Settings → Pages**, choose **GitHub Actions**.
5. Wait for the included workflow to deploy.
6. Visit `/reset.html` once when replacing a prior service-worker release.
7. Test both wallet routes using a wallet-enabled browser on Ethereum Mainnet.

The public application verifies current direct eligibility and creates a bounded signed access receipt. It is a public static control surface, not confidential DRM or a protected proof plane.

## B. Local synthetic Foundry

```bash
cd toolkit
./START_MAC.command       # macOS
./START_LINUX.sh          # Linux
# Windows: START_WINDOWS.bat
```

The service binds to `127.0.0.1:8788`. The GitHub Pages application can connect to the loopback service without exposing it publicly.

## C. Independent proof plane

Run the proof worker from `independent-proof-plane/` against an exact frozen cycle record. In real deployments, protected cases, scorer internals, signing credentials and release authority must remain physically and organizationally separate from formation.

## D. Production institution

A real deployment additionally requires:

- organizational identity and role separation;
- server-side authorization and least privilege;
- protected evidence custody and secrets management;
- immutable or append-only receipts and release signing;
- rights, privacy, security, professional and legal review;
- action gateways enforcing scope, spend, expiry and prohibited systems;
- monitoring, incident response, tested rollback and known-good fallback;
- independent realized-value calculation and dated requalification.

## E. Post-deployment acceptance

- `/index.html`, `config.js`, executable JS and JSON are network-first.
- `/reset.html` clears stale caches and service workers.
- account, network, eligibility loss, expiry and inactivity relock access.
- eligible AGI Club owner and qualifying AGIALPHA wallet succeed.
- ineligible wallet, wrong network, expired name and insufficient balance fail closed.
- no approval or transaction request appears.
