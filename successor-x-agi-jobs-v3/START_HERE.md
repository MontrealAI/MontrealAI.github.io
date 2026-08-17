# START HERE — Successor Ω × AGI Jobs v3.0.0

## Fastest review

1. Open `index.html` through HTTPS or a local web server.
2. Qualify with either:
   - current direct ownership of one exact single-label `*.club.agi.eth` name; or
   - a current direct balance of at least **1,000,000 official AGIALPHA** in the connected wallet on Ethereum Mainnet.
3. Start at **Cycle Command**.
4. Inspect **Actual AGI Jobs** for the live Prime ledger.
5. Follow the cycle through Constitution, Protected Execution, Fresh Proof, Admission, Authority, Requalification and Chronicle.

For local QA only:

```bash
python3 -m http.server 8080
# open http://127.0.0.1:8080/?qa=1
```

The local QA route is intentionally non-qualifying and is rejected by the server-side access gateway.


## Genuine exclusive deployment

Use the packaged `protected-deployment/` architecture:

1. Deploy `protected-deployment/public-gate/` to GitHub Pages.
2. Deploy `protected-deployment/private-origin/` with the private application Assets binding and `PRIVATE_ORIGIN_SECRET`.
3. Deploy `protected-deployment/access-gateway/` with `SESSION_SECRET`, the same `PRIVATE_ORIGIN_SECRET`, `PRIVATE_APP_ORIGIN`, `ALLOWED_ORIGINS` and an Ethereum Mainnet RPC endpoint.
4. Set `protectedGatewayEndpoint` in the public gate configuration.

Only the access shell is public. The application payload is served after wallet-signature verification and a fresh server-side recheck of direct AGI Club ownership or direct AGIALPHA balance.

## Reproduce the reference cycle

```bash
cd toolkit
PYTHONPATH="$PWD" python3 -m goalos_alpha_foundry cycle \
  --mission grid_resilience \
  --seed 20260817 \
  --formation 220 \
  --fresh 160 \
  --transfer 80 \
  --out ../evidence/runtime-raw/grid_resilience_initial.json

cd ..
node scripts/build-reference-cycle.mjs
python3 scripts/validate-schemas.py
node authority-gateway/test.mjs
```

## Exact status

v3.0.0 is the first **executable proof-bearing reference succession cycle** in this release lineage. It combines real Mainnet AGI Job connectivity with a deterministic synthetic mission runtime, exact candidate releases, separate Fresh Proof, distinct signatures, accountable reference admission, a signed Authority Envelope, actual action denials, impairment, rollback and a separately proved requalification release.

It does **not** claim that a customer system has earned production Specialist ASI, that universal ASI exists, or that the package controls a physical grid or any consequential external system.
