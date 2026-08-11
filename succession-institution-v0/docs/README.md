# GoalOS v7.0.0-UVSI2

## Executable Verified Succession Institution

This release is a new, from-scratch GitHub Pages implementation of **GoalOS Singularity Navigator Ω + SEIZE: Gym, Specialist ASI & Mission-Sovereign Successor Ω**.

It operationalizes:

`Successor Manifold → Mission Advantage Gradient → SEIZE Underwriting → Bounded AGI Jobs → Sovereign Mission Gym → Fresh Proof → Specialist ASI → Mission-Sovereign Successor Ω → Chronicle → Requalification`

The public application is a gated, offline-first PWA with an autonomous deterministic engine. A secure optional AI backend executes the methodology through the OpenAI Responses API without exposing an API key in the browser. Before every AI run, the Cloudflare Worker verifies the signed access receipt and rechecks current Mainnet eligibility.

## Access rule

Access is granted when the connected Ethereum Mainnet wallet is either:

1. the current direct owner of one exact single-label `label.club.agi.eth` name; or
2. the current direct holder of at least 1,000,000 official AGIALPHA at `0xA61a3B3a130a9c20768EEBF97E21515A6046a1fA`.

The gate makes read-only calls and a domain-bound signature. It requests no approval, transfer, payment, staking, locking, burning, deposit or custody. The signed receipt records `authorityCreated = NONE`.

## Main editions

- **GitHub Pages Strict Edition**: deploy as-is; offline engine, access gate, PWA, AI Studio bridge.
- **Secure AI Edition**: GitHub Pages frontend plus Cloudflare Worker.
- **Local Secure Edition**: Python server prompts for the API key and keeps it only in process memory.
- **Standalone Edition**: one HTML file for local review and preservation.

See `DEPLOYMENT_GUIDE.md` and `USER_GUIDE.md`.
