# Technical architecture

## Public static plane

- GitHub Pages PWA
- dual-access wallet gate
- local project storage
- deterministic Mission Gym and Foundry
- Mission Pack export
- no external runtime library

## Protected opportunity plane

Recommended for confidential missions:

- organizational identity
- encrypted mission and economics workspace
- rights and data register
- Succession Constitution
- proof-capital decision

## Isolated Successor Foundry

- formation cases
- candidate architectures
- recursive improvement
- tools and simulations
- Bounded AGI Jobs

## Independent proof plane

- protected cases and seeds
- scorer internals
- exact candidate manifest
- critical gates
- Fresh-Successor scorecard
- Release Certificate

## Chronicle and authority plane

- admission record
- Authority Envelope
- evidence receipts
- realized value
- impairment, requalification and revocation
- Capital-to-Capacity

## Secure AI

The frontend sends no API key. The optional backend uses the OpenAI Responses API with strict structured output and `store: false`. It rejects unapproved origins and oversized requests, verifies the wallet signature, and rechecks current on-chain eligibility before every AI execution. A production paid deployment should additionally use organizational authentication or Cloudflare Access, durable rate limiting, spend controls and incident monitoring.
