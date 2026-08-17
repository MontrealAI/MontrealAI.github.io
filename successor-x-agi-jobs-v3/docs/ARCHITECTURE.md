# v3.0.0 Architecture

## Control relation

```text
Mission Constitution
  → Typed AGI Job Graph
  → Formation Gym / Foundry
  → Exact Candidate Release
  → Separate Fresh Proof
  → Specialist ASI eligibility
  → Accountable Admission
  → Signed Authority Envelope
  → External Authority Gateway
  → Monitored operation
  → Impairment / rollback / requalification
  → Chronicle
```

No arrow may be silently bypassed. Work and evidence may compose. Authority does not.

## Role separation

The generated reference cycle uses distinct Ed25519 identities for:

- independent verifier;
- accountable admitter;
- authority governor;
- Chronicle custodian.

The producer is a separate non-signing formation role. Public keys are distributed; private keys are not.

## Public versus protected

The GitHub Pages bundle contains only non-confidential reference evidence. The browser worker rehearsal is intentionally inspectable. Production protected cases, scorer internals, release credentials and confidential mission evidence belong behind the included access gateway and independent proof plane.

## Event-sourced state

`evidence/reference-cycle/chronicle-ledger.json` links every accepted transition by SHA-256. Current state is reconstructed from the event stream, not from a mutable narrative summary.
