# Successor Ω × AGI Jobs — Verified Succession Institution v3.0.0

**The First Executable Proof-Bearing Reference Succession Cycle**

> AGI Jobs manufacture candidacy. Fresh proof may establish Specialist ASI. Only accountable admission creates Successor Ω. Chronicle preserves what survives reality and prepares the successor to the successor.

## What changed from v2.0.0

v2.0.0 was an advanced interactive institutional twin. v3.0.0 adds executable and independently verifiable state transitions:

- live Ethereum Mainnet reads from actual AGI Jobs contracts;
- optional simulation-first Mainnet job writes;
- a deterministic Mission Gym runtime with two complete runs;
- exact candidate freeze and cryptographic release identity;
- a separately executed non-learning Fresh Proof plane;
- Ed25519 signatures from distinct verifier, admitter, governor and Chronicle identities;
- an accountable reference Admission Record;
- an externally enforced, signed Authority Envelope;
- action-level ALLOW / DENY decisions and append-only logs;
- material impairment and automatic authority contraction;
- tested rollback to a known-good deterministic fallback;
- a second exact release with new proof, new admission and new authority;
- an append-only Chronicle hash chain;
- current-holder access gating equivalent to the GoalOS UVSI3 release pattern;
- a server-side edge gateway for genuine protected application delivery.

## Access constitution

The public application opens only when the connected wallet is one of:

1. the **current direct owner** of exactly one single-label `label.club.agi.eth` name; or
2. the **current direct holder of at least 1,000,000 official AGIALPHA** on Ethereum Mainnet.

Official AGIALPHA:

```text
0xA61a3B3a130a9c20768EEBF97E21515A6046a1fA
```

Only the connected wallet's current direct state counts. No balance aggregation, delegation, resolver control, approval, operator status, former ownership, staking position, custody claim or beneficial-interest claim qualifies.

Access verification requests no token approval, transfer, payment, staking, locking, burning, deposit or custody. The signed receipt always states:

```text
authorityCreated: NONE
```

A client-side GitHub Pages gate is an experience control, not confidentiality. Use `access-gateway/` to proxy the application from a private origin after server-side signature and current-eligibility revalidation.


## Distribution modes

The release ships three deliberately different deployment modes:

1. **Protected deployment — recommended for genuine exclusivity.** `protected-deployment/` separates a public GitHub Pages access shell, an edge eligibility gateway, a secret-protected private origin and the private application payload. The protected payload is never placed in the public Pages bundle.
2. **Client-gated GitHub Pages reference.** The complete static application can be hosted on GitHub Pages for evaluation, with current-holder verification before the interface opens. Because static assets are public by construction, this mode is not confidentiality.
3. **Portable offline review.** `Successor_Omega_x_AGI_Jobs_v3_0_0_Portable.html` contains the complete reference interface in one file. It is intended for inspection and demonstrations, not protected delivery.

The protected path rechecks current direct eligibility server-side, issues an `HttpOnly`, `Secure`, `SameSite=Strict` session cookie, and proxies every private application request through the edge gateway.

## Actual AGI Jobs integration

The interface is configured for:

| Object | Ethereum Mainnet address |
| --- | --- |
| Official AGIALPHA | `0xA61a3B3a130a9c20768EEBF97E21515A6046a1fA` |
| AGIJobManagerPrime | `0xF8fc6572098DDcAc4560E17cA4A683DF30ea993e` |
| AGIJobDiscoveryPrime | `0xd5EF1dde7Ac60488f697ff2A7967a52172A78F29` |
| ENSJobPagesPrime | `0x703011EF1C6E4277587eFe150e6cd74cA18F0069` |
| AGIJobManager Genesis | `0xB3AAeb69b630f0299791679c063d68d6687481d1` |

`mainnet.js` discovers the deployed code range, reads contract logs through the connected wallet's provider and reconstructs job state without treating a hosted indexer as authoritative.

The optional transaction workbench can:

- approve an exact AGIALPHA payout amount;
- create a job on AGIJobManagerPrime;
- request job completion;
- finalize, dispute, expire or cancel a job.

Transaction mode is off by default. Every write is estimated, described and confirmed before the wallet is opened.

## The proof-bearing cycle

The packaged reference mission is **Grid Storage Resilience**. It is synthetic, deterministic and non-physical.

### Initial release

```text
Release: grid_resilience:behavior_clone:c55755ec68c61aaa
Formation episodes: 220
Protected fresh cases: 160
Transfer cases: 80
Champion: Behaviour Cloning
Reference: Deterministic Rules
Fresh-proof verdict: SYNTHETIC_REFERENCE_PROMOTION_ELIGIBLE
Authority: A1_RECOMMEND_ONLY
External action limit: 0
Spend limit: 0
```

### Impairment

A material telemetry and environment-schema change invalidates the prior fresh-proof and transfer claims. The system:

1. contracts authority from `A1_RECOMMEND_ONLY` to `A0_READ_ONLY`;
2. denies recommendation and external-action requests;
3. preserves read-only observability;
4. activates the deterministic rules fallback;
5. opens the requalification job.

### Requalified release

```text
Release: grid_resilience:behavior_clone:a4d8c4d065ab85e4
Proof inherited: false
Authority inherited: false
New Fresh Proof: required and signed
New Admission: required and signed
New Authority: required and signed
Verdict: REQUALIFICATION_TEST_PASS
```

## Six operational planes

1. **Public orientation and access** — qualification, documentation, Mainnet job ledger and non-confidential records.
2. **Mission Constitution and Job Graph** — typed work, dependencies, evidence, prohibitions, budget, rollback and lifecycle closure.
3. **Protected formation and execution** — candidate search and bounded experience without proof credentials.
4. **Independent Fresh Proof** — separately held cases, exact release, zero proof-time learning and signed verdict.
5. **Admission and Authority** — distinct human/institutional decision and external capability gateway.
6. **Chronicle and Requalification** — event-sourced memory, impairment, rollback, renewed proof and the successor to the successor.

## Repository map

```text
index.html / app.js / styles.css    public institutional interface
access.js                          direct-owner / direct-balance client gate
mainnet.js                         live AGI Jobs reader + transaction workbench
data.js                            packaged canonical reference records
assets/workers/                    one-way browser proof rehearsal
actual-jobs/                       contract and event configuration
access-gateway/                    server-side current-holder protected delivery
protected-deployment/             public gate + edge gateway + secret private origin + private app
authority-gateway/                 signed-envelope enforcement service and tests
independent-proof-plane/           separately deployable proof-custody package
toolkit/                           executable GoalOS Alpha Foundry runtime
schemas/                           canonical JSON Schemas
scripts/                           build, validation, QA and packaging
research/                          governing papers
evidence/                          raw runs, signed records, reports and manifests
```

## Reproducibility

```bash
node scripts/build-reference-cycle.mjs
node scripts/build-data.mjs
python3 scripts/validate-schemas.py
node authority-gateway/test.mjs
PYTHONPATH=toolkit python3 -m unittest discover -s toolkit/tests -p 'test_*.py'
```

The release manifest and `SHA256SUMS` identify every distributed artifact.

## Claim boundary

This package establishes that the reference mechanism is executable and inspectable: jobs can be connected, candidates can be frozen, a separate proof process can sign a verdict, admission can remain distinct, signed authority can be enforced, prohibited actions can be denied, evidence impairment can contract authority, rollback can occur and a descendant can be requalified without inheriting proof or authority.

It does not establish production mission dominance, customer value, professional fitness, universal ASI or authority over an external organization or physical system. A production Successor Ω requires rights-cleared mission data, protected infrastructure, independent proof custody, accountable corporate authority, secure operational adapters, monitoring, incident response and tested rollback.
