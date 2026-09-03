# MCAP-001 — Machine Civilization Addressability Protocol

**Category:** Open institutional specification  
**Version:** 0.1.0  
**Status:** Genesis publication  
**Root implementation:** `asi.eth`  
**Publisher:** MONTREAL.AI

## Abstract

MCAP-001 defines a minimal, resolver-independent architecture for making a Machine Civilization identifiable, discoverable, composable, auditable and capable of succession across changing models, agents, operators and infrastructure.

The protocol separates **addressability** from **legitimacy**. An ENS name can identify and route to an institution; it does not by itself prove capability, trustworthiness or authority. MCAP-001 therefore binds a persistent namespace to a public manifest, a constitutional Covenant, an identity and lineage registry, an append-only Chronicle, independent proof, admission state and bounded authority.

## 1. Core doctrine

A conforming implementation MUST preserve the following distinction:

> **Memory may cross. Authority must requalify.**

A successor MAY inherit identity, mission, memory, validated knowledge, tools, skills, obligations and Chronicle references.

A successor MUST NOT automatically inherit current proof, trust, reputation, permissions, privileged keys or operational authority.

## 2. Terminology

**Machine Civilization** — A governed intelligence institution capable of preserving mission, memory, accountability, ownership, proof and succession beyond the lifespan of any individual model, agent, operator or infrastructure provider.

**Successor** — An institutionally qualified continuation admitted after fresh proof against a declared mission and authority scope.

**Root** — A persistent human-readable and machine-resolvable namespace from which canonical institutional components can be discovered.

**Covenant** — Stable constitutional invariants that constrain every implementation and successor.

**Chronicle** — An append-only sequence of commitments to admissions, evidence, decisions, consequences, failures, repairs, revocations and succession events.

**Proof** — Evidence and evaluation produced under declared conditions by mechanisms sufficiently independent of the claimant and proportional to the value at risk.

**Admission** — The explicit transition by which a frozen candidate receives a defined institutional role and bounded authority.

**Authority** — Permission to affect systems, assets, people or environments. Authority is distinct from capability.

## 3. Root discovery

A conforming root MUST publish:

1. A canonical human interface.
2. A machine-readable Root Manifest.
3. A versioned Covenant.
4. A namespace registry.
5. A public status indicating which components are active, proposed, suspended or superseded.
6. A means to discover proof, verifier, admission and Chronicle state.

The Root Manifest MUST use a stable URL or content-addressed identifier and SHOULD be referenced from ENS text records.

## 4. Canonical namespace roles

MCAP-001 reserves the following semantic roles beneath a root such as `asi.eth`:

| Role | Example | Required semantics |
|---|---|---|
| Root | `asi.eth` | Discovery and namespace; no operational authority implied |
| Successor | `successor.asi.eth` | Current or nominated qualified continuation |
| Covenant | `covenant.asi.eth` | Stable constitutional invariants |
| Constitution | `constitution.asi.eth` | Versioned machine-readable rules and policy schemas |
| Registry | `registry.asi.eth` | Identities, missions and lineage relationships |
| Chronicle | `chronicle.asi.eth` | Append-only institutional commitments |
| Proof | `proof.asi.eth` | Proof requests, frozen candidates, evidence and results |
| Verifier | `verifier.asi.eth` | Independent verifier discovery |
| Admission | `admission.asi.eth` | Qualification, scope, expiry, suspension and revocation |
| Policy | `policy.asi.eth` | External policy constraints and authority ceilings |
| Rollback | `rollback.asi.eth` | Reversal and emergency recovery procedures |
| Treasury | `treasury.asi.eth` | Bounded resources, escrow and settlement |

An implementation MAY use other labels, but it MUST publish an unambiguous role mapping.

## 5. Separation of powers

A conforming institution MUST NOT permit one uncontrolled operational principal to simultaneously:

- produce the consequential work;
- alter the evidence used to evaluate that work;
- act as the sole final verifier;
- grant itself consequential authority;
- erase the Chronicle of the decision.

Control of `successor`, `proof`, `verifier`, `admission`, `chronicle` and `treasury` SHOULD be separated according to risk. Where one legal organization performs multiple roles, technical controls and public conflict disclosures MUST preserve meaningful independence.

## 6. Successor state machine

A successor record MUST expose one of the following states:

```text
NOMINATED
→ CANDIDATE
→ FROZEN
→ UNDER_PROOF
→ ADMITTED
→ ACTIVE
→ SUSPENDED | SUPERSEDED | REVOKED
```

Required transitions:

- A candidate MUST be frozen before final independent proof begins.
- Evidence generated after freeze MUST be attributable to the relevant proof process.
- Admission MUST reference the candidate version, mission, proof result, authority scope and expiry.
- Material changes to the admitted candidate MUST trigger requalification.
- A superseded successor MUST remain discoverable through the Chronicle.

## 7. Authority model

MCAP-001 defines a default authority ladder:

| Level | Meaning |
|---|---|
| A0 | Observe |
| A1 | Recommend |
| A2 | Act in a sandbox |
| A3 | Execute reversible bounded actions |
| A4 | Execute explicitly approved consequential actions |

Every authority grant MUST declare:

- subject identity;
- mission and capability claim;
- permitted actions and targets;
- prohibited actions;
- value or consequence ceiling;
- start and expiry;
- monitoring requirements;
- revocation and rollback procedure;
- proof and admission references.

Authority MUST remain narrower than demonstrated capability and proportional to value at risk.

## 8. Chronicle events

A Chronicle event SHOULD contain:

```json
{
  "eventId": "urn:uuid:...",
  "eventType": "admission|decision|action|failure|repair|revocation|succession",
  "timestamp": "RFC3339",
  "subject": "ens:successor.asi.eth",
  "predecessor": "optional canonical identifier",
  "missionCommitment": "hash or content identifier",
  "candidateCommitment": "hash or content identifier",
  "evidenceCommitment": "hash or content identifier",
  "proofReference": "canonical identifier",
  "authorityReference": "canonical identifier",
  "resultCommitment": "hash or content identifier",
  "signatures": ["typed or contract-valid signatures"]
}
```

Chronicle storage MAY be off-chain, but accepted event commitments SHOULD be independently timestamped and SHOULD be anchored to tamper-evident infrastructure.

## 9. ENS records

A root SHOULD publish standardized ENS profile records where applicable:

- `url`
- `description`
- `com.github`
- `com.twitter`
- `header`
- `avatar`

Custom records SHOULD use a collision-resistant prefix. The ASI.ETH reference implementation uses `org.montrealai.mcap.*`.

Draft compatibility MAY include:

- `agent-context`
- `agent-endpoint[web]`
- `agent-endpoint[mcp]`
- `agent-endpoint[a2a]`
- `agent-registration[<registry>][<agentId>]`

Draft records MUST be labeled as draft-compatible and MUST NOT be represented as finalized standards.

## 10. Signatures and control

Human-controlled accounts SHOULD use hardware-backed signing and institutional multisignature controls proportional to risk.

Contract-controlled identities SHOULD support ERC-1271 signature validation where appropriate.

Structured institutional statements SHOULD use EIP-712 typed data or an equivalently unambiguous signing format.

No irreversible ENS restriction SHOULD be applied before resolver, controller, expiry, recovery, migration and emergency assumptions have been independently reviewed.

## 11. Conformance levels

### MCAP-Discoverable

- Root Manifest is available.
- Covenant is available.
- Namespace roles and statuses are published.
- Root resolves to a canonical human interface.

### MCAP-Auditable

All MCAP-Discoverable requirements, plus:

- Chronicle commitments are public.
- Proof and admission references are resolvable.
- role controllers and conflicts are disclosed.
- version and succession history are preserved.

### MCAP-Operational

All MCAP-Auditable requirements, plus:

- at least one frozen candidate has completed independent proof;
- an admission record with bounded authority exists;
- monitoring, revocation and rollback have been exercised;
- a successor or requalification event has been recorded.

No implementation may self-declare full conformance solely through namespace ownership. Conformance claims require public evidence and MAY be challenged.

## 12. Positive capabilities

A conforming Machine Civilization SHOULD be designed to:

1. Create measurable value.
2. Coordinate by meaningful consent.
3. Earn trust through proof.
4. Defend shared infrastructure and knowledge.
5. Make cooperation strategically durable under repeated interaction.

These are design objectives, not automatic properties of blockchain, AI or ENS.

## 13. Security and non-claims

The following assumptions are explicitly rejected:

- An ENS name proves intelligence.
- A token balance proves legitimacy.
- Reputation substitutes for fresh proof.
- On-chain data is automatically true.
- A verifier is independent merely because it has a separate address.
- Immutability is always safer than recoverability.
- Succession should transfer all permissions.

A conforming implementation MUST publish material limitations, unresolved risks and dependency assumptions.

## 14. Reference implementation

The Genesis reference implementation is published at:

- `asi.eth`
- `https://montrealai.github.io/asi/`
- `https://montrealai.github.io/asi/manifest.json`

The publication of this specification establishes an open architecture. Mainnet activation of ENS records and subnames remains separately evidenced by owner-signed transactions and resolved-record snapshots.

## 15. Founding principle

> **The root provides identity. The Covenant limits power. The Chronicle preserves consequence. Independent proof qualifies the successor. Authority must requalify.**
