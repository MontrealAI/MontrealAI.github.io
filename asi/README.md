# ASI.ETH — Machine Civilization Root

**Status:** Genesis publication / owner-signature gate  
**Version:** 0.1.0  
**Publisher:** MONTREAL.AI  
**Canonical root:** `asi.eth`

ASI.ETH is defined here as the addressable root of a proof-gated Machine Civilization architecture: a persistent, human-readable and machine-resolvable namespace for constitutions, successors, registries, Chronicles, proofs, verifiers, admission state, policy, rollback and bounded economic resources.

This repository publishes the open architecture and canonical machine-readable documents. It does **not** claim that every proposed ENS record, subname, contract or verifier is already active. Mainnet activation remains gated by explicit owner signatures, independent review and published transaction references.

## Governing doctrine

- Memory may cross. Authority must requalify.
- Capability may compound. Trust must be earned anew.
- The claimant must not control the evidence and certify its own legitimacy.
- Consequential authority must be scoped, expiring, attributable, observable and reversible.
- The institution must survive replacement of any individual model, agent, operator or infrastructure provider.

## Canonical public documents

- Human interface: `https://montrealai.github.io/asi/`
- Root manifest: `https://montrealai.github.io/asi/manifest.json`
- Machine Civilization Addressability Protocol: `https://montrealai.github.io/asi/MCAP-001.md`
- Successor Covenant: `https://montrealai.github.io/asi/COVENANT.md`
- Namespace registry: `https://montrealai.github.io/asi/registry.json`
- ENS activation record set: `https://montrealai.github.io/asi/ens-records.json`
- Well-known discovery document: `https://montrealai.github.io/.well-known/asi-machine-civilization.json`

## Root namespace

| ENS name | Canonical role | Control principle |
|---|---|---|
| `asi.eth` | Root discovery and namespace | Root stewardship; no operational authority implied |
| `successor.asi.eth` | Current admitted successor institution | Updated only through admitted succession |
| `covenant.asi.eth` | Founding covenant | Stable constitutional reference |
| `constitution.asi.eth` | Machine-readable rules | Versioned and hash-bound |
| `registry.asi.eth` | Identity and lineage graph | Publicly auditable state |
| `chronicle.asi.eth` | Append-only institutional memory | Independent from current successor |
| `proof.asi.eth` | Proof requests, evidence commitments and results | Claimant cannot be final verifier |
| `verifier.asi.eth` | Independent verification set | Separate control from claimant/operator |
| `admission.asi.eth` | Qualification and authority state | Proof-gated, expiring, revocable |
| `policy.asi.eth` | External policy and authority constraints | Narrower than demonstrated capability |
| `rollback.asi.eth` | Reversal and emergency recovery | Independently triggerable under published rules |
| `treasury.asi.eth` | Bounded settlement and resources | Limits, timelocks and separation of duties |

## Activation sequence

1. Publish and review this architecture.
2. Freeze the Genesis manifest and Covenant hashes.
3. Configure ENS standard profile records on `asi.eth`.
4. Configure prefixed custom records that resolve the manifest, registry, Covenant and status.
5. Publish draft-compatible `agent-context` and `agent-endpoint[web]` records.
6. Create only the subnames whose functions are real and inspectable.
7. Assign independent controllers for proof, verifier, Chronicle and admission roles.
8. Publish all mainnet transaction hashes and resolved-record snapshots.
9. Register compatible agent identities only after the relevant standards and deployment assumptions are independently reviewed.
10. Begin external conformance and proof exercises; record results in the Chronicle.

## Standards posture

The architecture uses stable ENS resolver capabilities today: addresses, text records, content hashes, ABIs and public keys. It also prepares compatibility with draft agent standards without treating drafts as final:

- ENSIP-25 — AI Agent Registry ENS Name Verification (draft)
- ENSIP-26 — Agent Text Records (draft)
- ENSIP-27 — Node Classification and Metadata (draft)
- ERC-8004 — Trustless Agents (draft)
- EIP-712 — Typed structured data hashing and signing
- ERC-1271 — Contract signature validation

Draft compatibility is optional and explicitly labeled. The ASI.ETH root remains useful through ordinary ENS resolution even if draft specifications change.

## Non-claims

Control of `asi.eth` does not prove artificial superintelligence, universal legitimacy, alignment, or authority over any person, organization, network or agent. The name provides addressability. Claims require evidence; authority requires fresh independent proof.

## License

The architecture, schemas and Covenant text are published for open inspection, implementation, criticism and peaceful fork. Attribution to MONTREAL.AI is requested for provenance. No trademark or endorsement is implied by compatible implementation.