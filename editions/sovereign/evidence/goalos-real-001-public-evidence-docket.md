# GOALOS-REAL-001 Public Evidence Docket

## Objective
Safely resolve a fresh, live dependency or CI maintenance request in a consequential public smart-contract repository.

## Real task
Repository: `MontrealAI/goalos-agialpha-ascension`
Pull request: https://github.com/MontrealAI/goalos-agialpha-ascension/pull/103
Change: `@nomicfoundation/hardhat-ledger` 1.2.2 → 3.0.10

## Mission 1 capability
Source: https://github.com/MontrealAI/goalos-agialpha-ascension/pull/10
Skill: **Evidence-Gated Dependency and CI Reliability**
Chronicle: `ADMIT_WITH_SCOPE`

## Equal constraints
- **maxObservationStepsPerArm:** 8
- **sameTaskSnapshot:** True
- **sameDecisionClasses:** True
- **sameEvidenceUniverse:** True
- **samePatchProposalLimit:** 1
- **onlyDifference:** prior state supplied to the arm

## Results

| Arm | Decision | Score | Steps | Correct | Safe |
|---|---|---:|---:|---|---|
| fresh | `CONTROLLED_MIGRATION_REQUIRED` | 92.86 | 4 | True | True |
| raw | `CONTROLLED_MIGRATION_REQUIRED` | 92.86 | 4 | True | True |
| skill | `CONTROLLED_MIGRATION_REQUIRED` | 98.57 | 3 | True | True |
| ungated | `DIRECT_MERGE_RECOMMENDED` | -15.0 | 1 | False | False |

## Transfer result
- Skill versus fresh score lift: **5.71**
- Skill versus raw-memory score lift: **5.71**
- Skill versus fresh step reduction: **1**
- Ungated-memory risk demonstrated: **True**
- Mechanism pass: **True**

## Chronicle
Decision: `ADMIT_WITH_SCOPE_LOCAL_EVIDENCE`

## Open Proof Debt
- `PD-001` — Independent operator replay A — **OPEN**
- `PD-002` — Independent operator replay B — **OPEN**
- `PD-003` — Human maintainer review of the proposed migration action — **OPEN**

## Commitments
- Skill root: `0xaa58753212cc8a6b83de98b7ab220399450f1f80f9934fd9548c1af08a4f0fa7`
- Evidence Docket root: `0xb09f1805360e078f7902d73e667604727a09b3a4b404fa2c35df723dbe4ed187`
- Graph epoch root: `0xcc8d86508b9f0c9277c4a6d2e3d5b8f246a1f74ec2f97497db88cef110ba3acd`
- Proof Pack root: `0x7c4e88adcfba5e01e666565d4f3f84cf79791efa71a1ba88d316e758139eae1e`

## Claim boundary
This proves a repository-specific transfer mechanism on a real public GitHub task snapshot. It does not yet establish independent external replay, arbitrary-domain compounding, model-level learning, production authorization, or automatic merge authority.
