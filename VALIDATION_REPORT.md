# GoalOS Contract-Exact Autonomous Demo — Validation Report

**Release:** `GOS-CONTRACT-DEMO-2026.07-v1.0`  
**Validation date:** 2026-07-13  
**Disposition:** **PASS — 68 / 68 automated checks**

## Executive verdict

The standalone browser application completes the entire deterministic proof-mission fixture without a wallet, RPC endpoint, server or external asset. The fresh autonomous run executes **46 of 46 ABI-encoded calls** across **13 phases**, settles Job `0`, issues completion NFT `0`, grants the four bounded authority transitions, commits the Reality/Chronicle epoch, submits the graph checkpoint and records Mission 2 as **canary-only**.

The validation distinguishes two rails:

1. **Deployed-core mirror:** ERC-20 `$AGIALPHA`, `AGIJobManager` and the best-effort `ENSJobPages` hook surface.
2. **GoalOS reference rail:** participation receipt, AGI Node, bonded-authority, Reality evidence, checkpoint and constitutional-upgrade fixtures.

The second rail is explicitly simulated beside the first. The report does not represent those reference modules as internal calls made by the deployed `AGIJobManager`.

## Scope tested

| Area | Result |
|---|---:|
| Operating views | 10 / 10 populated |
| Autonomous phases | 13 / 13 |
| Top-level transactions | 46 / 46 |
| Registered function entries | 31 |
| Contract surfaces | 9 |
| Fresh invariant suite | 12 / 12 |
| Automated validation checks | 68 / 68 |
| Browser console/page errors | 0 |
| Required external runtime assets | 0 |
| Desktop horizontal overflow | 0 material failures |
| Mobile horizontal overflow | 0 |

## ABI, selector and calldata conformance

The transaction plan was independently checked outside the application runtime:

- every function selector was recomputed using legacy Ethereum Keccak-256;
- all 46 calldata payloads were independently decoded from raw bytes and compared with the declared arguments;
- all observed event `topic0` values were independently recomputed;
- the canonical ERC-20 regression selector `transfer(address,uint256) = 0xa9059cbb` passed;
- the 31-function ABI subset is exported in `artifacts/ABI_SUBSET.json`;
- the full transaction and internal-call graph is exported in `artifacts/CALL_GRAPH.json`.

## ENS identity conformance

The offline ENS implementation uses exact namehash/subnode semantics. The computed root for `alpha.agent.agi.eth` is:

```text
0xc74b6c5e8a0d97ed1fe28755da7d06a84593b4de92f6582327bc40f41d6c2d5e
```

It matches the constant used by the official AGIJobManager console. The fixture also evaluates direct `*.node.agi.eth` ownership and the `PARENT_CANNOT_CONTROL` emancipation fuse before node activation.

## Source-defined economic reconciliation

| Quantity | Exact fixture result |
|---|---:|
| Employer escrow | 1,000 `$AGIALPHA` |
| Agent payout snapshot | 82% |
| Agent payout | 820 |
| Agent duration-weighted bond | 53.024 |
| Validator reward budget | 80 |
| Validator bond per voter | 150 |
| Incorrect-side slash | 120 |
| Reward per correct validator | 66.666666666666666666 |
| Validator-pool wei remainder to agent | 2 wei |
| Manager retained remainder | 100 |
| Agent reputation points | 38 |

The fresh browser self-test confirms exact ERC-20 conservation, a zero final balance in the reference bond kernel and a final manager balance of exactly `100 $AGIALPHA` for this deployed-core fixture.

## Functional scenario result

The run demonstrates:

- six versioned participation receipts;
- four direct AGI Node registrations;
- exact token approval and escrow pull;
- job creation and ENS-authorized agent assignment;
- source-defined agent bond calculation;
- ProofBundle completion submission and proof-root binding;
- four source-defined validator bonds;
- three approvals and one preserved dissent;
- challenge-window advancement;
- permissionless finalization;
- reward distribution, incorrect-side slashing and NFT issuance;
- best-effort ENS terminal hook;
- paid, trusted, challenge-safe and reusable authority grants;
- Reality/Chronicle commitment and institution attestation;
- sequential graph checkpoint;
- constitutional Mission 2 record with `CanaryOnly` authority.

## Revert and branch coverage

The isolated Revert Lab verifies the expected logic for:

1. early finalization;
2. duplicate voting;
3. unowned ENS authorization;
4. fee-on-transfer token rejection;
5. under-quorum dispute routing;
6. inactive-node validator bond rejection;
7. missing reusable-authority bond;
8. broken checkpoint continuity.

Each probe runs against an isolated snapshot and leaves the main demonstration state unchanged.

## Browser and usability validation

The application was tested in headless Chromium at desktop and mobile viewport sizes. The full autonomous run completed at both sizes. Every operating view rendered meaningful content, local links resolved, inline JavaScript passed `node --check`, and no browser warnings, errors or page exceptions were recorded.

Visual review assets are included in `qa/`:

- `overview.png`
- `run_complete.png`
- `contracts.png`
- `settlement.png`
- `mobile_command.png`
- `mobile_run_complete.png`

## Source and artifact integrity

- Bundled Solidity and semantic source files are fingerprinted in `artifacts/SOURCE_FINGERPRINTS.json`.
- Every release file is covered by `SHA256SUMS.txt`.
- `ARTIFACT_INDEX.json` identifies purpose, size and checksum for each included artifact.
- `CONTRACT_CALL_MANIFEST.json` preserves exact transaction signatures, selectors, calldata, events, relevant internal calls and deterministic receipt identifiers.

## Claim boundary

This validation establishes the correctness and internal consistency of the **offline deterministic simulation**. It does not establish:

- live RPC execution or wallet signing;
- bytecode equivalence to a particular deployed block;
- audited contract security;
- mainnet transaction success;
- suitability of any reference fixture for live value;
- reviewed production deployment;
- independent external replay;
- broad empirical recursive-self-improvement or Mission 2 compounding.

Those remain separate evidence gates.

## Final disposition

> **PASS — ready for autonomous MASTERCLASS demonstration as a contract-exact, source-bound, claim-bounded offline simulation.**
