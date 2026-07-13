# Smart-Contract Conformance Matrix

**Release:** `GOS-CONTRACT-DEMO-2026.07-v1.0`

## Interpretation

“Contract-exact” in this package means that each demonstrated call uses the declared Solidity signature, independently verified selector, ABI-encoded arguments and source-bound state transition for the fixture. It does **not** mean a live fork, bytecode proof or security audit.

| Contract surface | Address represented | Status | Top-level functions used | Relevant simulated internal functions |
|---|---|---|---|---|
| `ERC20 $AGIALPHA` | `0xa61a3b3a130a9c20768eebf97e21515a6046a1fa` | mainnet address from official console | approve(address,uint256) | — |
| `AGIJobManager` | `0xb3aaeb69b630f0299791679c063d68d6687481d1` | verified mainnet address; current official repository/source semantics | applyForJob(uint256,string,bytes32[]); createJob(string,uint256,uint256,string); disapproveJob(uint256,string,bytes32[]); finalizeJob(uint256); lockJobENS(uint256,bool); requestJobCompletion(uint256,string); validateJob(uint256,string,bytes32[]) | — |
| `ENSJobPages` | `0x06188e77c1c38d392b16d9d9fb24673363ce1da0` | official console fallback address; best-effort hook target | — | handleHook(uint8,uint256) |
| `UserResponsibilityReceiptRegistryV4` | `0x9000000000000000000000000000000000000001` | deterministic reference fixture; not a claimed live deployment | commit(bytes32,bytes32,bytes32,bytes32,uint64) | — |
| `AGINodeRegistry` | `0x9000000000000000000000000000000000000002` | deterministic reference fixture; source says unaudited | registerDirectNode(bytes32,bytes32,bytes32,bytes32) | isActiveOperator(address) |
| `BondedAuthorityKernel` | `0x9000000000000000000000000000000000000003` | deterministic reference fixture; source says unaudited/no live value | bindProofRoot(bytes32,bytes32); grantTransition(bytes32,uint8,bytes32,bytes32,bytes32[]); lockBond(bytes32,bytes32,uint8,uint256,bytes32,bytes32,uint64); resolveBond(bytes32,uint16,address) | — |
| `RealityEvidenceRegistry` | `0x9000000000000000000000000000000000000004` | deterministic reference fixture; commitment-only | attest(bytes32,bytes32); commitEpoch(bytes32,uint64,bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,uint8) | — |
| `PlanetaryCheckpointRegistry` | `0x9000000000000000000000000000000000000005` | deterministic reference fixture; compact commitments | submitCheckpoint(uint64,bytes32,bytes32,bytes32,bytes32,bytes32) | — |
| `ConstitutionalUpgradeRegistry` | `0x9000000000000000000000000000000000000006` | deterministic reference fixture; canary-only RSI epilogue | recordUpgrade(uint64,bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,uint8) | — |

## Deployed-core lifecycle

The complete core use case executes:

```text
approve → createJob → approve → applyForJob → requestJobCompletion
→ approve/validateJob × 3 → approve/disapproveJob × 1
→ finalizeJob → lockJobENS
```

The simulation additionally exposes source-defined reads such as `getJobCore`, `getJobValidation` and `tokenURI` in the function inspector.

## Reference proof lifecycle

The complete GoalOS overlay executes:

```text
commit participation receipts
→ registerDirectNode × 4
→ lockBond × 4
→ bindProofRoot
→ grantTransition × 4
→ resolveBond × 4
→ commitEpoch → attest
→ submitCheckpoint
→ recordUpgrade(CanaryOnly)
```

These calls are deliberately shown as a separate orchestrated rail. No claim is made that the deployed AGIJobManager calls those reference modules internally.

## Machine-readable conformance artifacts

- `artifacts/ABI_SUBSET.json`
- `artifacts/CONTRACT_CALL_MANIFEST.json`
- `artifacts/FUNCTION_REGISTRY.json`
- `artifacts/CALL_GRAPH.json`
- `artifacts/SIMULATION_FIXTURE.json`
- `artifacts/SOURCE_FINGERPRINTS.json`
