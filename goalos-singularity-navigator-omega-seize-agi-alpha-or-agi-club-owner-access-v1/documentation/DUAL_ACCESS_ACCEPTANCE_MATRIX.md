# GoalOS Dual-Access Acceptance Matrix — v3.3.0-SN5-BI2

| ID | Case | Expected | Status |
|---|---|---|---|
| A01 | Existing current AGI Club direct owner | Existing route remains available and unchanged | TESTED BY PRESERVATION / LIVE SIGNATURE PENDING |
| A02 | Direct official AGIALPHA balance = 1,000,000 | Qualifies | PASS (deterministic fixture) |
| A03 | Direct official AGIALPHA balance > 1,000,000 | Qualifies | PASS (deterministic fixture) |
| A04 | Direct official AGIALPHA balance = 999,999.999999999999999999 | Denied | PASS (deterministic fixture) |
| A05 | Balance at wrong token contract or bridged wrapper | Denied; only exact contract is queried | PASS (static/runtime inspection) |
| A06 | Wrong chain | Requests Ethereum Mainnet or denies | PASS (deterministic fixture) |
| A07 | Signature rejected | Denied; no access receipt | PASS (deterministic fixture) |
| A08 | Balance falls below threshold | Fail-closed relock | PASS (deterministic fixture) |
| A09 | Account changes | Fail-closed relock | PASS (deterministic fixture) |
| A10 | Network changes | Fail-closed relock | PASS (deterministic fixture) |
| A11 | Wallet disconnects | Fail-closed relock | PASS (runtime inspection) |
| A12 | 30-minute expiry | Fail-closed relock | PASS (runtime inspection) |
| A13 | 10-minute inactivity | Fail-closed relock | PASS (runtime inspection) |
| A14 | Five-minute, focus, visibility and pre-export checks | Revalidation occurs | PASS (runtime inspection) |
| A15 | Token approval/transfer/payment methods | Never requested | PASS (runtime inspection) |
| A16 | English and French entry pages | Complete, equivalent access disclosure | PASS |
| A17 | All original deployment files | No original path removed | PASS |
| A18 | Actual qualifying AGI Club wallet after publication | Signed production record | PENDING POST-PUBLICATION |
| A19 | Actual qualifying AGIALPHA wallet after publication | Signed production record | PENDING POST-PUBLICATION |

The two pending cases cannot truthfully be completed before deployment with actual current qualifying wallets. They are operational acceptance records, not build defects.
