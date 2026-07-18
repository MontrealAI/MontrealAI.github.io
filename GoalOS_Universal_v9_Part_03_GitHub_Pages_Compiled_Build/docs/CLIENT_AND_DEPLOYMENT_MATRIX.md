# Client and Deployment Matrix

| Audience | Entry | Authentication | Data posture | Recommended deployment |
|---|---|---|---|---|
| Public visitor | `buyer-funnel.html` | None | Public/non-confidential | GitHub Pages or IPFS |
| AGI Club member | `agi-club.html` | MetaMask + current direct `*.club.agi.eth` ownership | Browser-local | GitHub Pages or IPFS |
| Enterprise single user | `enterprise/index.html` / one-file app | Local acceptance receipt | Local device | Downloaded HTML |
| Enterprise team | Same compiled static build | Organization-controlled SSO/gateway | Private/confidential | Internal static hosting |
| Restricted enterprise | One-file app | Device/room controls | Restricted | Air-gapped or approved controlled environment |
| Operator/reviewer | `operator/index.html` | Operator-controlled environment | QA/evidence | Private operations host |

Public static hosting does not provide content secrecy. Wallet gating verifies entitlement and activates the interface; it does not make delivered bytes confidential.
