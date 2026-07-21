# Security policy / Politique de sécurité

## Private reporting / Signalement privé

Report a potential vulnerability to **secretariat@montreal.ai** with subject `Security / Sécurité`.

Include the affected version or commit, URL or file, minimum reproducible steps, likely impact and a proposed mitigation where possible. Do not send production secrets, wallet keys, seed phrases, unnecessary personal information or third-party confidential material.

## Public release boundary

This release is a static, data-minimized website. It contains no public authentication, server form, analytics, wallet, payment, transaction, private execution secret or customer-data workflow. Downloadable products have separate threat models and activation requirements.

## Release controls

- restrictive meta CSP with exact script hashes;
- no external runtime scripts, fonts, frames or APIs;
- explicit local file paths and same-origin assets;
- secret/placeholder scan;
- browser, navigation, accessibility and static QA;
- deterministic ZIPs and SHA-256 manifests;
- rollback and live-acceptance procedures.

## Coordinated disclosure boundary

This policy is not a bug-bounty promise, safe harbour, employment offer or obligation to investigate, pay or remediate. Researchers must act lawfully, avoid disruption, avoid accessing data belonging to others and stop after the minimum proof necessary.

Activated enterprise services, private execution, ENS, wallet, smart-contract or protected-capability systems require separate security architecture, independent review, key management, incident response and production acceptance.
