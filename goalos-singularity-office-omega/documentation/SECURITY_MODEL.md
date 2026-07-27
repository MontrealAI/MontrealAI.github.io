# Security Model

The public-static edition has no GoalOS telemetry by design, stores state browser-locally, uses temporary access receipts, revalidates ENS ownership, supports signed licences, encrypts Evidence Vault exports optionally with AES-256-GCM, and separates Execute / Accept / Admit. It has no external transaction authority.

The optional protected runtime requires TLS, exact origin allowlists, short-lived tokens, rate limits, payload caps, no prompt logging, managed secrets, audit evidence, incident response and customer-specific contracts.

The static access gate is not confidential DRM. Encryption protects stored bytes but does not prove the underlying claim.
