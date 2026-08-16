# Security Policy

The public repository contains encrypted payloads, public IVs and hashes. The decryption key is stored only as a Cloudflare Worker secret. The Worker issues one-time challenges, verifies the wallet signature, rechecks current on-chain eligibility and creates a short-lived session.

The signed request states `Authority created: NONE`. Eligibility does not establish identity assurance, professional competence, corporate mandate, Chronicle admission or Successor Ω authority.

Report suspected vulnerabilities privately to `president@montreal.ai`.
