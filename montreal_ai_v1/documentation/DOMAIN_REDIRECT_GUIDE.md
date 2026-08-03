# `montreal.ai` Redirect Guide

The intended public architecture is:

`https://montreal.ai/` → permanent HTTPS redirect → `https://montrealai.github.io/`

Use the DNS and redirect service currently controlling `montreal.ai`. Configure a permanent redirect preserving paths and query strings where supported. Verify:

1. `http://montreal.ai/` upgrades to HTTPS.
2. `https://montreal.ai/` returns a permanent redirect to the canonical GitHub Pages root.
3. No redirect loop occurs.
4. Security headers and TLS are valid.
5. Search engines see only `https://montrealai.github.io/` as canonical, consistent with the website metadata.

The release package does not claim that this external redirect has already been configured.
