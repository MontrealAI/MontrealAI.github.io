# Deployment and rollback — MONTREAL.AI v11.0.0

## Safe publication path

1. Preserve the current repository commit and create a protected publication branch.
2. Extract `MONTREAL_AI_GITHUB_PAGES_PRODUCTION_v11_0_0.zip` so that `index.html`, `fr/`, `assets/`, `data/`, `.nojekyll`, `robots.txt` and `sitemap.xml` sit at repository root.
3. Review the full diff. Preserve unrelated historical applications unless they are intentionally superseded.
4. Confirm the ZIP SHA-256 against its sidecar and inspect the clean-room verification receipt.
5. Publish through the owner-controlled GitHub Pages source.
6. Execute the live-origin checks below. Revert the publication commit if any required check fails.

## Required live-origin checks

- `/` and `/fr/` load their correct languages.
- Every visible EN/FR control reaches the exact counterpart and returns correctly.
- Security and legal surfaces retain their dark, readable shared header.
- Federation pages have substantive structural parity.
- GoalOS stage selection, research filtering, command palette, mobile navigation and verifier operate.
- `sitemap.xml`, `robots.txt`, `.well-known/security.txt`, `llms.txt` and `release.json` resolve.
- No unrelated historical application disappears in the repository diff.

## Rollback

Keep the pre-publication commit SHA. On failure, revert to that SHA, repair on a branch, rerun the complete QA gate and publish a new versioned receipt.
