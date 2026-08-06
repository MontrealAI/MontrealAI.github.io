# MONTREAL.AI Sovereign Orchestration — Public Deployment Guide

## Recommended GitHub Pages deployment

1. Verify the SHA-256 sidecar for `MONTREAL_AI_SOVEREIGN_ORCHESTRATION_PRODUCTION_v3_0_0_GITHUB_PAGES_DEPLOY.zip`.
2. Extract the ZIP locally. Its contents are already arranged for the repository root; do not upload the ZIP itself.
3. Create a dedicated branch in `MontrealAI/MontrealAI.github.io`.
4. Review the complete diff. Preserve unrelated historical application routes and repository settings.
5. Copy the extracted files into the repository root, commit and open a pull request.
6. Let repository validation and the GitHub Pages build complete before merge.
7. Verify the principal live routes: `/`, `/fr/`, `/sovereign-orchestration.html`, `/fr/sovereign-orchestration.html`, `/goalos.html`, `/fr/goalos.html`, `/institution.html`, `/fr/institution.html`, `/founder.html`, `/fr/founder.html`, `/canon.html`, `/fr/canon.html`, `/commission.html`, `/fr/commission.html`, `/legal.html` and `/fr/legal.html`.
8. Verify all **66** exact EN/FR counterpart switches on the live origin.
9. Confirm `CANON-40`, sitemap, RSS, `llms.txt`, canonical URLs, `hreflang` and `x-default` on the live origin.
10. Configure and test the `montreal.ai` redirect separately.
11. Apply reviewed response security headers through an edge layer if available.
12. Complete live Safari, Firefox, iOS, Android, keyboard, screen-reader, zoom, forced-colours, performance and qualified Québec legal-language review.

## Rollback

Deploy through a versioned branch and tagged commit. If a material live regression appears, revert to the immediately preceding known-good commit rather than editing production ad hoc. Preserve logs, screenshots and the reason for rollback in the release record.
