# GitHub Pages Deployment Guide — MONTREAL.AI Sovereign Crown Ω v8.1.0

## Correct archive

Use `MONTREAL_AI_SOVEREIGN_CROWN_OMEGA_V8_1_0_PUBLISH_READY.zip`.

## Controlled publication

1. Preserve the current `MontrealAI/MontrealAI.github.io` commit as the rollback point.
2. Extract the publish-ready ZIP locally.
3. Create a reviewed publication branch.
4. Upload the **extracted contents** directly to the repository root.
5. Preserve unrelated historical application, paper, demonstration and archive directories that are not present in the replacement payload.
6. Review all additions, modifications and deletions before merge.
7. Confirm the included Site Integrity workflow passes.
8. Merge and wait for GitHub Pages to publish.
9. Verify the English root, `/fr/`, `goalos.html`, `proof.html`, `founder.html`, `sprint.html`, legal pages, `sitemap.xml`, `feed.xml`, the manifest and one preserved historical application.
10. Configure and verify the permanent HTTPS redirect from `montreal.ai` to `https://montrealai.github.io/`.
11. Record the deployed commit and UTC publication time in the release certificate.

Do not upload the ZIP itself. Do not place the extracted files inside an additional enclosing directory. Do not deploy the complete audit package as the web root.

## Rollback

If live-origin verification fails, revert the publication commit or restore the preserved root files. Do not repair a failed public release directly on the production branch without a reviewed diff.

## Public/private boundary

Never upload private Publisher Vaults, private signing material, seed phrases, production credentials, confidential Evidence Dockets, customer secrets, protected runtimes or unauthorized third-party data.
