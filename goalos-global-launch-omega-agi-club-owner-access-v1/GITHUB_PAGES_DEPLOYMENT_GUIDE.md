# GitHub Pages Deployment Guide — GoalOS Global Launch Ω v13.0.0-GL1

## Publish the Deployment ZIP

1. Download and unzip `GoalOS_Global_Launch_Omega_AGI_CLUB_GITHUB_PAGES_INSTITUTION_v13_0_0_GL1_DEPLOY.zip` locally.
2. Copy the root `.nojekyll` file and the complete `goalos-global-launch-omega-agi-club-owner-access/` directory into the root of `MontrealAI/MontrealAI.github.io` on the Pages publishing branch.
3. Do **not** upload the ZIP itself as the website and do **not** introduce another enclosing release directory.
4. Preserve all unrelated repository content, commit and publish.

Canonical URL:

`https://montrealai.github.io/goalos-global-launch-omega-agi-club-owner-access/`

## What to deploy

```text
MontrealAI.github.io/
├── .nojekyll
└── goalos-global-launch-omega-agi-club-owner-access/
    ├── index.html
    ├── 404.html
    ├── START_HERE.html
    ├── assets/
    ├── governance/
    ├── research/
    ├── data/
    ├── schemas/
    ├── tools/
    └── ...
```

## Static protection boundary

GitHub Pages publicly serves client-side source. The AGI Club gate provides conditional-use access, signed provenance and repeated on-chain verification for ordinary use; it is not confidential server-side DRM. Do not place private keys, application secrets, confidential customer data, private package payloads, custody, filing or external execution authority in this repository.
