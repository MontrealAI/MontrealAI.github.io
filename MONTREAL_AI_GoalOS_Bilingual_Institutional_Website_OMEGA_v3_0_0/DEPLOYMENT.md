# Déploiement et retour arrière / Deployment and rollback

## Cible / Target

- Repository / Dépôt: `MontrealAI/MontrealAI.github.io`
- Branch / Branche: `master`
- Location / Emplacement: repository root / racine du dépôt
- Release: `MONTREAL_AI_GoalOS_Bilingual_Institutional_Website_OMEGA_v3_0_0`

## Déploiement recommandé / Recommended deployment

```bash
git checkout master
git pull --ff-only
git checkout -b goalos-bilingual-institutional-v3

unzip MONTREAL_AI_GoalOS_GITHUB_PAGES_DEPLOY_OVERLAY_v3_0_0.zip \
  -d /path/to/MontrealAI.github.io

cd /path/to/MontrealAI.github.io
python scripts/verify_site.py
python scripts/content_accessibility_qa.py
python scripts/navigation_qa.py
python scripts/browser_qa.py
node --check goalos-assets/js/site.js

git status
git diff --stat
```

1. Preserve the current root page and commit as the immediate rollback point.
2. Inspect `SITE_OVERLAY_MANIFEST.json`; the overlay must not delete unlisted legacy paths.
3. Confirm contact details, operating entity, marks, legal text, downloads and source dates.
4. Preview through HTTPS on a review branch or environment.
5. Complete `LEGAL_ACTIVATION_CHECKLIST.md` and `LIVE_DEPLOYMENT_ACCEPTANCE.md`.
6. Merge only after an authorized activation decision.

## Local preview / Aperçu local

- Simplest: open `START_HERE.html`.
- HTTP preview: run `START_LOCAL_PREVIEW.command` on macOS or `START_LOCAL_PREVIEW.bat` on Windows.
- The preview server binds to `127.0.0.1` only and opens the French homepage locally.

## Rollback / Retour arrière

Revert the launch commit or restore the previous `index.html`. Do not delete legacy routes during rollback unless separately approved.

## GitHub Pages boundary / Limite GitHub Pages

GitHub Pages does not permit every custom response header. The release uses a restrictive meta CSP and no external runtime dependencies. Higher-assurance hosting should additionally enforce server-level HSTS, CSP, Permissions-Policy, Referrer-Policy, X-Content-Type-Options, framing protection and cache controls.

## Activation boundary / Frontière d’activation

This package publishes research, product architecture, public proof status and non-confidential qualification. Accounts, customer data, confidential workflows, wallet connections, payments, smart-contract execution, high-impact automated decisions and regulated professional services require separate protected infrastructure, contracts and review.
