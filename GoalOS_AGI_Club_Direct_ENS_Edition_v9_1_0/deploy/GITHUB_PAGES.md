# GitHub Pages deployment

## Fastest path

1. Create or choose a repository/folder.
2. Copy the deployable v8.1.0 files into the publishing root.
3. Keep `.nojekyll` in the root.
4. In repository **Settings → Pages**, select **GitHub Actions** as the source.
5. Commit the included `.github/workflows/deploy-pages.yml`.
6. Run the workflow and open the resulting HTTPS origin.
7. Complete `LIVE_MAINNET_ACCEPTANCE_CHECKLIST.md`.

## Project-page paths

All application URLs are relative. The same build works under a project path such as:

```text
https://montrealai.github.io/GoalOS_AGI_Club_Direct_ENS_Web3_Production_Final_v8/
```

Do not open the member pages by `file://` for production verification. MetaMask membership activation should be tested from the deployed HTTPS origin.

## Branch-only alternative

GitHub Pages can publish static files from a branch. Keep `.nojekyll` to bypass Jekyll processing. The included Actions workflow is preferred because it provides a repeatable deployment gate.

## Security limitations

GitHub Pages is static hosting and does not allow the operator to set every desired response header. The package includes a restrictive CSP in the principal pages, no analytics, no server form, no secrets, and local assets. Use an appropriate secure gateway/CDN if stricter headers, WAF, access logs, or incident controls are required.
