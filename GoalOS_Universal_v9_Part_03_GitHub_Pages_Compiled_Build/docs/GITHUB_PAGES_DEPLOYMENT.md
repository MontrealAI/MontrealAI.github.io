# GitHub Pages Deployment

1. Create a repository and place the **source package** at its root.
2. Review `public/config.json`, legal documents, support contact, and custom-domain settings.
3. Run locally: `npm ci && npm run qa`.
4. In repository **Settings → Pages**, choose **GitHub Actions** as the source.
5. Push to `main` or run the included `Deploy GoalOS to GitHub Pages` workflow manually.
6. Add environment protection to the `github-pages` environment.
7. Configure a custom domain in repository settings; a `CNAME` file alone does not configure the domain.
8. Run the controlled live-wallet acceptance suite on the final HTTPS origin.

The workflow uses `actions/configure-pages@v5`, `actions/upload-pages-artifact@v4`, and `actions/deploy-pages@v4`, with `pages: write` and `id-token: write` permissions.

**Confidentiality:** GitHub Pages is a public/static distribution surface unless the organization’s plan and repository/access arrangement explicitly provide otherwise. Do not treat it as a confidential enterprise data room.
