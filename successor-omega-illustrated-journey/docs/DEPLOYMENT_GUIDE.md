# Deployment Guide

The secure edition uses two components:

1. **GitHub Pages** hosts the public gate and encrypted publication payloads.
2. **Cloudflare Worker** verifies current wallet eligibility and supplies the short-lived decryption key.

A client-only gate cannot make plaintext files in a public repository exclusive. Do not place readable PDFs or ZIPs in the GitHub repository.

## Deploy the Worker

```bash
npm install
npx wrangler login
npx wrangler kv namespace create ACCESS_KV
npx wrangler kv namespace create ACCESS_KV --preview
```

Paste the IDs into `wrangler.toml`, then:

```bash
npx wrangler secret put ETHEREUM_RPC_URL
npx wrangler secret put CONTENT_KEY_B64
npm run check
npm run deploy
```

Use the exact content key in the private package.

## Configure GitHub Pages

Replace the placeholder `access.brokerUrl` in `config.js`. If the repository slug changes, update the Worker `APP_PATH` too.

Create `MONTREALAI/successor-omega-illustrated-journey`, upload the public package to the repository root, then choose **GitHub Actions** under **Settings → Pages**.

Suggested URL: `https://montrealai.github.io/successor-omega-illustrated-journey/`

## Browser-safe multipart asset

The encrypted Complete Publication Suite is divided into six parts below 20 MiB. The website automatically downloads, verifies, decrypts and reassembles them. Upload all six `complete_suite.part-*.enc` files and `protected/manifest.json`; do not rename or combine them.

For detailed browser-upload steps, see `BROWSER_UPLOAD_GUIDE.md`.

## Acceptance tests

Test both eligible routes, ineligible wallets, wrong network, account/network changes, session expiry, inactivity lock, every protected PDF and the multipart Complete Publication Suite download.

## Never commit

The content key, RPC credentials, Cloudflare credentials, `.dev.vars`, or decrypted files.
