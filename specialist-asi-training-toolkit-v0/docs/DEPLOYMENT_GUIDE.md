# GoalOS v8.8.0-UVSI3 — Deployment guide

## GitHub Pages

1. Unzip the GitHub Pages package.
2. Upload all extracted files to the repository root, including `.github` and `.nojekyll`.
3. In **Settings → Pages**, select **GitHub Actions**.
4. Wait for the included workflow to complete.
5. Open the published site in a wallet-enabled browser.

GitHub Pages is the public control surface. It cannot run confidential or GPU-scale training.

## Local Toolkit

Run the platform-specific launcher. The server binds to `127.0.0.1:8788` by default. No mandatory third-party Python dependency is required for the reference core.

## Protected backend

Use the Dockerfile or install the package into an isolated environment. Set allowed origins and, when used behind the access gateway, `GOALOS_GATEWAY_SECRET`.

## Optional access gateway

Deploy the Cloudflare Worker, store the gateway secret as a Worker secret, set the exact allowed origin and protected Foundry origin, then configure the PWA endpoint. The gateway revalidates current on-chain eligibility before forwarding a protected cycle request.
