# Deployment guide

## A. Deploy the strict GitHub Pages edition

### No-build GitHub Actions route

1. Create a new GitHub repository.
2. Unzip `GoalOS_UVSI2_v7_GitHub_Pages.zip`.
3. Upload **all** extracted contents to the repository root, including `.github` and `.nojekyll`.
4. Commit to the `main` branch.
5. Open **Settings → Pages**.
6. Set **Source** to **GitHub Actions**.
7. Open the **Actions** tab and wait for `Deploy GoalOS UVSI2 to GitHub Pages` to finish.
8. Open the deployment URL shown by GitHub.

No npm install, build command or database is required.

### Branch route

You can alternatively choose **Deploy from a branch**, select `main` and `/ (root)`. The site is already static.

## B. Confirm the access gate

1. Open the deployed page in a browser with an injected EIP-1193 Ethereum wallet.
2. Verify that the wallet is on Ethereum Mainnet.
3. Test one route:
   - enter the one label before `.club.agi.eth`; or
   - verify the direct AGIALPHA balance.
4. Confirm that the wallet asks only for connection and a personal signature—never token approval or a transaction.
5. Confirm that changing the account or network relocks the application.

## C. Configure secure AI

The GitHub Pages frontend must never contain an OpenAI API key.

### Cloudflare Worker

1. Unzip `GoalOS_UVSI2_v7_Secure_AI_Backend.zip`.
2. Install Node.js.
3. Run `npm install`.
4. Run `npx wrangler login`.
5. Store the key as a secret: `npx wrangler secret put OPENAI_API_KEY`.
6. In `wrangler.toml`, set `ALLOWED_ORIGINS` to the exact GitHub Pages origin.
7. Run `npm run deploy`.
8. Copy the Worker URL.
9. In the GoalOS app, open **AI Studio**, paste the Worker URL and click **Save endpoint**.
10. Click **Test**.

The Worker calls the Responses API with `store: false` and strict structured output. It also verifies the domain-bound wallet signature and rechecks current direct AGI Club ownership or current direct AGIALPHA balance before each AI request.

### Local secure route

1. Unzip `GoalOS_UVSI2_v7_Local_Secure.zip`.
2. Double-click the launcher for macOS or Windows, or run the Linux launcher.
3. Enter the API key in the hidden terminal prompt.
4. Open `http://127.0.0.1:8788/`.
5. Set the AI Studio endpoint to `http://127.0.0.1:8788`.

The key remains only in the Python process and is discarded when the server closes. This local server binds only to `127.0.0.1`; the browser gate remains the local access boundary.

## D. Customize safely

Edit only `config.js` for public configuration:

- AI endpoint;
- session length;
- revalidation cadence;
- official links.

Do not change the qualifying token contract or threshold unless the access constitution itself changes. Never put secrets in `config.js`.

## E. Custom domain

After the Pages deployment works:

1. Add the custom domain under **Settings → Pages**.
2. Configure DNS as instructed by GitHub.
3. Enable HTTPS.
4. Reverify the wallet signature flow because the receipt is domain-bound.
5. Add the custom origin to the secure AI backend’s `ALLOWED_ORIGINS`.

## F. Production hardening for confidential missions

GitHub Pages is public-static infrastructure. For protected customer data or consequential operations, add:

- authenticated server sessions;
- organizational identity and role-based access;
- encrypted evidence storage;
- independent proof custody;
- server-side release signing;
- secrets management;
- audit logging;
- incident response;
- tested rollback and continuity;
- legal, privacy, security and professional review.
