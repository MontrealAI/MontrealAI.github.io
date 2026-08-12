# Deployment guide

## A. GitHub Pages — no terminal

1. Download `GoalOS_UVSI3_v8_GitHub_Pages.zip`.
2. Unzip it.
3. Create a GitHub repository.
4. Upload every extracted item to the repository root, including `.github`, `.nojekyll`, `index.html`, scripts, styles and folders.
5. Commit to `main`.
6. Open **Settings → Pages**.
7. Under **Build and deployment**, select **GitHub Actions**.
8. Open **Actions** and wait for `Deploy GoalOS UVSI3 to GitHub Pages` to finish.
9. Open the published URL with a wallet-enabled browser.

The public deployment supports the complete browser experience and deterministic Alpha portfolio simulator. It does not execute GPU or confidential training.

## B. Local executable Foundry — no package installation

1. Download `GoalOS_UVSI3_v8_Local_Foundry.zip`.
2. Unzip it.
3. Double-click the launcher for your operating system.
4. Open `http://127.0.0.1:8788` if it does not open automatically.
5. Run the executable backend tournament.

Python 3.10+ is required. The core has no mandatory third-party dependency.

## C. Docker protected backend

1. Copy `Dockerfile`, `docker-compose.yml`, `alpha-foundry/` and `site/` to a private server.
2. Replace `CHANGE_ME_TO_A_LONG_RANDOM_SECRET`.
3. Set the exact allowed GitHub Pages origin.
4. Run `docker compose up -d --build`.
5. Put the service behind HTTPS and a private network or firewall.
6. Do not expose the training service directly to the public Internet when it contains confidential mission data.

## D. Wallet-gated production gateway

1. Download `GoalOS_UVSI3_v8_Alpha_Foundry_Gateway.zip`.
2. Configure `ALLOWED_ORIGINS` and `FOUNDRY_ORIGIN`.
3. Set Worker secrets `GATEWAY_SECRET` and `ETHEREUM_RPC_URL`.
4. Set the same `GOALOS_GATEWAY_SECRET` on the private backend.
5. Deploy the Worker.
6. Paste the Worker URL into **Alpha Foundry Ω → Protected training endpoint**.

## E. Production hardening checklist

- organizational identity and role-based access;
- encrypted evidence storage;
- separate development and proof custodians;
- immutable run receipts;
- exact release signing;
- compute and proof-capital quotas;
- model/data/tool rights records;
- incident response and kill switches;
- canary and rollback drills;
- independent calculation of realized Alpha;
- periodic requalification.


## F. Optional secure AI Studio backend

The Alpha Foundry training backend and the OpenAI-powered AI Studio are separate services. The default product works without an OpenAI key. To enable AI Studio:

- deploy `GoalOS_UVSI3_v8_Secure_AI_Backend.zip` as a Cloudflare Worker, keep `OPENAI_API_KEY` in Worker secrets, set the exact allowed origin, then paste the Worker URL into **AI Studio**; or
- unzip `GoalOS_UVSI3_v8_Local_Secure_AI.zip`, run the operating-system launcher, enter the key in the hidden terminal prompt, and use `http://127.0.0.1:8790` as the AI Studio endpoint.

The browser never stores the OpenAI key. The secure Worker revalidates current wallet eligibility before each AI request.
