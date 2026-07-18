# Controlled Live-Wallet Acceptance Runbook

## Purpose

This runbook closes the final environment-specific production gate for GoalOS Autonomous Money Machine Ω — AGI Club Direct ENS Member Edition v9.0.

The software package can compile, test, build, and verify simulated Registry/Wrapper cases offline. A genuine production activation still requires the deployed HTTPS or IPFS origin and controlled MetaMask wallets because a real wallet signature and live Ethereum Mainnet state cannot be fabricated inside an offline build environment.

## What the member should experience

1. Open the deployed GoalOS URL.
2. Enter only the club label, such as `elite`.
3. Select **Connect MetaMask & Verify**.
4. Choose the MetaMask account that owns `elite.club.agi.eth`.
5. Approve one readable, free signature.
6. Confirm that GoalOS displays the exact direct name, selected wallet, ownership path, expiry status, legal-bundle hash, resources, and the complete Money Machine.

The member must not need ETH, gas, a token approval, an API key, a separate GoalOS account, or a manual network switch.

## Before testing

- Deploy the exact reviewed `dist/` directory to a controlled HTTPS origin or IPFS subdomain gateway.
- Confirm that `config.json` points to Ethereum Mainnet, the official ENS Registry, the official Ethereum Mainnet Name Wrapper, any separately reviewed additional Mainnet wrapper, and at least two working HTTPS RPC endpoints. Never configure the Sepolia Name Wrapper on Mainnet.
- Prepare controlled labels and wallets in `LIVE_ACCEPTANCE_CASES.example.json`.
- Never place seed phrases or private keys in any file, issue, screenshot, chat, or test report.
- Make sure the browser is using the production build—not a development server with modified code.

## Required cases

| Case | Expected result |
|---|---|
| Direct unwrapped name; current Registry owner selected | Access allowed |
| Official Mainnet-wrapper direct name; current owner; future expiry | Access allowed |
| Additional reviewed Mainnet-wrapper direct name, if one exists | Access allowed; otherwise mark N/A with rationale |
| Expired wrapped name | Access denied |
| Transferred name tested from former owner | Access denied |
| Valid name tested from a different account | Access denied |
| Nested name such as `x.elite.club.agi.eth` | Input rejected before wallet signature |

## Case procedure

For each case:

1. Open `operator/LIVE_WALLET_ACCEPTANCE_CONSOLE.html` from the same deployed origin or from the release package.
2. Enter the label and controlled wallet address.
3. Select **Open GoalOS verification**.
4. In MetaMask, select only the intended controlled account.
5. Read the signature message. Confirm that it includes:
   - the exact direct membership name;
   - the selected wallet;
   - Ethereum Mainnet (1);
   - the deployed site origin;
   - the legal version and legal-bundle SHA-256;
   - a nonce and issue time;
   - the statement that this is not a transaction and cannot move assets.
6. Approve or cancel according to the case.
7. Record the observed result and evidence in the acceptance console.
8. For allowed cases, download the access receipt and confirm that its name, wallet, ownership path, wrapper, expiry, origin, legal hash, message, signature, and timestamps are correct.
9. Export the final JSON acceptance report.

## Transfer and expiry revalidation

For one allowed membership:

1. Verify access successfully.
2. Transfer the direct ENS name to another controlled wallet, or use a controlled test name whose expiry can be changed safely.
3. Wait for the configured revalidation interval or select **Recheck membership**.
4. Confirm that the member interface locks and requires verification from the current owner.

Do not use a production member’s valuable name for destructive or expiry testing without explicit authorization.

## RPC failover

- Temporarily replace the first RPC URL with an unreachable controlled endpoint in a staging branch.
- Confirm that verification succeeds through the next configured RPC.
- Restore the approved production configuration and rebuild before launch.

## Mobile acceptance

Test MetaMask Mobile using a real HTTPS URL or IPFS subdomain gateway. Confirm:

- app handoff or QR connection;
- account selection;
- readable signature;
- return to GoalOS;
- no horizontal overflow;
- member resources and Money Machine open correctly.

## Final production record

The production record should contain:

- release ZIP SHA-256;
- deployed commit hash;
- GitHub Pages URL and/or IPFS CID;
- legal-bundle version/hash;
- production `config.json` hash;
- the complete acceptance-matrix results;
- access-receipt samples with personal information redacted where appropriate;
- reviewer names and approval time;
- known limitations and rollback decision.

Production activation is approved only when every required case matches the expected result and legal/security reviewers have approved the actual deployment facts.


## Optional legacy/custom Mainnet wrapper condition

No legacy/custom wrapper is trusted by default. Run this optional case only if the deployed `config.json` intentionally contains another independently verified Ethereum Mainnet wrapper. Otherwise record **N/A** with rationale. The Sepolia Name Wrapper must never be used in a Mainnet configuration.
