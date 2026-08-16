# Successor Ω — The Illustrated Guided Institutional Journey
## Encrypted GitHub Pages access edition · Browser-upload-safe v1.0.1

The public repository contains the access interface and AES-256-GCM ciphertext only. Readable publication files and the content key are not committed.

## Eligibility

Access is granted when the signing Ethereum Mainnet wallet is either:

1. the current direct owner of one exact single-label `*.club.agi.eth` name; or
2. the current direct holder of at least 1,000,000 official AGIALPHA at `0xA61a3B3a130a9c20768EEBF97E21515A6046a1fA`.

No approval, transfer, payment, staking, locking, burning or custody is requested. Balances are not aggregated. Access creates no Successor Ω authority.

## Browser-upload-safe protected assets

The Complete Publication Suite is divided into six independently authenticated encrypted parts. The reader automatically downloads, verifies, decrypts and reassembles them into the original ZIP. Every public repository file is below 25 MiB.

See `docs/BROWSER_UPLOAD_GUIDE.md`.

## Deployment

1. Deploy the separately packaged Cloudflare access Worker.
2. Set its private `ETHEREUM_RPC_URL` and `CONTENT_KEY_B64` secrets.
3. Replace `access.brokerUrl` in `config.js`.
4. Upload this folder to `MONTREALAI/successor-omega-illustrated-journey`.
5. In GitHub **Settings → Pages**, select **GitHub Actions**.

Suggested URL: `https://montrealai.github.io/successor-omega-illustrated-journey/`

Never commit the private Worker package, content key, RPC credentials or decrypted files.
