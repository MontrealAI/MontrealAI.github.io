# Successor Ω — The Illustrated Guided Institutional Journey
## GitHub Pages direct-wallet access edition · v1.0.2

This edition uses the same **browser-direct on-chain verification model** as the live Sovereign Specialist ASI Training Toolkit. It requires no Cloudflare Worker, no private RPC secret and no legacy broker setting configuration.

## Eligibility

The connected Ethereum Mainnet wallet must be either:

1. the current direct owner of one exact single-label `*.club.agi.eth` name; or
2. the current direct holder of at least 1,000,000 official AGIALPHA at `0xA61a3B3a130a9c20768EEBF97E21515A6046a1fA`.

The app reads current on-chain state through the connected wallet provider, then requests a readable, domain-bound sign-in receipt. It requests no approval, transfer, payment, staking, locking, burning, deposit or custody.

## Deployment

Upload this folder to `MONTREALAI/successor-omega-illustrated-journey`, enable GitHub Actions under **Settings → Pages**, and hard-refresh the published page. No second service is required.

## GitHub browser uploader

Every repository file remains below 25 MiB. The Complete Suite is stored as six browser-safe authenticated parts and is reassembled automatically after access.

## Security boundary

GitHub Pages is a public static host. This edition provides the same client-side eligibility gate as the reference Toolkit; it does not provide server-side confidentiality. The AES-GCM packaging provides corruption detection and casual delivery separation, not a private-secret boundary. A recipient can retain or redistribute any file after opening it.
