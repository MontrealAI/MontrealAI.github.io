# IPFS Deployment

1. Run `npm ci && npm run qa && npm run export:ipfs`.
2. Publish the complete `ipfs-export/` directory as one immutable build.
3. Record the CID and release SHA-256 manifest.
4. Pin with Storacha, Kubo, IPFS Cluster, Pinata, Filebase, or other reviewed infrastructure. Multiple independent pins are recommended for resilience.
5. Prefer a subdomain gateway (`https://<CID>.ipfs.<gateway>/`) for origin isolation.
6. Optionally publish the CID through DNSLink or an ENS contenthash after review.
7. Re-run member verification and exact-URL signature acceptance on the chosen gateway/domain; the signed message binds the page URL.

**Confidentiality:** Ordinary public IPFS content is public. Use encryption and a separately reviewed access-condition/key-release architecture for confidential files.
