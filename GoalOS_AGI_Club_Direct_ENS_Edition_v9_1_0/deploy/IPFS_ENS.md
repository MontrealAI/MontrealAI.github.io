# IPFS + ENS deployment

IPFS + ENS contenthash is the canonical decentralized publication path. GitHub Pages may remain an HTTPS mirror.

## Publish

1. Verify the release manifest and SHA-256 checksums.
2. Generate or use the included CAR file.
3. Pin the root CID through at least two independent pinning/storage arrangements or one independently operated node plus one provider.
4. Retrieve the site from an independent gateway and verify the root CID.
5. Review the generated ENS contenthash packet.
6. Submit the resolver `setContenthash` transaction through the wallet or Safe controlling the chosen website ENS name.
7. Verify resolution through more than one compatible gateway/browser path.

## Important

- A CID proves content identity, not truth, security, legality, confidentiality, or availability.
- GitHub/IPFS static files are inspectable. The ENS membership gate controls normal use, not code secrecy.
- Never put secrets, private keys, RPC credentials, or confidential customer data in the static bundle.
- The member’s activation signature uses the actual gateway/origin displayed in MetaMask. Changing gateways may require a new session signature.
