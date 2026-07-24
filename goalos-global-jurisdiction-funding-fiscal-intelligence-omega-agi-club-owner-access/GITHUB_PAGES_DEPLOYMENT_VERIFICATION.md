# GitHub Pages Deployment Verification

**Release:** v10.0.0-GHP1  
**Canonical URL:** https://montrealai.github.io/goalos-global-jurisdiction-funding-fiscal-intelligence-omega-agi-club-owner-access/  
**Result:** **PASS**

## Deployment shape

The upload-ready archive contains:

```text
.nojekyll
goalos-global-jurisdiction-funding-fiscal-intelligence-omega-agi-club-owner-access/
```

The root `.nojekyll` file bypasses Jekyll processing for the prebuilt static site. The deployment directory contains `index.html`, `404.html`, the Legal Center, schemas, notices and release documentation.

## Verified controls

- Every deployment file is below 25 MB.
- No server, database, Worker, KV, R2, cookie or `/api/access/` route is required.
- No external JavaScript, CSS, font, model API or runtime network request is used.
- The wallet provider is the only chain-access boundary.
- Exact one-label `label.club.agi.eth` validation is enforced.
- Ethereum Mainnet is enforced.
- Only Registry direct ownership or the official Ethereum Mainnet ENS Name Wrapper `0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401` qualifies.
- `ownerOf` and `getData.owner` must agree for wrapped names.
- Non-mainnet wrappers, delegates, approvals, operators, resolver control, nested names and expired names are rejected.
- The access receipt explicitly records `serverAuthoritative: false` and the public-static protection boundary.

## Publication boundary

The static source is public after deployment. The gate controls ordinary use, supplies signed evidence and applies the owner-only licence; it is not confidentiality or unbypassable DRM.
