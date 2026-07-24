# GitHub Pages deployment guide

## Canonical destination

Publish this release at:

`https://montrealai.github.io/goalos-global-jurisdiction-funding-fiscal-intelligence-omega-agi-club-owner-access/`

The repository is `MontrealAI/MontrealAI.github.io`, using its configured GitHub Pages publishing branch.

## Use the upload-ready ZIP

The upload-ready ZIP contains exactly:

```text
.nojekyll
goalos-global-jurisdiction-funding-fiscal-intelligence-omega-agi-club-owner-access/
```

1. Unzip the upload-ready package locally.
2. Open the repository root on the GitHub Pages publishing branch.
3. Upload the root `.nojekyll` file and the entire `goalos-global-jurisdiction-funding-fiscal-intelligence-omega-agi-club-owner-access` directory.
4. Commit the files without renaming the directory or its internal files.
5. Wait for GitHub Pages to publish, then open the canonical URL above.
6. Connect an Ethereum wallet, use Ethereum Mainnet, enter the first label of an exact direct `label.club.agi.eth` name, accept the owner-access conditions and verify.

Every public deployment file is below 25 MB. No file splitting is required.

## What not to upload

Do not upload the Complete Preservation & Audit Package ZIP into the public repository. Keep that archive offline as the controlled audit and preservation record.

Do not upload private keys, seed phrases, RPC credentials, unpublished datasets, signing secrets or confidential materials.

## Access model

The static institution performs browser-local, wallet-mediated ownership verification. It accepts only:

- an unwrapped name for which the ENS Registry owner is the connected wallet; or
- a wrapped name for which the ENS Registry owner is the official Ethereum Mainnet ENS Name Wrapper at `0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401`, and both `ownerOf(namehash)` and `getData(namehash).owner` equal the connected wallet, with any reported expiry still current.

Delegates, resolver controllers, token approvals, operators, managers, nested names, prior owners, non-mainnet wrappers and expired wrapped names do not qualify.

## Static-hosting boundary

GitHub Pages publishes the HTML, JavaScript and legal pages publicly. The gate is a strong ordinary-use, contractual and evidentiary control for cooperating users, not server-side confidentiality or unbypassable DRM. No confidential payload is embedded in this release.
