# Live Ethereum Mainnet acceptance checklist

Do not label the release “production activated” until the deployed HTTPS/IPFS origin passes these controlled tests.

## Origin and supply chain

- [ ] The deployed files match the release manifest and SHA-256 record.
- [ ] The GitHub Pages artifact and IPFS CID contain the same intended release.
- [ ] The production origin is correct and protected from repository/domain takeover.
- [ ] No secrets, credentials, analytics, or unexpected external scripts are present.

## MetaMask and name grammar

- [ ] Desktop MetaMask connects and chain switching works.
- [ ] MetaMask mobile/browser path works.
- [ ] A direct label such as `elite` becomes `elite.club.agi.eth`.
- [ ] The parent name, nested names, and other namespaces are rejected.
- [ ] The readable activation message contains the correct origin, wallet, name, chain, nonce, expiry, and legal versions.
- [ ] No activation transaction, token approval, payment, or asset transfer is requested.

## ENS ownership

- [ ] An unwrapped direct name owned by the connected wallet passes.
- [ ] The same unwrapped name fails from a different wallet.
- [ ] A wrapped direct name owned by the connected wallet passes.
- [ ] An expired wrapped direct name fails.
- [ ] Transfer to a new wallet moves access after revalidation.
- [ ] Approvals, resolver addresses, and operator permissions do not substitute for ownership.

## Session and legal

- [ ] Terms/Privacy/Risk acceptance is required.
- [ ] The activation signature verifies locally.
- [ ] A different account or network invalidates/resets the session.
- [ ] Session expiry requires re-verification.
- [ ] Legal-version change requires renewed acceptance.
- [ ] Activation receipt downloads correctly.

## Member functions

- [ ] Complete Money Machine opens after verification.
- [ ] Direct access to member pages without a valid session is blocked in normal use.
- [ ] Three-Customer Proof Run saves/exports locally and synthetic records do not count.
- [ ] Execution request signs and downloads without a transaction.
- [ ] Financial model downloads.
- [ ] End Session clears the current session.

## Independent review

- [ ] Qualified security review completed.
- [ ] Qualified privacy/legal review completed.
- [ ] Incident/rollback and contenthash-update rehearsal completed.
- [ ] Operator contact, entity, address, and privacy-responsible person are finalized.
