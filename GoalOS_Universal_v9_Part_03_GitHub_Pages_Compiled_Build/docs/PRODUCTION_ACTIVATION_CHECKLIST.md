# Production Activation Checklist — GoalOS Universal v9

## Release and build

- [ ] Clean `npm ci`, typecheck, tests, build, link validation, preflight, and dependency audit pass.
- [ ] Release hashes and manifest are recorded.
- [ ] No source maps, private keys, seed phrases, production credentials, or unresolved placeholders are present.
- [ ] Custom domain and support contacts are correct.

## AGI Club direct-ENS access

- [ ] Direct unwrapped owner succeeds.
- [ ] Official Ethereum Mainnet Name Wrapper owner with current expiry succeeds.
- [ ] Expired wrapped name fails.
- [ ] Transferred former owner fails and new owner succeeds.
- [ ] Wrong MetaMask account fails.
- [ ] Parent, nested, and wrong-role names fail before membership activation.
- [ ] Cancelled, pending, mobile, QR, and account-reselection flows are understandable.
- [ ] The signed message displays the exact application URL and legal-bundle hash.
- [ ] Periodic revalidation revokes access after transfer or expiry.

## Enterprise and local

- [ ] Public, internal, confidential, and restricted deployment guidance is approved.
- [ ] The enterprise acceptance receipt works.
- [ ] The one-file application opens without a server and makes no external application calls.
- [ ] Private hosting, IdP, access logging, endpoint, backup, retention, and incident controls are approved where required.

## Legal, security, and operations

- [ ] Counsel approval recorded.
- [ ] Security review and threat model recorded.
- [ ] Incident contact, support contact, recovery owner, and emergency publication/revocation authority recorded.
- [ ] Publication checklist and claim boundary completed.
- [ ] Live acceptance report signed by the operator.

Only after all applicable gates pass should the operator use the designation **Production Activated**.
