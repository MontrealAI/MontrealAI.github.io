# Production Go-Live Checklist

## Package and repository

- [ ] Deployment ZIP SHA-256 verified.
- [ ] Clean-room verification report reviewed.
- [ ] Repository diff reviewed in a dedicated branch and pull request.
- [ ] Existing unrelated routes and repository settings preserved.
- [ ] No private vault, credential, customer secret or untouched owner archive entered the deploy diff.

## Live bilingual integrity

- [ ] All **66** EN/FR exact counterpart switches tested on the live origin.
- [ ] `/sovereign-orchestration.html` and `/fr/sovereign-orchestration.html` tested, including all six states and keyboard navigation.
- [ ] CANON-40 visible in both canonical-corpus pages and machine-readable data.
- [ ] French Commission, forecast, canon filtering and generated dynamic states reviewed.
- [ ] Qualified Québec counsel/translator reviewed material French legal and privacy surfaces.

## Metadata, infrastructure and security

- [ ] Canonical, `hreflang`, `x-default`, sitemap, RSS, robots and `llms.txt` behaviour verified.
- [ ] `montreal.ai` redirect verified.
- [ ] Security response headers configured and tested where supported.
- [ ] No material console, mixed-content or network errors observed.

## Real-device and accessibility

- [ ] Safari and Firefox reviewed.
- [ ] iOS and Android reviewed on physical devices.
- [ ] Keyboard-only use reviewed.
- [ ] VoiceOver and NVDA or equivalent screen-reader review completed.
- [ ] 200%/400% zoom and forced-colours reviewed.
- [ ] Live Core Web Vitals and major performance regressions measured.

## Claims and publication authority

- [ ] Production announcement does not claim customer acceptance, realized economics, certification, third-party subordination or endorsement without evidence.
- [ ] Responsible publisher approves the final live state.
- [ ] Version tag, deployment date and rollback commit recorded.
