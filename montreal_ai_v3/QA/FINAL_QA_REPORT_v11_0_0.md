# Final QA report — MONTREAL.AI v11.0.0

## Verdict

**PASS — static production package complete. Live-origin publication and origin-level verification remain owner-controlled operations.**

## Estate

- HTML surfaces: **136**
- Exact English/French route pairs: **68**
- Source records represented: **39 / 39**
- Preliminary archive: **541 entries (540 files and 1 directory)**
- Generic French source placeholders remaining: **0**

## Static gate

- Issues: **0**
- Warnings: **0**
- Canonical and reciprocal hreflang: **PASS**
- Local references and dead anchors: **PASS**
- JavaScript syntax: **PASS**
- French substantive parity: every pair at or above the 0.95 word-ratio threshold, with matching section and H2 structure

## Chromium gate

- Desktop surfaces rendered: **136 / 136**
- Mobile surfaces rendered: **136 / 136**
- Desktop issues: **0**
- Mobile issues: **0**
- French runtime leakage pages: **0**
- Mobile menus: **124 / 124 passed**
- Representative interactions: **8 / 8 passed**
- Language directions resolved: **136 / 136**, covering all 68 pairs

The representative interactions cover the English and French command palettes, GoalOS eight-stage proof architecture, 39-record research filtering, and the browser-local SHA-256 verifier.

## Browser-method boundary

The execution environment blocks direct navigation to `file://`, localhost and the synthetic host. Chromium rendering therefore used `page.set_content()` with filesystem assets intercepted through a synthetic same-site origin. The production CSP was removed only from the in-memory QA document because `set_content()` retains an `about:blank` origin. Production files retain their CSP. Language destinations were resolved with browser-equivalent URL semantics and every target page was rendered independently.

## Claims boundary

Passing this QA gate means the static estate is internally coherent, bilingual, renderable and packageable. It does not certify external facts, customer outcomes, Mainnet state, security posture, legal compliance, live settlement or the continued availability of external links.
