[See docs/DISCLAIMER_SNIPPET.md](../docs/DISCLAIMER_SNIPPET.md)

# Hosting Instructions

This project uses [MkDocs](https://www.mkdocs.org/) to build the static documentation.
The generated site is hosted at <https://montreal-ai.github.io/AGI-Alpha-Agent-v0/alpha_agi_insight_v1/>.

## Quick Deployment

1. Run `./scripts/deploy_insight_demo.sh`.
2. When it finishes, open the printed URL or push to `main` to trigger the “📚 Docs” workflow.
3. Verify the page at `https://<org>.github.io/AGI-Alpha-Agent-v0/alpha_agi_insight_v1/`.

## Prerequisites

- **Python 3.11 or 3.12**
- `mkdocs` and `mkdocs-material`
- **Node.js 20+** *(optional, only for building the React dashboard)*
- `unzip` to extract `insight_browser.zip`

Install MkDocs:

```bash
pip install mkdocs mkdocs-material
```

Before building the demo, ensure optional Python packages are available:

```bash
python scripts/check_python_deps.py
python check_env.py --auto-install
```

## Build the Insight Demo

The static browser bundle lives under
`alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1`. Install the
Node dependencies then create the distribution archive:

```bash
cd alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1
npm install
npm run build:dist
```

`npm run build:dist` produces `insight_browser.zip`. Extract the archive and copy
its contents into `docs/alpha_agi_insight_v1` so MkDocs can include the files:

```bash
unzip -o insight_browser.zip -d ../../../docs/alpha_agi_insight_v1
```

The helper script `scripts/build_insight_docs.sh` automates the steps above.
Run it from the repository root to build the bundle, refresh
`docs/alpha_agi_insight_v1` and generate the site.


## Building the Site

Run the following from the repository root:

```bash
mkdocs build
```

This generates the HTML under `site/`. Verify that the Insight demo was copied
correctly:

```bash
ls site/alpha_agi_insight_v1
```
Ensure `lib/workbox-sw.js` resides under `site/alpha_agi_insight_v1/lib/` because the service worker expects the file relative to `index.html`.

Serve the site locally to test it:

```bash
python -m http.server --directory site 8000
```

Then browse to <http://localhost:8000/alpha_agi_insight_v1/>. Direct `file://`
access is unsupported due to the service worker; use a minimal HTTP server or
GitHub Pages.

The "📚 Docs" workflow
[`docs.yml`](../.github/workflows/docs.yml) automatically runs
`scripts/build_insight_docs.sh`, builds the site and pushes the result to the
`gh-pages` branch.

### Manual Publish

To trigger a one-off deployment outside of CI run:

```bash
./scripts/publish_insight_pages.sh
```

This wrapper script rebuilds the browser bundle, regenerates the MkDocs site and
uses `mkdocs gh-deploy` to push the contents of `site/` to the `gh-pages` branch.
Use it when testing changes locally or publishing from a personal fork.

## Publishing to GitHub Pages

When changes land on `main` or a release is published, `docs.yml` pushes the
`site/` directory to the `gh-pages` branch. GitHub Pages serves the result at
`https://montreal-ai.github.io/AGI-Alpha-Agent-v0/alpha_agi_insight_v1/`.
The repository’s `docs/index.html` file redirects to this path so that opening
`https://montreal-ai.github.io/AGI-Alpha-Agent-v0/` automatically loads the
demo.
The standard [project disclaimer](DISCLAIMER_SNIPPET.md) applies.

## Verifying Deployment

Confirm the workflow is enabled under **Actions** and that
[`docs.yml`](../.github/workflows/docs.yml) specifies
`permissions: contents: write`. Run the "📚 Docs" workflow from the GitHub UI or
push to `main` to trigger it. The initial run creates the `gh-pages` branch.
After it finishes, browse to
<https://montreal-ai.github.io/AGI-Alpha-Agent-v0/alpha_agi_insight_v1/> and
check that the insight demo loads.
