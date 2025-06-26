[See docs/DISCLAIMER_SNIPPET.md](../docs/DISCLAIMER_SNIPPET.md)

# Project Documentation

[α‑AGI Insight Demo](alpha_agi_insight_v1/index.html)

## Building the React Dashboard

The React dashboard sources live under `alpha_factory_v1/demos/alpha_agi_insight_v1/src/interface/web_client`. Build the static assets before serving the API:

```bash
pnpm --dir alpha_factory_v1/demos/alpha_agi_insight_v1/src/interface/web_client install
pnpm --dir alpha_factory_v1/demos/alpha_agi_insight_v1/src/interface/web_client run build
```

The compiled files appear in `alpha_factory_v1/demos/alpha_agi_insight_v1/src/interface/web_client/dist` and are automatically served when running `uvicorn alpha_factory_v1.demos.alpha_agi_insight_v1.src.interface.api_server:app` with `RUN_MODE=web`.

## Ablation Runner

Use `alpha_factory_v1/core/tools/ablation_runner.py` to measure how disabling individual innovations affects benchmark performance. The script applies each patch from `benchmarks/patch_library/`, runs the benchmarks with and without each feature and generates `docs/ablation_heatmap.svg`.

```bash
python -m alpha_factory_v1.core.tools.ablation_runner
```

The resulting heatmap visualises the pass rate drop when a component is disabled.

## Manual Workflows

The repository defines several optional GitHub Actions that are disabled by
default. They only run when the repository owner starts them from the GitHub
UI. These workflows perform heavyweight benchmarking and stress testing.

To launch a job:

1. Open the **Actions** tab on GitHub.
2. Choose either **📈 Replay Bench**, **🌩 Load Test** or **📊 Transfer Matrix**.
3. Click **Run workflow** and confirm.

Each workflow checks that the person triggering it matches
`github.repository_owner`, so it executes only when the owner initiates the
run.

## Macro-Sentinel Demo

A self-healing macro risk radar powered by multi-agent α‑AGI. The stack ingests
macro telemetry, runs Monte-Carlo simulations and exposes a Gradio dashboard.
See the [alpha_factory_v1/demos/macro_sentinel/README.md](../alpha_factory_v1/demos/macro_sentinel/README.md)
for full instructions.

## α‑AGI Insight v1 Demo

`docs/alpha_agi_insight_v1` provides a self-contained HTML demo that
visualises capability forecasts with Plotly. The GitHub Actions workflow
copies this directory into the generated `site/` folder, serves it on GitHub
Pages and deploys the page automatically. Visit
[the published demo](https://montreal-ai.github.io/AGI-Alpha-Agent-v0/alpha_agi_insight_v1/)
to preview it.

The old `static_insight` directory has been removed in favour of this
official static demo.

To update the charts, edit `forecast.json` and `population.json` and rebuild
the site:

```bash
./scripts/build_insight_docs.sh
```

This helper fetches all assets, compiles the browser bundle and runs `mkdocs build`.
Open `site/alpha_agi_insight_v1/index.html` in your browser to verify the
changes before committing. Alternatively run `./scripts/preview_insight_docs.sh`
to build and serve the demo locally on `http://localhost:8000/`.
