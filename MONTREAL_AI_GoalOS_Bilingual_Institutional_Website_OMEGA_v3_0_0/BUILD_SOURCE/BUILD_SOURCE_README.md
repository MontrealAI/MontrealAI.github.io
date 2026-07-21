# Build source — v3.0.0

The shipped static output is the authoritative release artifact.

- `build_montreal_goalos_site.py` preserves the original bilingual generator baseline.
- `finalize_v2.py` and `finalize_v2_release.py` preserve the v2 transformation provenance.
- The v3 release was completed as a reviewed static upgrade over the verified v2 baseline: local-safe navigation, Optimal Business Model integration, Mission Lab, non-confidential communications boundary, v3 visual system, bilingual parity and expanded QA.
- `package_release.py` deterministically generates the v3 payload manifest, GitHub overlay manifest, SHA-256 register, split ZIPs, clean-extraction tests and reproducibility checks.

Run packaging only from a disposable verified build directory. The package script does not deploy to GitHub or activate enterprise services.
