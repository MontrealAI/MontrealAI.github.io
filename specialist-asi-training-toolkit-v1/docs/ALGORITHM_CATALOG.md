# Algorithm and architecture catalog

## Executable dependency-free reference market

| Candidate | Institutional role |
|---|---|
| Incumbent Workflow | Current reference process |
| Deterministic Rules | Transparent lowest-burden baseline |
| Behaviour Cloning | Learn accepted demonstrations |
| IQL-Lite | Offline expectile-style value learning |
| CQL-Lite | Conservative offline action values |
| Q-Learning | Model-free sequential baseline |
| AT²PO-Lite | Multi-turn backward credit assignment |
| Dyna-Q | Learned model plus replay |
| Dreamer-Lite | Imagined tabular trajectories |
| MuZero-Lite | Latent dynamics plus tree search |
| Shielded MuZero-Lite | MuZero logic plus external action shield |
| DUPLEX-Lite | Bounded semantic extraction plus deterministic planning |
| TRINITY-Lite | Learned specialist router |
| AsyncThink-Lite | Organizer-worker parallel proposals |
| OpenEvolve-Lite | Quality-diverse policy population |
| ThetaEvolve-Lite | Formation-only test-time evolution |
| HGM-Lite | Present fitness plus descendant potential |

## Optional production adapters

The release defines installation boundaries for Gymnasium, PettingZoo, MO-Gymnasium, RLlib, veRL, MCTX, DreamerV3, OpenEvolve, PyRibs/MAP-Elites and mission-selected symbolic solvers. These packages are not bundled and no frontier-scale training result is claimed.
