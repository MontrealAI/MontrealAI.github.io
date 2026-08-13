# Production training adapters

The dependency-free core is intentionally inspectable. Larger deployments can replace individual candidate components while preserving the same mission, proof and authority constitution.

- **Gymnasium:** maintained environment interface.
- **PettingZoo:** role-specific multiplayer environments.
- **MO-Gymnasium:** vector-valued reward environments.
- **RLlib:** distributed classical RL, offline RL and multi-agent orchestration.
- **veRL:** language-model RL and agentic post-training.
- **MCTX:** JAX AlphaZero, MuZero and Gumbel-MuZero search.
- **DreamerV3:** neural imagined-trajectory world model.
- **OpenEvolve:** executable program evolution.
- **PyRibs:** quality-diversity and MAP-Elites archive.
- **Symbolic solvers:** PDDL, CP-SAT, SMT or domain-specific formal planning for DUPLEX-style candidates.

Adapters must expose exact versions, resource accounting, state manifests, proof receipts and rollback. They may not gain access to protected proof cases or authority credentials.
