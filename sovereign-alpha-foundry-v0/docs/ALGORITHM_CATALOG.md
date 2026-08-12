# Algorithm catalog and claim boundaries

| Candidate | Role | Best fit | Principal limitation |
|---|---|---|---|
| Incumbent Workflow | Real reference | Current process | May be slow or impaired |
| Deterministic Rules | Transparent simple baseline | Formal, bounded missions | Brittle under novel regimes |
| Behaviour Cloning | Warm-start from expert action | Abundant accepted examples | Repeats demonstrator limits |
| Q-Learning | Model-free sequential learning | Small discrete environments | Sample inefficient, brittle without shielding |
| Dyna-Q | Learned transition replay | Small model-based planning | Tabular scaling limit |
| MuZero-Lite | Representation + learned dynamics/reward + policy/value + MCTS | Delayed sequential decisions with a useful simulator | Educational tabular reference, not neural MuZero |
| Shielded MuZero-Lite | MuZero-Lite plus external action mask | Safety-sensitive discrete experiments | Safety shield may reduce performance and does not establish production safety |

## Optional protected-compute adapters

The release includes adapter contracts and deployment guidance for:

- Gymnasium environments;
- PettingZoo multi-agent environments;
- RLlib PPO, SAC, APPO, IMPALA, IQL, CQL and DreamerV3;
- MCTX AlphaZero, MuZero and Gumbel MuZero search;
- MAPPO specialist-role coordination;
- MO-Gymnasium vector rewards;
- external Sampled MuZero and EfficientZero implementations.

These adapters are not vendored into the dependency-free core and were not represented as executed during QA.
