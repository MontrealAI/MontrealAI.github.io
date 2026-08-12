# Sovereign Alpha Foundry operating guide

## 1. Freeze a mission before choosing an algorithm

Define the objective, beneficiary, incumbent, best credible alternatives, hard failures, rights, proof budget, authority ceiling and rollback. The Router must not infer a mission from an algorithm preference.

## 2. Route a portfolio—not one favourite

The Router always retains the incumbent and a transparent rules baseline. It then adds candidates according to mission structure:

- strong formal rules → deterministic architecture;
- abundant accepted history → behaviour cloning / offline learning;
- sequential decisions → Q-learning or PPO-style candidate;
- learned-model replay → Dyna-Q;
- delayed consequences plus credible simulation → MuZero-style world model and MCTS;
- safety-critical action → external action shield and constrained learning;
- multi-agent roles → PettingZoo / MAPPO adapter;
- continuous control → SAC / Dreamer-style adapter;
- complete architecture search → MAP-Elites / quality-diversity archive.

## 3. Formation modes

- **Quick**: smoke test, economical baseline, early rejection.
- **Standard**: stronger training and matched proof suitable for a synthetic board demonstration.
- **Deep**: larger reference run; still synthetic unless connected to protected real mission evidence.

## 4. The algorithm tournament

The core backend trains:

- Incumbent Workflow;
- Deterministic Rules;
- Behaviour Cloning;
- Tabular Q-Learning;
- Dyna-Q;
- MuZero-Lite;
- Shielded MuZero-Lite.

The same fresh-case seeds are used for paired comparison. Training cases and fresh cases are separated. The fresh-seed list is committed by SHA-256 in the record.

## 5. Why MuZero may lose

MuZero-style planning can add value when hidden dynamics and delayed effects matter. It can still lose because:

- the mission is solved sufficiently by rules;
- historical imitation is more sample-efficient;
- the simulator does not represent reality well;
- the world model is undertrained;
- the action space or reward is poorly grounded;
- its compute and maintenance burden exceeds the gain;
- a hard gate fails.

A Foundry that always selects MuZero is not an Alpha Foundry. It is an algorithm marketing system.

## 6. Proof-adjusted Alpha

The Foundry selects the maximum conservative residual advantage, not the maximum development score. It subtracts:

- training burden;
- inference burden;
- architecture complexity;
- complete operating cost;
- Proof Debt;
- transfer degradation;
- Gym-to-reality basis risk.

Every candidate must also satisfy zero-tolerance constraints.

## 7. Promotion boundary

A synthetic tournament may produce `SPECIALIST_ASI_GATE_SIMULATION_PASS`. This is not a real Specialist ASI claim. Promotion to a real mission-bounded state requires an exact frozen release, protected production-representative work, independent evaluator custody, complete denominator and accountable acceptance.
