# Mission-to-Gym Compiler

A custom finite mission specification contains:

- `mission_id`, `name`, `description`;
- states and observations;
- actions;
- weighted initial states;
- probabilistic transitions;
- vector rewards;
- hard constraints;
- terminal states;
- transparent expert and rule policies;
- mission features and episode counts.

The compiler rejects unknown states/actions, inconsistent observation widths, invalid probabilities and unsupported reward or constraint keys. A compiled transition model remains a hypothesis subject to Gym-to-reality basis risk and independent fresh proof.
