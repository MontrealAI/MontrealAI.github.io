# Empirical Verifier Mutation Lab

The Verifier Foundry now attacks actual synthetic proof receipts rather than checking only whether a control is named.

The reference mutations include:

- unsupported acceptance;
- non-compensable constraint violation hidden behind high reward;
- deleted constraint state;
- fabricated evidence coverage;
- hidden human rescue;
- omitted complete cost;
- protected-case leakage;
- candidate-controlled or collusive validation.

A candidate cannot pass the verifier hard gate unless the mutation detection rate is at least 95 percent and false acceptance remains zero in the protected reference run.

This is reference calibration. Real deployments require independently designed attacks, professional review where reserved, and ongoing calibration after environmental change.
