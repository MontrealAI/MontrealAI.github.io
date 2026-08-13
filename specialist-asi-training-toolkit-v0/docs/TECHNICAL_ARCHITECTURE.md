# Technical architecture

## Five planes

1. **Public control plane:** Navigator, mission qualification, wallet gate, public-safe architecture and browser simulation.
2. **Protected opportunity plane:** confidential mission constitution, rights, economics, proof capital and stop-loss.
3. **Isolated formation plane:** Mission Gym, curriculum, training, world models, specialists, evolutionary search and sandboxes.
4. **Independent proof plane:** exact frozen release, sealed cases, separate credentials, independent scoring and signed verdict.
5. **Chronicle and authority plane:** accountable admission, Authority Envelope, monitoring, rollback, impairment, revocation and requalification.

## Backend modules

- `compiler.py`: strict finite Mission Gym compiler.
- `curriculum.py`: formation and protected curriculum commitments.
- `verifiers.py`: independent verifier registry and adversarial audit.
- `router.py`: mission-structure-based algorithm routing.
- `algorithms.py`: executable reference candidates.
- `release.py`: exact release freezing and cryptographic manifest.
- `tournament.py`: formation market and one-way proof cycle.
- `proof.py`: lower-confidence comparison, hard gates and proof-adjusted Alpha.
- `architecture_market.py`: complete architecture bundles and quality-diversity archive.
- `safety.py`: collective-agent risk stress taxonomy.
- `chronicle.py`: append-only lineage records and descendant proposal.
- `server.py`: protected local HTTP service.

## Non-learning proof guarantee

The evaluation runner never calls candidate update methods. Every policy is deep-copied and committed before protected evaluation. Fresh and transfer seed commitments are recorded before results are produced.
