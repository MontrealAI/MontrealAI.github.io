# Independent Proof Custody

The definitive release separates formation from final adjudication at process level.

Formation may train, replay and evolve. After the exact release is frozen, the proof record is submitted to a separate custodian worker that imports no training or optimizer module. It checks:

- release freeze state;
- zero proof-time learning calls;
- authority reset to `NONE`;
- sealed proof-plane declaration;
- candidate hard-gate state;
- signed verdict integrity.

The local reference worker signs with HMAC-SHA256. Production deployment must place the proof secret, protected cases and scorer credentials under separately controlled infrastructure and identities.
