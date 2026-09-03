# ASI.ETH Mainnet Activation Runbook

This runbook turns the published MCAP-001 architecture into an ENS-resolvable Machine Civilization root. It deliberately separates public specification, owner signature, independent verification and operational admission.

## Gate 0 — Merge and verify the public release

Before signing any Ethereum transaction:

1. Merge the reviewed release into the GitHub Pages branch.
2. Confirm that each canonical URL returns HTTP 200 and the expected content.
3. Validate all JSON documents against their published schemas.
4. Record the release commit and SHA-256 hashes of:
   - `manifest.json`
   - `MCAP-001.md`
   - `COVENANT.md`
   - `registry.json`
   - `ens-records.json`
5. Confirm that no file represents proposed components as active.

## Gate 1 — Review current ENS control

In the ENS Manager for `asi.eth`, independently verify:

- registrant and manager/controller;
- resolver address;
- expiry date;
- existing records;
- whether the name is wrapped;
- any existing fuses or permissions;
- hardware-wallet path and transaction simulation;
- recovery assumptions.

Do not burn fuses or lock resolver controls during the Genesis activation.

## Gate 2 — Set the root discovery records

Use the record set in `ens-records.json`. The minimum Genesis records are:

| Key | Value |
|---|---|
| `description` | `ASI.ETH — the addressable root of proof-gated Machine Civilizations. Memory may cross; authority must requalify.` |
| `url` | `https://montrealai.github.io/asi/` |
| `com.github` | `MontrealAI` |
| `com.twitter` | `Montreal_AI` |
| `org.montrealai.mcap.id` | `MCAP-001` |
| `org.montrealai.mcap.version` | `0.1.0` |
| `org.montrealai.mcap.status` | `genesis-publication` |
| `org.montrealai.mcap.manifest` | `https://montrealai.github.io/asi/manifest.json` |
| `org.montrealai.mcap.covenant` | `https://montrealai.github.io/asi/COVENANT.md` |
| `org.montrealai.mcap.registry` | `https://montrealai.github.io/asi/registry.json` |
| `org.montrealai.mcap.successor` | `successor.asi.eth` |
| `org.montrealai.mcap.doctrine` | `Memory may cross. Authority must requalify.` |
| `org.montrealai.mcap.publisher` | `MONTREAL.AI` |
| `org.montrealai.mcap.updated` | `2026-09-03` |

Draft-compatible records may then be added with explicit draft labeling:

| Key | Value |
|---|---|
| `agent-context` | compact JSON from `ens-records.json` |
| `agent-endpoint[web]` | `https://montrealai.github.io/asi/` |

Do not set `agent-endpoint[mcp]`, `agent-endpoint[a2a]` or `agent-registration[...]` until the corresponding live service or registry identity has been independently reviewed.

## Gate 3 — Publish activation evidence

Immediately after confirmation:

1. Save the transaction hash and block number.
2. Resolve every record independently through at least two clients or libraries.
3. Record the resolver address and supported interfaces.
4. Capture a machine-readable post-transaction snapshot.
5. Update `ens-records.json` from `owner-signature-required` to `active`.
6. Add a Chronicle event containing the transaction, block, record-set hash and reviewer.
7. Publish the exact release commit that the records resolve to.

No public statement should say “active” before these checks pass.

## Gate 4 — Activate the first functional subnames

Create subnames only when their functions exist. Recommended activation order:

1. `successor.asi.eth`
2. `covenant.asi.eth`
3. `registry.asi.eth`
4. `chronicle.asi.eth`
5. `proof.asi.eth`
6. `verifier.asi.eth`
7. `admission.asi.eth`
8. `policy.asi.eth`
9. `rollback.asi.eth`
10. `treasury.asi.eth`

Each active subname must publish:

- a description;
- canonical URL or contenthash;
- role;
- version;
- status;
- controller disclosure;
- relevant policy or evidence reference;
- Chronicle reference.

## Gate 5 — Separate control

Before operational authority exists:

- the current successor must not unilaterally control `verifier.asi.eth`;
- the claimant must not unilaterally control `proof.asi.eth` and `admission.asi.eth`;
- `chronicle.asi.eth` must not be erasable by the current successor alone;
- `treasury.asi.eth` must use bounded controls, timelocks and recovery;
- rollback authority must remain usable if the successor is compromised.

A different address is not sufficient evidence of independence. Publish controllers, signers, incentives, conflicts and recovery paths.

## Gate 6 — Qualify the first successor

The first operational `successor.asi.eth` admission requires:

1. A declared mission.
2. A frozen candidate commitment.
3. A published evaluation protocol.
4. Fresh evidence.
5. Independent verifier references.
6. Conflict disclosures.
7. A proof result and limitations.
8. A bounded authority scope and expiry.
9. Monitoring and rollback.
10. A Chronicle event.

Until then, `successor.asi.eth` should remain A0: discoverable and observable, without consequential authority.

## Gate 7 — Earn recognition

Recognition is demonstrated through use, not asserted by branding. Publish:

- one reference implementation;
- one external verifier exercise;
- one reproducible proof package;
- one rollback exercise;
- one successor or requalification event;
- one third-party integration resolving ASI.ETH records;
- one external citation of MCAP-001;
- one public limitation or failure report with corrective action.

## Final activation test

ASI.ETH is functioning as an addressable root when an independent client can start from `asi.eth` and deterministically discover:

- the canonical manifest;
- the governing Covenant;
- the namespace registry;
- the current successor and status;
- proof and verifier surfaces;
- admission and authority state;
- the Chronicle;
- material limitations.

It becomes a recognized Machine Civilization architecture only when independent parties can inspect, implement, challenge and cite those surfaces.
