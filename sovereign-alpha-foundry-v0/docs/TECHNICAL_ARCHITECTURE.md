# Technical architecture

## Public control plane

The GitHub Pages PWA provides access qualification, Navigator Ω, mission constitution, Successor Manifold, SEIZE, AGI Jobs, browser simulation, board views and export. It contains no server secret and should not contain confidential customer evidence.

## Protected training plane

The Python Foundry server provides:

- `/api/health`;
- `/api/catalog`;
- `/api/route`;
- `/api/tournament`;
- versioned run files and SHA-256 manifests.

The dependency-free core serves the site and performs actual training on synthetic mission environments. For production, run the backend privately with `GOALOS_GATEWAY_SECRET` and exact `GOALOS_ALLOWED_ORIGINS`.

## Access-revalidating gateway

The optional Cloudflare Worker:

1. validates the domain-bound signed access receipt;
2. rechecks current direct AGI Club or AGIALPHA qualification on Ethereum Mainnet;
3. enforces exact origins and rate limits;
4. forwards eligible requests to the private Foundry with a shared gateway secret.

The gateway is not the proof plane and does not grant authority.

## Independent proof plane

Production certification must use a separate service and evidence custodian. It should hold protected cases, scorer internals, thresholds, validator credentials, release signatures and rollback packages outside the Formation Foundry.

## Chronicle and authority plane

Tournament results may create a Promotion Request or candidate manifest. They do not create an Authority Envelope. Chronicle admission, authority, value allocation, impairment and requalification remain separate accountable decisions.
