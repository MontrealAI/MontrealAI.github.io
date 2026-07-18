# External acceptance status

## Completed in the release build

- JavaScript syntax validation
- CSP reconciliation and inline-script removal
- Desktop and mobile browser QA
- Mocked unwrapped ENS Registry owner path
- Mocked wrapped ENS Name Wrapper owner path
- Mocked expired and wrong-owner rejection paths
- Link, JSON, Office archive, secret-pattern, and clean-extraction checks
- Deterministic IPFS CAR and ENS contenthash packet
- Deploy, operator/legal, and complete ZIP packages

## External gates that cannot be honestly completed by an offline build environment

- Controlled-wallet live Ethereum Mainnet signature and ownership acceptance for one unwrapped and one wrapped direct `*.club.agi.eth` name
- Transfer, expiry, account change, network change, gateway, and incident drills on the production origin
- Independent legal opinion or counsel sign-off
- Independent security assessment or penetration test

Until these gates are recorded, use the designation **deployment-ready production release candidate**, not “legally approved”, “security certified”, or “production activated”.
