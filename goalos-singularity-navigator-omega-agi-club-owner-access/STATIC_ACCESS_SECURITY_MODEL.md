# Static Access Security Model

- Ethereum network: Mainnet only.
- ENS Registry: `0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e`.
- Official Mainnet Name Wrapper: `0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401`.
- Eligibility: current direct owner of one exact `label.club.agi.eth` name.
- Signature: EIP-712 preferred; domain-bound `personal_sign` fallback.
- Session: 30-minute absolute limit; 10-minute inactivity lock; five-minute revalidation; event-driven relock.
- Boundary: GitHub Pages source is public. The gate is current-owner verification, signed provenance, conditional licensing and community access—not confidential server-side DRM.
