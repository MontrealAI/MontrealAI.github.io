# AGI Club Access and Tier Standard — GoalOS AMM Ω v4

## Membership credential

Production access is granted only after the connected wallet proves control of a **direct ENS subname** with the exact shape:

`<member>.club.agi.eth`

The verifier MUST reject deeper descendants such as `x.member.club.agi.eth`. It checks Ethereum Mainnet and supports both:

- unwrapped names, where the ENS Registry owner is the connected wallet; and
- wrapped names, where the ENS Registry owner is the ENS Name Wrapper and `NameWrapper.ownerOf(namehash)` is the connected wallet.

A wallet signature proves current possession. No transaction, gas payment, token approval, or asset transfer is requested.

## Tier authority

Owning a qualifying direct subname grants **Pioneer** access by default. Higher tiers are assigned by the AGI Club operator through the server-side tier registry. Member-controlled ENS text records are never trusted as authoritative tier assignments.

Canonical tier-registry key: `tier:<normalized-name>`.

Supported values:

- `pioneer`
- `business`
- `sovereign`
- `agent`
- `node`

## Capability matrix

| Tier | Included capability |
|---|---|
| Pioneer | Core business formation, offer architecture, pipeline mathematics, forecast, plan export |
| Business | Pioneer + sales assets, Lead Lab, proof-producing delivery, proposal and recurring-revenue tools |
| Sovereign | Complete commercial machine + proof flywheel, gated autonomy, Audit Room, capability reserve, priority institutional execution |
| Agent | Core commercial context + proof-job factory, Evidence Docket, ProofBundle, machine-readable agent workflows |
| Node | Agent capabilities + validator, replay, sentinel, audit, and runtime/settlement-handoff surfaces |

## Session rules

- Short-lived, signed, HttpOnly session cookie.
- Membership is rechecked at every new login.
- A transferred or revoked name loses access after session expiry, or earlier if an operator invalidates the session.
- Member routes are served only after the edge worker validates the session.
- Public preview assets never contain the protected full application.

## Important boundary

Client-side hiding is not access control. Production exclusivity depends on server-side route protection. The included local demo mode exists only for QA and operator review.
