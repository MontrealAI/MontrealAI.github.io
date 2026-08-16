# Access Gate Specification

## AGIALPHA route
The broker reads `balanceOf(signingWallet)` from `0xA61a3B3a130a9c20768EEBF97E21515A6046a1fA`. Only the direct wallet balance counts. The threshold is `1,000,000 × 10^18` raw units.

## AGI Club route
The broker constructs `label.club.agi.eth`, computes the ENS namehash, reads the Registry owner and, when wrapped, the NameWrapper effective owner and expiry. The effective owner must equal the signing wallet.

## Challenge and session
A one-time KV-backed message binds application, origin, path, wallet, route, time and nonce. After signature and current eligibility verification, the broker returns an opaque short-lived session and the content key. The message states `Authority created: NONE`.
