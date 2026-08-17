# Access Gate Specification — Direct Browser Verification

## AGIALPHA route

The browser calls `balanceOf(connectedWallet)` on `0xA61a3B3a130a9c20768EEBF97E21515A6046a1fA` through the connected Ethereum wallet. Only the current direct balance counts. The threshold is `1,000,000 × 10^18` raw units.

## AGI Club route

The browser constructs `label.club.agi.eth`, computes the ENS namehash, reads the ENS Registry owner and, when wrapped, reads the NameWrapper effective owner and expiry. The effective owner must equal the connected wallet.

## Receipt and session

After eligibility passes, the wallet signs a readable message binding application, version, origin, path, wallet, route, issue time, expiry and nonce. The receipt states `Authority created: NONE`. The app rechecks eligibility on focus, account changes, network changes, before publication delivery and every 5 minutes.
