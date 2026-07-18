# Controlled Live-Wallet Acceptance

Use controlled wallets and direct `*.club.agi.eth` names on the final deployed origin.

Required cases:

1. Direct unwrapped owner — allow.
2. Official Mainnet Name Wrapper owner, current expiry — allow.
3. Expired wrapped subname — deny.
4. Transferred name — deny former owner; allow new owner.
5. Wrong account — deny.
6. Parent name, nested name, and wrong role — deny.
7. MetaMask cancellation — show safe, non-technical recovery.
8. Existing pending MetaMask request — explain and recover.
9. Mobile/QR handoff — complete.
10. RPC primary outage — fallback succeeds or fails closed with a clear message.
11. Revalidation after transfer/expiry — lock the member utility.
12. Access receipt — correct exact URL, wallet, ENS name, block, ownership path, legal version, hash, and signature.

Store the completed report with release hashes and operator approvals.
