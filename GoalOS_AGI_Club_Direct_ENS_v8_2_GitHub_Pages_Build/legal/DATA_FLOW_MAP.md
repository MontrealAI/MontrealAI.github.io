# Data Flow Map

1. User enters a club label locally.
2. MetaMask returns a selected wallet account and signs a versioned access message.
3. The browser recovers the signer locally.
4. The browser queries Ethereum Mainnet RPC endpoints for ENS Registry / Name Wrapper ownership and expiry.
5. The browser stores the last label and access receipt in localStorage.
6. GoalOS Money Machine plans, drafts, and preferences may be stored locally by the embedded application.
7. No application-backend account or analytics database is included by default.
8. GitHub Pages/IPFS gateways, RPC providers, MetaMask, browsers, support email, and external links process data under their own terms.
