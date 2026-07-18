# Migration from the v7 Encrypted Vault Scaffold

The v7 page displayed “Activation paused” because it required a deployed access-registry address and an encrypted-vault adapter. v8 uses direct ENS ownership for the complete Club Edition.

## Replace the current GitHub Pages folder

1. Back up the existing `GoalOS_AGI_Club_Web3_Encrypted_Vault_Scaffold_v7/` folder.
2. Delete its published contents.
3. Copy the contents of the v8 deployable package into the same folder, or publish v8 under a new folder.
4. Ensure `.nojekyll` is present.
5. Commit and push.
6. Open `index.html`, connect the controlled owner wallet, enter the direct label, and complete the activation signature.
7. Run `operator/deployment-checker.html` with the same controlled wallet.

## No registry deployment is required

Basic Club Edition access is based only on:

- the current connected MetaMask account;
- Ethereum Mainnet;
- the ENS Registry owner for an unwrapped name; or
- the Name Wrapper owner plus unexpired wrapped state.

An optional operator-signed tier manifest may add profiles. It cannot block basic membership unless the operator intentionally pins a signer and signs a disabled entry.
