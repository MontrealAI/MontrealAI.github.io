// Provider-neutral encrypted-vault hook.
// Production operators may replace this file with an independently reviewed decentralized
// threshold-decryption adapter and set vaultMode accordingly. The default build deliberately
// fails closed rather than pretending that client-side UI gating makes public static assets secret.
window.GoalOSVaultProvider = {
  async unlock() {
    throw new Error('No audited decentralized encrypted-vault adapter is configured. Use open-web3 mode or follow the encrypted-vault deployment runbook.');
  },
};
