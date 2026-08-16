/* Public configuration only. Never place private keys, RPC credentials or the content key here. */
window.SUCCESSOR_OMEGA_CONFIG = Object.freeze({
  appId: 'successor-omega-illustrated-journey',
  version: '1.0.1',
  edition: 'The Illustrated Guided Institutional Journey',
  releaseDate: '2026-08-16',
  ethereumChainId: '0x1',
  ethereumChainName: 'Ethereum Mainnet',
  token: {
    symbol: 'AGIALPHA',
    contract: '0xA61a3B3a130a9c20768EEBF97E21515A6046a1fA',
    decimals: 18,
    minimumWhole: '1000000'
  },
  ens: { suffix: 'club.agi.eth' },
  access: {
    /* Replace after deploying the separately packaged access Worker. */
    brokerUrl: 'https://REPLACE-WITH-YOUR-WORKER.workers.dev',
    sessionMinutes: 15,
    revalidateMinutes: 5,
    inactivityMinutes: 20,
    requireBroker: true
  },
  deployment: {
    expectedOrigin: 'https://montrealai.github.io',
    repository: 'successor-omega-illustrated-journey',
    suggestedUrl: 'https://montrealai.github.io/successor-omega-illustrated-journey/'
  },
  links: {
    website: 'https://montreal.ai/',
    email: 'president@montreal.ai',
    tokenExplorer: 'https://etherscan.io/token/0xA61a3B3a130a9c20768EEBF97E21515A6046a1fA'
  }
});
