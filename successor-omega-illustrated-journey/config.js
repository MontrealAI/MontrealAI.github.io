/* Public, browser-only configuration. No Worker or private RPC is required. */
window.SUCCESSOR_OMEGA_CONFIG = Object.freeze({
  appId: 'successor-omega-illustrated-journey',
  version: '1.0.2',
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
  ens: {
    suffix: 'club.agi.eth',
    registry: '0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e',
    nameWrappers: [
      '0x0635513f179D50A207757E05759CbD106d7dFcE8',
      '0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401'
    ]
  },
  access: {
    mode: 'direct-browser-verification',
    sessionMinutes: 30,
    revalidateMinutes: 5,
    inactivityMinutes: 20
  },
  delivery: {
    manifest: 'protected/manifest.json',
    materialSource: 'delivery.js',
    encryptedAtRest: true,
    serverSideConfidentiality: false
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
