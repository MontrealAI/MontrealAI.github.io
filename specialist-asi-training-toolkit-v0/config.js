/* GoalOS UVSI3 v8 - public configuration only. Never place secrets in this file. */
window.GOALOS_CONFIG = Object.freeze({
  appId: 'goalos-uvsi3-v8-8',
  version: '8.8.0-UVSI3',
  edition: 'Sovereign Specialist ASI Training Toolkit Ω',
  releaseDate: '2026-08-13',
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
    sessionMinutes: 30,
    revalidateMinutes: 5,
    inactivityMinutes: 20,
    allowLocalDemo: true
  },
  foundry: {
    endpoint: '',
    localEndpoint: 'http://127.0.0.1:8788',
    defaultMission: 'building_operations',
    defaultMode: 'quick'
  },
  ai: {
    endpoint: '',
    defaultAction: 'full_cycle',
    maxEvidenceChars: 120000,
    modelLabel: 'Secure AI backend not configured'
  },
  links: {
    website: 'https://montreal.ai/',
    email: 'president@montreal.ai',
    paperA4: 'research/AGI_Jobs_OpenAI_Gym_Successor_Omega_v2_0_0_FLAGSHIP_A4.pdf',
    paperWeb: 'research/AGI_Jobs_OpenAI_Gym_Successor_Omega_v2_0_0_FLAGSHIP_A4.pdf',
    visualAbstract: 'research/AGI_Jobs_OpenAI_Gym_Successor_Omega_v2_0_0_VISUAL_ABSTRACT.pdf',
    boardBrief: 'research/AGI_Jobs_OpenAI_Gym_Successor_Omega_v2_0_0_BOARD_EXECUTIVE_BRIEF.pdf',
    evidenceDossier: 'research/AGI_Jobs_OpenAI_Gym_Successor_Omega_v2_0_0_EVIDENCE_REPRODUCIBILITY_A4.pdf'
  }
});
