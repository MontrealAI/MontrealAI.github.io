/* GoalOS UVSI3 v9 public configuration. Never place secrets in this file. */
window.GOALOS_CONFIG = Object.freeze({
  appId: 'goalos-uvsi3-v9',
  version: '9.0.0-UVSI3',
  edition: 'Sovereign Specialist ASI Training Toolkit Ω',
  releaseDate: '2026-08-17',
  empiricalState: 'UVSI3 · deterministic synthetic reference institution',
  nextGate: 'UVSI4 · Fresh Reality',
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
    allowLocalDemo: false,
    securityBoundary: 'The official static interface verifies eligibility. Public GitHub Pages is not confidential DRM.'
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
    paperA4: 'research/GoalOS_Navigator_SEIZE_Gym_Successor_Omega_v6_A4.pdf',
    paperWeb: 'research/GoalOS_Navigator_SEIZE_Gym_Successor_Omega_v6_Web.pdf',
    visualAbstract: 'research/GoalOS_Navigator_SEIZE_Gym_Successor_Omega_v6_Visual_Abstract.pdf',
    boardBrief: 'research/GoalOS_Navigator_SEIZE_Gym_Successor_Omega_v6_Board_Brief.pdf',
    evidenceDossier: 'research/GoalOS_Navigator_SEIZE_Gym_Successor_Omega_v6_Evidence_Dossier.pdf',
    guidedJourney: 'research/Successor_Omega_Illustrated_Guided_Journey_WEB.pdf',
    proofRun: 'research/Successor_Omega_Proof_Run_001_A4.pdf',
    masterclass: 'research/Successors_Omega_Masterclass_v9_0_0.pdf'
  }
});
