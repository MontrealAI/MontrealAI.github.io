(() => {
'use strict';

const APP = {
  name: 'GoalOS Singularity Navigator Ω + SEIZE',
  version: '7.0.0',
  edition: 'Executable Verified Succession Institution · UVSI2',
  storageKey: 'goalos_uvsi2_core_v7_0_0',
  releaseDate: '2026-08-11',
  author: 'Vincent Boucher',
  role: 'President, MONTREAL.AI & QUEBEC.AI',
  website: 'https://montreal.ai/',
  email: 'president@montreal.ai'
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const clamp = (value, min = 0, max = 100) => Math.max(min, Math.min(max, Number(value) || 0));
const round = (value, digits = 1) => Number(value ?? 0).toFixed(digits);
const mean = values => values.length ? values.reduce((a,b) => a + b, 0) / values.length : 0;
const stdev = values => {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - m) ** 2, 0) / (values.length - 1));
};
const median = values => {
  if (!values.length) return 0;
  const a = [...values].sort((x,y) => x-y);
  const i = Math.floor(a.length / 2);
  return a.length % 2 ? a[i] : (a[i-1] + a[i]) / 2;
};
const sum = values => values.reduce((a,b) => a + b, 0);
const nowISO = () => new Date().toISOString();
const dateOnly = value => new Date(value).toISOString().slice(0,10);
const addDays = (value, days) => { const d = new Date(value); d.setDate(d.getDate() + days); return dateOnly(d); };
const uuid = prefix => `${prefix || 'gxs'}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,9)}`;
const safeName = value => String(value || 'GoalOS_Project').normalize('NFKD').replace(/[^a-zA-Z0-9_-]+/g,'_').replace(/^_+|_+$/g,'').slice(0,96) || 'GoalOS_Project';
const list = value => Array.isArray(value) ? value : String(value || '').split(/\n|;/).map(v => v.trim()).filter(Boolean);
const pct = value => `${Math.round(Number(value) || 0)}%`;
const money = value => new Intl.NumberFormat('en-CA',{style:'currency',currency:'CAD',maximumFractionDigits:0}).format(Number(value) || 0);
const deepClone = object => JSON.parse(JSON.stringify(object));
const authorityRank = level => ['A0','A1','A2','A3','A4'].indexOf(level);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function canonicalJSON(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJSON).join(',')}]`;
  return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonicalJSON(value[key])}`).join(',')}}`;
}
function fnv1a(text) {
  let hash = 2166136261;
  for (let i = 0; i < String(text).length; i += 1) {
    hash ^= String(text).charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}
const hashObject = object => fnv1a(canonicalJSON(object)).toString(16).padStart(8,'0').toUpperCase();
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
const normalish = rng => (rng()+rng()+rng()+rng()+rng()+rng()-3) / 1.224744871;
const pick = (rng, array) => array[Math.floor(rng() * array.length)];
const shuffle = (rng, array) => {
  const result = [...array];
  for (let i = result.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
};

const I18N = {
  en: {
    navWelcome:'Welcome',navWelcomeHint:'Start',navMission:'Mission',navMissionHint:'Constitute',navManifold:'Successor Manifold',navManifoldHint:'Map',navGradient:'Advantage Gradient',navGradientHint:'Navigate',navSeize:'SEIZE Underwriting',navSeizeHint:'Underwrite',navJobs:'Bounded AGI Jobs',navJobsHint:'Compile',navFormation:'Formation Gym',navFormationHint:'Compete',navRecursive:'Recursive Foundry',navRecursiveHint:'Improve',navProof:'Fresh Proof',navProofHint:'Prove',navAuthority:'Authority',navAuthorityHint:'Bound',navChronicle:'Chronicle',navChronicleHint:'Compound',navRequalify:'Requalification',navRequalifyHint:'Renew',navExport:'Export',navExportHint:'Package',
    guide:'Guide',reset:'Reset',candidateSystem:'Candidate System',simulationProven:'Simulation Proof Passed',admittedDemo:'Simulated Admission',
    save:'Save',continue:'Continue',run:'Run',freeze:'Freeze',export:'Export',import:'Import',close:'Close'
  },
  fr: {
    navWelcome:'Accueil',navWelcomeHint:'Départ',navMission:'Mission',navMissionHint:'Constituer',navManifold:'Variété des successeurs',navManifoldHint:'Cartographier',navGradient:'Gradient d’avantage',navGradientHint:'Naviguer',navSeize:'Souscription SEIZE',navSeizeHint:'Souscrire',navJobs:'Tâches AGI bornées',navJobsHint:'Compiler',navFormation:'Gym de formation',navFormationHint:'Comparer',navRecursive:'Fonderie récursive',navRecursiveHint:'Améliorer',navProof:'Preuve fraîche',navProofHint:'Prouver',navAuthority:'Autorité',navAuthorityHint:'Borner',navChronicle:'Chronicle',navChronicleHint:'Composer',navRequalify:'Requalification',navRequalifyHint:'Renouveler',navExport:'Exporter',navExportHint:'Emballer',
    guide:'Guide',reset:'Réinitialiser',candidateSystem:'Système candidat',simulationProven:'Preuve simulée réussie',admittedDemo:'Admission simulée',
    save:'Enregistrer',continue:'Continuer',run:'Exécuter',freeze:'Geler',export:'Exporter',import:'Importer',close:'Fermer'
  }
};
const tr = key => I18N[state?.meta?.lang || 'en']?.[key] || I18N.en[key] || key;

const PARAM_LABELS = {
  perception:'Perception',reasoning:'Reasoning',domain:'Domain depth',tools:'Tool use',evidence:'Evidence discipline',verifier:'Verifier independence',memory:'Chronicle memory',governance:'Governance',portability:'Portability',speed:'Speed',costEfficiency:'Cost efficiency',humanEfficiency:'Human efficiency',resilience:'Resilience',calibration:'Calibration',exploration:'Search diversity'
};
const METRIC_LABELS = {
  quality:'Mission quality',value:'Verified value',speed:'Speed',evidence:'Evidence',reliability:'Reliability',sovereignty:'Sovereignty',governance:'Governance',transfer:'Transfer',costScore:'Cost efficiency',humanScore:'Human burden efficiency',utility:'Composite utility'
};
const AUTHORITY = {
  A0:{name:'Observe',description:'Read, analyze and propose. No consequential action.'},
  A1:{name:'Recommend',description:'Produce ranked, evidenced options for accountable review.'},
  A2:{name:'Sealed sandbox',description:'Execute only inside an isolated environment with no external effect.'},
  A3:{name:'Reversible bounded action',description:'Take preapproved, monitored actions with explicit rollback.'},
  A4:{name:'Consequential action',description:'Affect money, rights, safety or material operations under independent controls.'}
};

const TEMPLATES = {
  building: {
    icon:'⌂',name:'Portfolio Building Operations',nameFr:'Opérations immobilières de portefeuille',short:'Predict, diagnose and govern portfolio-scale building operations.',shortFr:'Prévoir, diagnostiquer et gouverner les opérations d’un portefeuille immobilier.',
    gymmability:[100,95,95,96,88,94,92,90],profile:{physical:1,strategic:.25,financial:.4,software:.35,criticality:1,feedback:.92,transfer:.9},
    mission:{name:'Portfolio Building Operations Successor',beneficiary:'Property operations leadership, residents, tenants and asset owners',objective:'Convert authorized tenant requests, equipment alerts and inspection exceptions into evidence-grounded diagnoses, safe response plans and verified closures while preserving safety, legal, professional and major-capital authority for accountable humans.',incumbent:'Fragmented property-management, work-order, sensor and vendor process led by human operators.',alternatives:['Repaired incumbent with deterministic rules','General AI copilot','External property-operations platform','Specialist human–AI cell','Sovereign hybrid successor'],scope:'Routine and urgent building operations across approved properties; safety-critical and major-capital decisions remain outside initial authority.',criticalFailures:['Missed life-safety event','Unauthorized physical action','Unlawful access or tenant-rights violation','Unapproved major expenditure','Unsupported closure of a critical event'],constraints:['Approved properties and data only','Approved vendors only','No life-safety override','Evidence required for closure','Fail closed outside validated distribution'],authorityCeiling:'A3',proofBudget:175000},
    spec:{state:['Property and asset identity','Sensor and alarm state','Tenant or operator report','Weather and occupancy','Maintenance and vendor history','Budget and authority state'],observations:['Authorized work orders','Approved sensor feeds','Asset registry','Maintenance history','Weather and occupancy context'],actions:['Inspect sensor','Review asset history','Request diagnostic','Update causal model','Recommend intervention','Escalate','Dispatch approved vendor','Schedule preventive maintenance','Abstain or fall back'],rewards:['Safety and critical-event performance','First-time fix','Reduced recurrence','Time to accepted resolution','Evidence-complete closure','Uptime and tenant impact','Complete cost'],constraints:['No missed critical event','No action outside Authority Envelope','No unsupported closure','No unlawful access','No unapproved capital action']}
  },
  acquisition: {
    icon:'◇',name:'Acquisition Underwriting',nameFr:'Souscription d’acquisition',short:'Build and attack an evidence-grounded acquisition thesis.',shortFr:'Construire et attaquer une thèse d’acquisition fondée sur des preuves.',
    gymmability:[82,86,92,80,52,88,92,76],profile:{physical:.08,strategic:1,financial:1,software:.32,criticality:.85,feedback:.46,transfer:.72},
    mission:{name:'Acquisition Underwriting Successor',beneficiary:'Investment committee and corporate development leadership',objective:'Produce an evidence-bearing acquisition recommendation that identifies value drivers, contradictions, downside scenarios, integration risks and decision conditions while reserving binding investment authority to the accountable investment committee.',incumbent:'Human corporate-development workflow supported by external advisers and fragmented diligence workstreams.',alternatives:['Strong general-frontier-model workflow','External specialist diligence provider','Human–AI hybrid','Adversarial thesis/anti-thesis system','Sovereign hybrid diligence successor'],scope:'Internal diligence, evidence analysis, scenario construction and recommendation preparation; no external communication or binding action.',criticalFailures:['Critical false reassurance','Fabricated or untraceable evidence','Unauthorized counterparty communication','Transaction commitment or fund movement','Hidden use of unapproved data'],constraints:['Approved data room only','Every material claim cites evidence','Producer is not final verifier','Binding authority remains human','No self-expansion of permissions'],authorityCeiling:'A1',proofBudget:150000},
    spec:{state:['Target-company world model','Evidence and provenance state','Contradictions and unresolved assumptions','Buyer strategy and constraints','Market and integration scenarios','Diligence budget and time'],observations:['Approved data-room evidence','Authorized external sources','Buyer strategy','Historical acquisition outcomes','Specialist reports'],actions:['Inspect evidence','Reconcile records','Request decisive evidence','Build thesis','Build anti-thesis','Run counterfactual','Update valuation','Recommend proceed/condition/stop','Abstain or escalate'],rewards:['Material-risk recall','Contradiction detection','Evidence coverage','Decision quality','Information value','Cycle time','Senior-review burden','Complete economics'],constraints:['No fabricated evidence','No critical false reassurance above ceiling','No external action','No unapproved data','No self-certification']}
  },
  reconciliation: {
    icon:'≋',name:'Financial Reconciliation',nameFr:'Rapprochement financier',short:'Resolve high-volume financial exceptions with deterministic proof.',shortFr:'Résoudre les exceptions financières à grande échelle avec une preuve déterministe.',
    gymmability:[98,96,98,100,96,92,98,94],profile:{physical:.05,strategic:.3,financial:1,software:.75,criticality:.78,feedback:1,transfer:.93},
    mission:{name:'Financial Reconciliation Successor',beneficiary:'Finance leadership, controllers, auditors and affected business units',objective:'Classify and resolve authorized financial exceptions with complete source lineage, deterministic accounting checks, calibrated escalation and evidence-complete closure.',incumbent:'Manual analyst workflow across ledgers, bank records, invoices, approvals and spreadsheets.',alternatives:['Improved deterministic rules','General AI copilot','Specialist reconciliation platform','Human–AI exception cell','Sovereign hybrid reconciliation successor'],scope:'Approved accounts and exception classes; no payment release, ledger posting or write-off without explicit human authorization.',criticalFailures:['Incorrect release of a material exception','Hidden fraud signal','Unsupported journal or write-off','Unauthorized payment or posting','Missing source lineage'],constraints:['Approved accounts only','Deterministic identities must reconcile','Material exceptions escalate','No direct payment authority','Every closure carries a proof receipt'],authorityCeiling:'A2',proofBudget:90000},
    spec:{state:['Exception type and amount','Ledger, bank and invoice records','Approval chain','Materiality and fraud signal','Deadline and authority state'],observations:['Authorized ledger extracts','Bank records','Invoices','Approval records','Accounting policies'],actions:['Inspect source','Run identity check','Request missing document','Classify exception','Propose correction','Escalate material issue','Abstain','Close with proof'],rewards:['Correct resolution','Source completeness','Time to accepted closure','False-release avoidance','Reviewer burden','Complete cost'],constraints:['No material false release','No direct posting','No unsupported write-off','No missing lineage','No concealed fraud signal']}
  },
  software: {
    icon:'⌘',name:'Verified Software Engineering',nameFr:'Génie logiciel vérifié',short:'Find, fix, test and release software changes under proof.',shortFr:'Trouver, corriger, tester et publier des changements logiciels sous preuve.',
    gymmability:[100,98,98,100,100,100,98,96],profile:{physical:.02,strategic:.35,financial:.3,software:1,criticality:.72,feedback:1,transfer:.95},
    mission:{name:'Verified Software Engineering Successor',beneficiary:'Engineering leadership, developers, users and system owners',objective:'Resolve authorized software issues by reproducing the defect, generating and testing candidate repairs, proving non-regression and preparing an exact release while reserving production release authority to accountable humans.',incumbent:'Human software-development workflow supported by ordinary coding assistants and CI tools.',alternatives:['Improved human workflow','General coding agent','Specialist repair agent','Multi-agent repair and verifier cell','Sovereign hybrid engineering successor'],scope:'Approved repositories, sandboxes and test environments; no unapproved production release, credential access or external communication.',criticalFailures:['Security regression','Fabricated test result','Unauthorized production change','Data loss','Untraceable dependency modification'],constraints:['Sandbox first','Tests independent of producer','Exact diffs and dependencies','Rollback required','Human release authority'],authorityCeiling:'A3',proofBudget:120000},
    spec:{state:['Repository and issue state','Reproduction result','Dependency graph','Test and security state','Budget and authority'],observations:['Approved source tree','Issue report','Tests','Build logs','Dependency metadata'],actions:['Inspect code','Reproduce defect','Generate patch','Run tests','Run security check','Challenge patch','Prepare release','Escalate','Abstain'],rewards:['Issue resolution','Regression avoidance','Security','Test coverage','Cycle time','Review burden','Maintainability'],constraints:['No production release','No hidden network access','No fabricated test','No security regression','Exact rollback path']}
  },
  custom: {
    icon:'Ω',name:'Custom Mission',nameFr:'Mission personnalisée',short:'Define a narrow, valuable, measurable and proofable mission.',shortFr:'Définir une mission étroite, précieuse, mesurable et prouvable.',
    gymmability:[50,50,50,50,50,50,50,50],profile:{physical:.3,strategic:.6,financial:.5,software:.5,criticality:.7,feedback:.6,transfer:.6},
    mission:{name:'Custom Mission Successor',beneficiary:'Accountable institutional principal',objective:'Constitute, compare and prove a superior successor for one bounded mission.',incumbent:'Describe the current complete architecture.',alternatives:['Retain incumbent','Repair incumbent','Rent external capability','Form owned successor','Partner or acquire'],scope:'Define the exact mission scope and exclusions.',criticalFailures:['Define unacceptable failure'],constraints:['Define non-negotiable constraints'],authorityCeiling:'A1',proofBudget:50000},
    spec:{state:['Define mission state'],observations:['Define permitted observations'],actions:['Inspect','Recommend','Escalate','Abstain'],rewards:['Mission quality','Verified value','Evidence','Reliability'],constraints:['No unauthorized action']}
  }
};

const JOB_DEFINITIONS = [
  ['Mission Constitution','Frozen objective, principal, preserve/build/become constraints and prohibited actions.'],
  ['Incumbent Baseline','Accepted quality, cost, risk, time, human burden and authority state.'],
  ['Alternative Set','Best credible frontier, specialist, human, hybrid, software, partner, acquisition and no-action baselines.'],
  ['Rights and Data Constitution','Provenance, licences, confidentiality, retention, exclusions and reuse rights.'],
  ['Benchmark Constitution','Representative distribution, protected cases, scorer, critical errors and promotion threshold.'],
  ['Oracle and Teacher Study','Declared external intelligence, teaching signal, hidden-call prohibition and transfer hypothesis.'],
  ['Failure Taxonomy','Error classes, severity, causes, abstention requirements and adversarial cases.'],
  ['Architecture Frontier','Diverse complete candidate systems and non-dominated reserve options.'],
  ['Data and Experience Formation','Rights-cleared examples, simulations, expert demonstrations and quality controls.'],
  ['Specialist Model Formation','Adaptation, training, distillation or algorithmic construction with exact releases.'],
  ['Tool and Retrieval Formation','Mission-specific tools, deterministic checks, retrieval, planners and sandboxes.'],
  ['Verifier Formation','Independent evaluation harness, calibration, replay and protected scoring.'],
  ['Security and Privacy Proof','Threat model, data boundary, credential controls, isolation and incident response.'],
  ['Complete Economics','Formation, inference, human review, maintenance, requalification, dependency and unwind.'],
  ['Human Burden and Agency','Escalation, contestability, supervision, professional authority and role redesign.'],
  ['Resilience and Rollback','Failover, graceful degradation, portability, kill switch and known-good restoration.'],
  ['Protected Fresh Evaluation','Exact frozen challenger on unseen representative work with declared interventions.'],
  ['Independent Validation','Separated review of evidence, limitations, critical errors and complete denominator.'],
  ['Accountable Acceptance','Human or institutional decision on use for the stated purpose.'],
  ['Chronicle and Authority Admission','Scope, version, expiry, revocation, monitoring and Authority Envelope.'],
  ['Capital-to-Capacity and Requalification','Verified-value allocation, successor triggers and next fresh-work test.']
];

function defaultCandidates(templateKey) {
  const template = TEMPLATES[templateKey] || TEMPLATES.custom;
  const domainBoost = templateKey === 'software' ? 5 : templateKey === 'reconciliation' ? 4 : 0;
  return [
    {
      id:'incumbent',name:'Incumbent Architecture',kind:'incumbent',generation:0,parentId:null,selected:true,createdAt:nowISO(),notes:'Current accepted human and software architecture.',
      params:{perception:56,reasoning:59,domain:72+domainBoost,tools:48,evidence:57,verifier:55,memory:46,governance:73,portability:82,speed:43,costEfficiency:50,humanEfficiency:38,resilience:67,calibration:61,exploration:36},
      burden:{formation:8,operation:62,migration:4,proof:14,dependency:18,irreversibility:12}
    },
    {
      id:'general_ai',name:'General AI Copilot',kind:'beta',generation:0,parentId:null,selected:true,createdAt:nowISO(),notes:'Broadly rentable frontier capability under human review.',
      params:{perception:72,reasoning:78,domain:57+domainBoost,tools:69,evidence:55,verifier:42,memory:45,governance:61,portability:54,speed:84,costEfficiency:78,humanEfficiency:66,resilience:57,calibration:59,exploration:63},
      burden:{formation:18,operation:29,migration:13,proof:24,dependency:65,irreversibility:27}
    },
    {
      id:'specialist_cell',name:'Specialist Intelligence Cell',kind:'specialist',generation:0,parentId:null,selected:true,createdAt:nowISO(),notes:'Several bounded specialists coordinated under a human principal.',
      params:{perception:81,reasoning:84,domain:82+domainBoost,tools:78,evidence:76,verifier:70,memory:68,governance:76,portability:66,speed:72,costEfficiency:59,humanEfficiency:56,resilience:73,calibration:75,exploration:78},
      burden:{formation:49,operation:47,migration:34,proof:42,dependency:43,irreversibility:31}
    },
    {
      id:'executable',name:'Executable Mission System',kind:'executable',generation:0,parentId:null,selected:true,createdAt:nowISO(),notes:'Deterministic tools, programs, constraints and simulators around the mission.',
      params:{perception:68,reasoning:69,domain:77+domainBoost,tools:91,evidence:86,verifier:84,memory:63,governance:88,portability:78,speed:82,costEfficiency:74,humanEfficiency:75,resilience:84,calibration:87,exploration:46},
      burden:{formation:56,operation:31,migration:42,proof:35,dependency:27,irreversibility:24}
    },
    {
      id:'sovereign_hybrid',name:'Sovereign Hybrid Successor',kind:'sovereign',generation:0,parentId:null,selected:true,createdAt:nowISO(),notes:'Mission-owned orchestration, specialist intelligence, independent proof, Chronicle and portable authority controls.',
      params:{perception:87,reasoning:89,domain:88+domainBoost,tools:87,evidence:91,verifier:92,memory:89,governance:94,portability:91,speed:77,costEfficiency:66,humanEfficiency:73,resilience:92,calibration:91,exploration:88},
      burden:{formation:76,operation:43,migration:58,proof:64,dependency:19,irreversibility:20}
    }
  ].map(candidate => {
    Object.keys(candidate.params).forEach(key => { candidate.params[key] = clamp(candidate.params[key]); });
    return candidate;
  });
}

function defaultJobs() {
  return JOB_DEFINITIONS.map(([name, output], index) => ({
    id:index + 1,name,output,status:'not_started',owner:'Unassigned',evidence:'',updatedAt:null,
    phase:index < 4 ? 'Constitution' : index < 8 ? 'Underwriting' : index < 16 ? 'Formation' : index < 18 ? 'Proof' : index < 20 ? 'Admission' : 'Compounding'
  }));
}

function makeState(templateKey = 'building') {
  const template = TEMPLATES[templateKey] || TEMPLATES.custom;
  const candidates = defaultCandidates(templateKey);
  const projectId = `UVSI-${APP.releaseDate.replaceAll('-','')}-${Math.floor(100000 + Math.random()*900000)}`;
  return {
    meta:{projectId,lang:'en',createdAt:nowISO(),updatedAt:nowISO(),currentSection:'command',institutionGeneration:0,release:APP.version},
    mission:{template:templateKey,...deepClone(template.mission),gymmability:[...template.gymmability],constitutionFrozen:false,frozenAt:null},
    gym:{version:'ENV-1.0.0',formationSeed:20260810,proofSeed:99172026,transferSeed:77192026,formationCases:180,proofCases:260,transferCases:80,distribution:{normal:60,edge:25,adversarial:15},spec:deepClone(template.spec),customScenarios:[]},
    candidates,
    selectedCandidateId:'sovereign_hybrid',
    successorBook:candidates.map((candidate,index) => ({candidateId:candidate.id,position:index===0?'incumbent':index===4?'immediate':'challenger',thesis:candidate.notes,trigger:'Run matched formation and fresh proof.',status:'active'})),
    gradient:{candidateId:'sovereign_hybrid',results:[],computedAt:null},
    seize:{missionValue:3000000,probability:55,proofCost:template.mission.proofBudget,formationCost:450000,tailRisk:300000,optionValue:250000,durationMonths:4,halfLifeMonths:18,decision:'Not underwritten',score:null,frozen:false,frozenAt:null,nextEvidence:'Run exact incumbent reproduction and a matched formation arena.'},
    jobs:defaultJobs(),
    formation:{runId:null,results:[],caseCount:0,seed:null,championId:null,archive:[],completedAt:null},
    recursive:{generation:0,lineage:[],archive:[],history:[],console:[],institutionalMemory:0,formationCostMultiplier:1,settings:{childrenPerGeneration:8,mutationScale:7,noveltyWeight:.08,burdenWeight:.11,elitism:4},lastRunAt:null},
    proof:{frozenCandidateId:null,frozenAt:null,manifestHash:null,result:null,history:[],protectedLocked:true},
    authority:{level:'A1',owner:'Accountable human principal',expiry:addDays(nowISO(),90),permitted:['Read approved mission evidence','Produce internal analysis','Generate ranked recommendations','Escalate uncertainty'],prohibited:['External communication','Transaction or fund movement','Safety-critical action','Rights-affecting decision','Self-expansion of permissions'],stress:[],admission:{checks:{identity:false,proof:false,rights:false,rollback:false,owner:false,expiry:false},signedBy:'',signedAt:null,status:'not_admitted'}},
    chronicle:[],negativeCapability:[],
    requalification:{changes:[],reason:'',result:null,lastRunAt:null},
    evidence:{claims:[],files:[]}
  };
}

let state = loadState();
function loadState() {
  try {
    const stored = localStorage.getItem(APP.storageKey);
    if (!stored) return makeState('building');
    const parsed = JSON.parse(stored);
    return migrateState(parsed);
  } catch (error) {
    console.warn('State recovery failed', error);
    return makeState('building');
  }
}
function migrateState(input) {
  const base = makeState(input?.mission?.template || 'building');
  if (!input || typeof input !== 'object') return base;
  const merged = {...base,...input};
  merged.meta = {...base.meta,...input.meta,release:APP.version};
  merged.mission = {...base.mission,...input.mission};
  merged.gym = {...base.gym,...input.gym,spec:{...base.gym.spec,...input.gym?.spec},distribution:{...base.gym.distribution,...input.gym?.distribution}};
  merged.seize = {...base.seize,...input.seize};
  merged.formation = {...base.formation,...input.formation};
  merged.recursive = {...base.recursive,...input.recursive,settings:{...base.recursive.settings,...input.recursive?.settings}};
  merged.proof = {...base.proof,...input.proof};
  merged.authority = {...base.authority,...input.authority,admission:{...base.authority.admission,...input.authority?.admission,checks:{...base.authority.admission.checks,...input.authority?.admission?.checks}}};
  merged.requalification = {...base.requalification,...input.requalification};
  merged.jobs = Array.isArray(input.jobs) && input.jobs.length === 21 ? input.jobs : base.jobs;
  merged.candidates = Array.isArray(input.candidates) && input.candidates.length ? input.candidates : base.candidates;
  merged.chronicle = Array.isArray(input.chronicle) ? input.chronicle : [];
  merged.negativeCapability = Array.isArray(input.negativeCapability) ? input.negativeCapability : [];
  return merged;
}
function saveState() {
  state.meta.updatedAt = nowISO();
  try { localStorage.setItem(APP.storageKey, JSON.stringify(state)); }
  catch (error) { if (!saveState.warned) { console.warn('Local persistence is unavailable in this browser context; the current session remains usable.', error); saveState.warned = true; } }
  updateShell();
  document.dispatchEvent(new CustomEvent('goalos:state', {detail:{state:deepClone(state)}}));
}
function resetState(templateKey = 'building') {
  state = makeState(templateKey);
  saveState();
  renderAll();
}
function candidateById(id) { return state.candidates.find(candidate => candidate.id === id); }
function activeCandidates() { return state.candidates.filter(candidate => candidate.selected !== false); }
function template() { return TEMPLATES[state.mission.template] || TEMPLATES.custom; }
function record(type, title, detail = '', data = {}) {
  state.chronicle.unshift({id:uuid('rec'),type,title,detail,data,at:nowISO(),generation:state.meta.institutionGeneration});
  state.chronicle = state.chronicle.slice(0,300);
  saveState();
}
function addNegative(type, detail, candidateId = null) {
  state.negativeCapability.unshift({id:uuid('neg'),type,detail,candidateId,at:nowISO(),generation:state.meta.institutionGeneration});
  state.negativeCapability = state.negativeCapability.slice(0,150);
}

function toast(message) {
  const element = $('#toast');
  element.textContent = message;
  element.classList.remove('hidden');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.add('hidden'), 3300);
}
function modal(title, html) {
  $('#modalTitle').textContent = title;
  $('#modalBody').innerHTML = html;
  $('#modal').classList.remove('hidden');
  $('#modalClose').focus();
}
function closeModal() { $('#modal').classList.add('hidden'); }
function downloadBlob(name, blob) {
  const anchor = document.createElement('a');
  anchor.href = URL.createObjectURL(blob);
  anchor.download = name;
  document.body.appendChild(anchor);
  anchor.click();
  setTimeout(() => { URL.revokeObjectURL(anchor.href); anchor.remove(); }, 1000);
}
function downloadText(name, text, type = 'text/plain') { downloadBlob(name, new Blob([text], {type})); }

function generateCases(mode = 'formation', count = 180, seed = 1) {
  const rng = mulberry32(seed + fnv1a(`${state.mission.template}:${state.gym.version}:${mode}`));
  const profile = template().profile;
  const cases = [];
  for (let i = 0; i < count; i += 1) {
    const distributionRoll = rng() * 100;
    const distribution = distributionRoll < 60 ? 'normal' : distributionRoll < 85 ? 'edge' : 'adversarial';
    const modeLift = mode === 'proof' ? .06 : mode === 'transfer' ? .11 : 0;
    const difficulty = clamp((distribution === 'normal' ? 35 : distribution === 'edge' ? 61 : 78) + normalish(rng)*12 + modeLift*100, 8, 99) / 100;
    const criticalProbability = (.06 + profile.criticality*.13) * (distribution === 'adversarial' ? 1.7 : distribution === 'edge' ? 1.25 : .8);
    cases.push({
      id:`${mode.toUpperCase()}-${String(i+1).padStart(4,'0')}`,
      distribution,
      difficulty,
      critical:rng() < criticalProbability,
      ambiguity:clamp(20 + rng()*65 + (distribution==='adversarial'?15:0),0,100)/100,
      evidenceGap:clamp(10 + rng()*70 + (distribution==='adversarial'?18:0),0,100)/100,
      toolNeed:clamp(20 + rng()*70,0,100)/100,
      domainNeed:clamp(25 + rng()*70,0,100)/100,
      novelty:clamp((mode==='transfer'?55:15) + rng()*45 + (distribution==='adversarial'?15:0),0,100)/100,
      hazard:clamp(profile.criticality*45 + rng()*50 + (distribution==='adversarial'?15:0),0,100)/100,
      costPressure:clamp(20 + rng()*70,0,100)/100,
      speedPressure:clamp(20 + rng()*75,0,100)/100,
      deception:rng() < (distribution === 'adversarial' ? .38 : distribution === 'edge' ? .12 : .03),
      outage:rng() < (distribution === 'adversarial' ? .12 : .025),
      rightsTrap:rng() < (distribution === 'adversarial' ? .11 : .015),
      latent:normalish(rng),
      profile
    });
  }
  return cases;
}

function institutionalBurden(candidate) {
  const b = candidate.burden || {};
  const formation = Number(b.formation || 0) * state.recursive.formationCostMultiplier;
  const total = formation + Number(b.operation||0) + Number(b.migration||0) + Number(b.proof||0) + Number(b.dependency||0) + Number(b.irreversibility||0);
  return clamp(total / 3.6, 0, 100);
}
function candidateComplexity(candidate) {
  const p = candidate.params;
  return clamp((p.tools + p.exploration + p.memory + p.domain + p.reasoning) / 5,0,100);
}
function candidateScoreOnCase(candidate, item, mode, rng) {
  const p = candidate.params;
  const profile = item.profile;
  const memoryBonus = state.recursive.institutionalMemory * .55;
  const architectureBonus = candidate.kind === 'sovereign' ? 4.5 : candidate.kind === 'specialist' ? 2.5 : candidate.kind === 'executable' ? 2.2 : candidate.kind === 'beta' ? .5 : 0;
  const domainFit = (p.domain * item.domainNeed + p.reasoning * item.ambiguity + p.perception * (1-item.ambiguity)) / 100;
  const toolFit = p.tools * item.toolNeed / 100;
  const evidenceFit = p.evidence * (0.45 + item.evidenceGap*.55) / 100;
  const transferFit = (p.reasoning*.32 + p.domain*.22 + p.exploration*.22 + p.memory*.24) / 100;
  const physicalFit = (p.perception*.28 + p.tools*.25 + p.resilience*.25 + p.domain*.22) / 100;
  const strategicFit = (p.reasoning*.33 + p.evidence*.24 + p.domain*.23 + p.exploration*.20) / 100;
  const financialFit = (p.evidence*.29 + p.domain*.27 + p.reasoning*.24 + p.tools*.20) / 100;
  const softwareFit = (p.tools*.30 + p.reasoning*.25 + p.verifier*.23 + p.domain*.22) / 100;
  const missionFit = physicalFit*profile.physical + strategicFit*profile.strategic + financialFit*profile.financial + softwareFit*profile.software;
  const profileNorm = profile.physical + profile.strategic + profile.financial + profile.software;
  const normalizedMissionFit = missionFit / Math.max(.01, profileNorm);
  const baseCapability = .20*domainFit + .12*toolFit + .13*evidenceFit + .18*normalizedMissionFit + .12*(p.reasoning/100) + .09*(p.perception/100) + .08*(p.calibration/100) + .08*(p.resilience/100);
  const difficultyPenalty = item.difficulty * (.38 - p.reasoning/100*.12 - p.domain/100*.08);
  const noveltyPenalty = item.novelty * (.20 - transferFit*.13);
  const deceptionPenalty = item.deception ? Math.max(0,.16 - (p.evidence+p.verifier+p.calibration)/300*.12) : 0;
  const outagePenalty = item.outage ? Math.max(0,.18 - (p.resilience+p.portability)/200*.14) : 0;
  const stochastic = normalish(rng) * (mode === 'proof' ? .026 : .035);
  const raw = baseCapability - difficultyPenalty - noveltyPenalty - deceptionPenalty - outagePenalty + architectureBonus/100 + memoryBonus/100 + stochastic;
  const quality = clamp(raw * 112, 0, 100);
  const evidence = clamp(p.evidence*.66 + p.verifier*.20 + p.governance*.08 + p.tools*.06 - item.evidenceGap*18 + normalish(rng)*3.3,0,100);
  const reliability = clamp(p.resilience*.31 + p.calibration*.29 + p.verifier*.22 + p.governance*.18 - item.difficulty*12 - (item.outage?7:0) + normalish(rng)*2.5,0,100);
  const governance = clamp(p.governance*.59 + p.verifier*.17 + p.evidence*.11 + p.resilience*.13 - (item.rightsTrap?11:0) + normalish(rng)*1.7,0,100);
  const sovereignty = clamp(p.portability*.42 + p.memory*.22 + p.governance*.18 + p.evidence*.10 + p.resilience*.08 - (candidate.kind==='beta'?12:0),0,100);
  const transfer = clamp(transferFit*100 - item.novelty*13 + memoryBonus + normalish(rng)*2.8,0,100);
  const speed = clamp(p.speed*.74 + p.tools*.15 + p.reasoning*.11 - item.difficulty*13 - item.speedPressure*9 + normalish(rng)*2.2,0,100);
  const costScore = clamp(p.costEfficiency*.78 + p.speed*.08 + p.tools*.07 + p.humanEfficiency*.07 - institutionalBurden(candidate)*.16 - item.costPressure*7 + normalish(rng)*2.0,0,100);
  const humanScore = clamp(p.humanEfficiency*.72 + p.evidence*.08 + p.calibration*.08 + p.governance*.07 + p.tools*.05 - item.ambiguity*7 + normalish(rng)*2.0,0,100);
  const criticalRisk = item.critical ? (quality < 58 || reliability < 70 || evidence < 67) : false;
  const unauthorized = (item.rightsTrap || item.hazard > .82) && governance < 75 && candidate.kind !== 'incumbent';
  const abstained = quality < 45 && p.calibration > 65;
  const accepted = !criticalRisk && !unauthorized && (quality >= 51 || abstained);
  const value = clamp(quality*.34 + reliability*.15 + evidence*.13 + transfer*.06 + speed*.11 + costScore*.11 + humanScore*.10 - (criticalRisk?45:0) - (unauthorized?55:0),0,100);
  const utility = clamp(quality*.22 + value*.18 + evidence*.12 + reliability*.13 + sovereignty*.08 + governance*.10 + transfer*.07 + speed*.04 + costScore*.03 + humanScore*.03 - institutionalBurden(candidate)*.035 - (criticalRisk?32:0) - (unauthorized?45:0),-100,100);
  return {quality,value,speed,evidence,reliability,sovereignty,governance,transfer,costScore,humanScore,utility,criticalError:criticalRisk?1:0,unauthorized:unauthorized?1:0,abstained,accepted};
}

function evaluateCandidate(candidate, cases, mode = 'formation', seed = 1) {
  const rng = mulberry32(seed + fnv1a(`${candidate.id}:${candidate.generation}:${mode}:${state.meta.institutionGeneration}`));
  const episodes = cases.map(item => ({caseId:item.id,distribution:item.distribution,critical:item.critical,...candidateScoreOnCase(candidate,item,mode,rng)}));
  const metrics = {};
  Object.keys(METRIC_LABELS).forEach(key => { metrics[key] = mean(episodes.map(e => e[key])); });
  const criticalErrors = sum(episodes.map(e => e.criticalError));
  const unauthorizedActions = sum(episodes.map(e => e.unauthorized));
  const acceptedRate = mean(episodes.map(e => e.accepted ? 100 : 0));
  const abstentionRate = mean(episodes.map(e => e.abstained ? 100 : 0));
  const tail = [...episodes].sort((a,b) => a.utility-b.utility).slice(0,Math.max(1,Math.floor(episodes.length*.05)));
  const tailUtility = mean(tail.map(e => e.utility));
  const burden = institutionalBurden(candidate);
  return {candidateId:candidate.id,candidateName:candidate.name,generation:candidate.generation,metrics,criticalErrors,unauthorizedActions,acceptedRate,abstentionRate,tailUtility,burden,episodes,manifest:hashObject({candidate,env:state.gym.version,mode})};
}

function runMatchedFormation() {
  const cases = generateCases('formation', state.gym.formationCases, state.gym.formationSeed);
  const results = activeCandidates().map(candidate => evaluateCandidate(candidate,cases,'formation',state.gym.formationSeed));
  results.sort((a,b) => b.metrics.utility-a.metrics.utility);
  state.formation = {runId:uuid('formation'),results,caseCount:cases.length,seed:state.gym.formationSeed,championId:results[0]?.candidateId || null,archive:qualityDiversityArchive(results),completedAt:nowISO()};
  state.recursive.archive = deepClone(state.formation.archive);
  state.recursive.history.push({generation:state.recursive.generation,at:nowISO(),championId:state.formation.championId,utility:results[0]?.metrics.utility || 0,burden:results[0]?.burden || 0,criticalErrors:results[0]?.criticalErrors || 0});
  logRSI(`Formation arena completed on ${cases.length} matched cases. Champion: ${results[0]?.candidateName}.`, 'good');
  record('FORMATION_ARENA_COMPLETED','Formation Gym completed',`${cases.length} matched cases; champion ${results[0]?.candidateName || 'none'}.`);
  saveState();
  renderAll();
  toast('Formation arena completed');
}

function qualityDiversityArchive(results) {
  const niches = [
    ['Champion',r => r.metrics.utility],
    ['Safest',r => r.metrics.reliability + r.metrics.governance - r.criticalErrors*20 - r.unauthorizedActions*25],
    ['Lowest burden',r => 100-r.burden],
    ['Strongest transfer',r => r.metrics.transfer],
    ['Most sovereign',r => r.metrics.sovereignty],
    ['Fastest',r => r.metrics.speed],
    ['Best evidence',r => r.metrics.evidence]
  ];
  const seen = new Set();
  return niches.map(([niche,fn]) => {
    const winner = [...results].sort((a,b) => fn(b)-fn(a))[0];
    if (!winner) return null;
    const key = `${niche}:${winner.candidateId}`;
    if (seen.has(key)) return null;
    seen.add(key);
    return {niche,candidateId:winner.candidateId,candidateName:winner.candidateName,score:fn(winner),utility:winner.metrics.utility,burden:winner.burden};
  }).filter(Boolean);
}

function bestReferenceResult(excludeId = null) {
  const eligible = state.formation.results.filter(result => result.candidateId !== excludeId && ['incumbent','general_ai','specialist_cell','executable'].includes(result.candidateId));
  return [...eligible].sort((a,b) => b.metrics.utility-a.metrics.utility)[0] || state.formation.results.find(r => r.candidateId === 'incumbent');
}

function proofThresholds() {
  return {lcb:1.0,meanGain:3.0,evidence:82,reliability:80,governance:84,transfer:74,sovereignty:72,criticalErrors:0,unauthorizedActions:0,proofCoverage:84};
}

function freezeCandidate(candidateId) {
  if (!state.formation.results.length) runMatchedFormation();
  const candidate = candidateById(candidateId || state.formation.championId);
  if (!candidate) { toast('Select a candidate first'); return; }
  const manifest = {projectId:state.meta.projectId,mission:state.mission,environmentVersion:state.gym.version,candidate,proofThresholds:proofThresholds(),protectedSeedCommitment:hashObject({proofSeed:state.gym.proofSeed,transferSeed:state.gym.transferSeed,version:state.gym.version})};
  state.proof.frozenCandidateId = candidate.id;
  state.proof.frozenAt = nowISO();
  state.proof.manifestHash = hashObject(manifest);
  state.proof.result = null;
  state.authority.admission.status = 'not_admitted';
  state.authority.admission.signedAt = null;
  Object.keys(state.authority.admission.checks).forEach(key => state.authority.admission.checks[key] = false);
  record('CHALLENGER_FROZEN','One challenger frozen',`${candidate.name} · ${state.proof.manifestHash}`);
  saveState(); renderAll(); toast(`${candidate.name} frozen for fresh proof`);
}

function runFreshProof() {
  const candidate = candidateById(state.proof.frozenCandidateId);
  if (!candidate) { toast('Freeze one challenger first'); return; }
  const proofCases = generateCases('proof',state.gym.proofCases,state.gym.proofSeed);
  const transferCases = generateCases('transfer',state.gym.transferCases,state.gym.transferSeed);
  const candidateProof = evaluateCandidate(candidate,proofCases,'proof',state.gym.proofSeed);
  const candidateTransfer = evaluateCandidate(candidate,transferCases,'transfer',state.gym.transferSeed);
  const references = activeCandidates().filter(c => c.id !== candidate.id && c.generation === 0).map(c => evaluateCandidate(c,proofCases,'proof',state.gym.proofSeed));
  const reference = [...references].sort((a,b) => b.metrics.utility-a.metrics.utility)[0];
  const diffs = candidateProof.episodes.map((episode,index) => episode.utility - (reference?.episodes[index]?.utility || 0));
  const pairedMean = mean(diffs);
  const se = stdev(diffs) / Math.sqrt(Math.max(1,diffs.length));
  const lcb = pairedMean - 1.645*se;
  const proofCoverage = clamp((candidateProof.metrics.evidence*.48 + candidateProof.metrics.governance*.18 + candidateProof.metrics.reliability*.19 + candidateProof.metrics.sovereignty*.15),0,100);
  const thresholds = proofThresholds();
  const dimensions = {
    capability:candidateProof.metrics.quality,
    economics:(candidateProof.metrics.value + candidateProof.metrics.costScore)/2,
    reliability:candidateProof.metrics.reliability,
    sovereignty:candidateProof.metrics.sovereignty,
    governance:candidateProof.metrics.governance,
    transfer:candidateTransfer.metrics.transfer
  };
  const gates = [
    {name:'Matched superiority lower bound',pass:lcb>thresholds.lcb,detail:`LCB ${round(lcb,2)} > ${thresholds.lcb}`},
    {name:'Mean mission advantage',pass:pairedMean>thresholds.meanGain,detail:`Mean gain ${round(pairedMean,2)} > ${thresholds.meanGain}`},
    {name:'Critical-error gate',pass:candidateProof.criticalErrors<=thresholds.criticalErrors,detail:`${candidateProof.criticalErrors} critical errors`},
    {name:'Unauthorized-action gate',pass:candidateProof.unauthorizedActions<=thresholds.unauthorizedActions,detail:`${candidateProof.unauthorizedActions} unauthorized actions`},
    {name:'Evidence coverage',pass:candidateProof.metrics.evidence>=thresholds.evidence,detail:`${round(candidateProof.metrics.evidence)}%`},
    {name:'Reliability',pass:candidateProof.metrics.reliability>=thresholds.reliability,detail:`${round(candidateProof.metrics.reliability)}%`},
    {name:'Governance integrity',pass:candidateProof.metrics.governance>=thresholds.governance,detail:`${round(candidateProof.metrics.governance)}%`},
    {name:'Transfer',pass:candidateTransfer.metrics.transfer>=thresholds.transfer,detail:`${round(candidateTransfer.metrics.transfer)}%`},
    {name:'Sovereignty',pass:candidateProof.metrics.sovereignty>=thresholds.sovereignty,detail:`${round(candidateProof.metrics.sovereignty)}%`},
    {name:'Proof coverage',pass:proofCoverage>=thresholds.proofCoverage,detail:`${round(proofCoverage)}%`},
    {name:'Rollback and degraded mode',pass:candidate.params.resilience>=80 && candidate.params.governance>=80,detail:`Resilience ${round(candidate.params.resilience)}; governance ${round(candidate.params.governance)}`}
  ];
  const pass = gates.every(gate => gate.pass);
  const result = {
    id:uuid('proof'),candidateId:candidate.id,candidateName:candidate.name,manifestHash:state.proof.manifestHash,environmentVersion:state.gym.version,runAt:nowISO(),proofCases:proofCases.length,transferCases:transferCases.length,referenceId:reference?.candidateId,referenceName:reference?.candidateName,paired:{mean:pairedMean,se,lcb},dimensions,proofCoverage,criticalErrors:candidateProof.criticalErrors,unauthorizedActions:candidateProof.unauthorizedActions,gates,pass,status:pass?'SPECIALIST ASI SIMULATION GATE — PASS':'FRESH-PROOF GATE — FAIL',claimBoundary:'This is a deterministic local simulation result. It does not establish real-world Specialist ASI, Mission Alpha, customer value or institutional authority.'
  };
  state.proof.result = result;
  state.proof.history.unshift(deepClone(result));
  if (!pass) {
    gates.filter(g => !g.pass).forEach(g => addNegative('Fresh-proof failure',`${candidate.name}: ${g.name} — ${g.detail}`,candidate.id));
    logRSI(`Fresh proof rejected ${candidate.name}. ${gates.filter(g=>!g.pass).length} gate(s) failed.`, 'bad');
  } else {
    logRSI(`Fresh proof passed for ${candidate.name}. LCB ${round(lcb,2)}; proof coverage ${round(proofCoverage)}%.`, 'good');
  }
  record(pass?'FRESH_PROOF_PASSED':'FRESH_PROOF_FAILED',result.status,`${candidate.name}; LCB ${round(lcb,2)}; proof coverage ${round(proofCoverage)}%.`);
  saveState(); renderAll(); toast(pass?'Fresh-proof simulation passed':'Fresh-proof gate failed');
}

function logRSI(message, tone = 'dim') {
  state.recursive.console.unshift({at:nowISO(),message,tone,generation:state.recursive.generation});
  state.recursive.console = state.recursive.console.slice(0,160);
}

function mutationDistance(a,b) {
  const keys = Object.keys(a.params);
  return Math.sqrt(mean(keys.map(key => (Number(a.params[key])-Number(b.params[key]))**2)));
}
function mutateCandidate(parent, generation, index, gradientKeys = []) {
  const rng = mulberry32(state.gym.formationSeed + generation*10007 + index*997 + fnv1a(parent.id));
  const child = deepClone(parent);
  child.id = `g${generation}_${safeName(parent.kind)}_${Math.floor(rng()*999999).toString().padStart(6,'0')}`;
  child.name = `${parent.name.replace(/ · G\d+$/,'')} · G${generation}`;
  child.parentId = parent.id;
  child.generation = generation;
  child.kind = parent.kind === 'incumbent' ? 'specialist' : parent.kind;
  child.createdAt = nowISO();
  child.selected = true;
  const keys = Object.keys(child.params);
  const targeted = gradientKeys.length ? gradientKeys : shuffle(rng,keys).slice(0,3);
  const mutationCount = 2 + Math.floor(rng()*4);
  const chosen = [...new Set([...targeted.slice(0,2),...shuffle(rng,keys).slice(0,mutationCount)])];
  chosen.forEach((key,position) => {
    const direction = position < 2 ? 1 : (rng()>.24?1:-1);
    const scale = state.recursive.settings.mutationScale * (0.55+rng()*.9);
    child.params[key] = clamp(child.params[key] + direction*scale + state.recursive.institutionalMemory*.25);
  });
  const tradeoffKey = pick(rng,['formation','operation','proof','dependency','migration']);
  child.burden[tradeoffKey] = clamp(child.burden[tradeoffKey] + normalish(rng)*8 + (chosen.length*1.2),0,100);
  if (chosen.includes('portability')) child.burden.dependency = clamp(child.burden.dependency - 3 - rng()*6,0,100);
  if (chosen.includes('resilience')) child.burden.proof = clamp(child.burden.proof + 2 + rng()*5,0,100);
  if (chosen.includes('evidence') || chosen.includes('verifier')) child.burden.operation = clamp(child.burden.operation + 1 + rng()*4,0,100);
  child.notes = `Bounded descendant of ${parent.name}; mutated ${chosen.map(k=>PARAM_LABELS[k]).join(', ')}. Formation only; no inherited proof or authority.`;
  child.mutation = {keys:chosen,parentId:parent.id,generation};
  return child;
}

function finiteDifferenceGradient(candidateId = state.selectedCandidateId) {
  const candidate = candidateById(candidateId);
  if (!candidate) return [];
  const cases = generateCases('formation',Math.min(120,state.gym.formationCases),state.gym.formationSeed);
  const base = evaluateCandidate(candidate,cases,'formation',state.gym.formationSeed);
  const results = Object.keys(candidate.params).map(key => {
    const clone = deepClone(candidate);
    clone.id = `${candidate.id}_gradient_${key}`;
    clone.params[key] = clamp(clone.params[key] + 5);
    const evaluated = evaluateCandidate(clone,cases,'formation',state.gym.formationSeed);
    const utilityGain = evaluated.metrics.utility - base.metrics.utility;
    const burdenDelta = evaluated.burden - base.burden;
    const proofCapitalEfficiency = utilityGain / Math.max(.75,1+Math.max(0,burdenDelta));
    return {key,label:PARAM_LABELS[key],utilityGain,burdenDelta,proofCapitalEfficiency,base:candidate.params[key],next:clone.params[key]};
  }).sort((a,b) => b.proofCapitalEfficiency-a.proofCapitalEfficiency);
  state.gradient = {candidateId:candidate.id,results,computedAt:nowISO()};
  saveState();
  return results;
}

function createGradientChallenger() {
  const candidate = candidateById(state.gradient.candidateId || state.selectedCandidateId);
  if (!candidate) { toast('Select a candidate first'); return; }
  const gradient = state.gradient.results.length ? state.gradient.results : finiteDifferenceGradient(candidate.id);
  const child = deepClone(candidate);
  child.id = uuid('gradient');
  child.name = `${candidate.name.replace(/ · G\d+$/,'')} · Gradient Challenger`;
  child.parentId = candidate.id;
  child.generation = state.recursive.generation + 1;
  child.createdAt = nowISO();
  child.kind = candidate.kind === 'incumbent' ? 'specialist' : candidate.kind;
  const keys = gradient.slice(0,3).map(item => item.key);
  keys.forEach((key,index) => { child.params[key] = clamp(child.params[key] + 5 - index); });
  child.burden.formation = clamp(child.burden.formation + 6,0,100);
  child.burden.proof = clamp(child.burden.proof + 4,0,100);
  child.notes = `Bounded Mission Advantage Gradient move from ${candidate.name}: ${keys.map(k=>PARAM_LABELS[k]).join(', ')}.`;
  child.mutation = {keys,parentId:candidate.id,generation:child.generation,source:'gradient'};
  state.candidates.push(child);
  state.recursive.lineage.push({parentId:candidate.id,childId:child.id,generation:child.generation,source:'gradient',at:nowISO()});
  state.selectedCandidateId = child.id;
  state.formation.results = [];
  state.proof.result = null;
  logRSI(`Gradient challenger created from ${candidate.name}: ${keys.join(', ')}.`, 'good');
  record('GRADIENT_CHALLENGER_CREATED','Mission Advantage Gradient challenger created',child.name);
  saveState(); renderAll(); toast('Gradient challenger created');
}

function archiveParentPool() {
  const ids = [...new Set([
    state.formation.championId,
    ...state.recursive.archive.map(item => item.candidateId),
    state.proof.result?.pass ? state.proof.result.candidateId : null,
    state.recursive.admittedChampionId || null
  ].filter(Boolean))];
  const pool = ids.map(candidateById).filter(Boolean);
  return pool.length ? pool : activeCandidates();
}

async function runRecursiveFoundry(generations = 1) {
  if (!state.formation.results.length) runMatchedFormation();
  const startGeneration = state.recursive.generation;
  const buttonIds = ['evolve1','evolve5','evolve20'];
  buttonIds.forEach(id => { const b = $(`#${id}`); if (b) b.disabled = true; });
  for (let step = 0; step < generations; step += 1) {
    const generation = state.recursive.generation + 1;
    const pool = archiveParentPool();
    const parent = pool[step % pool.length] || candidateById(state.formation.championId);
    const gradient = finiteDifferenceGradient(parent.id).slice(0,5).map(item => item.key);
    const children = [];
    for (let i = 0; i < state.recursive.settings.childrenPerGeneration; i += 1) {
      const chosenParent = pool[i % pool.length] || parent;
      const child = mutateCandidate(chosenParent,generation,i,gradient);
      children.push(child);
      state.candidates.push(child);
      state.recursive.lineage.push({parentId:chosenParent.id,childId:child.id,generation,source:'bounded_mutation',at:nowISO(),keys:child.mutation.keys});
    }
    const cases = generateCases('formation',state.gym.formationCases,state.gym.formationSeed + generation*17);
    const candidateSet = [...pool,...children];
    const results = candidateSet.map(candidate => evaluateCandidate(candidate,cases,'formation',state.gym.formationSeed + generation*17));
    results.sort((a,b) => b.metrics.utility-a.metrics.utility);
    const previousChampionResult = state.formation.results.find(result => result.candidateId === state.formation.championId) || state.formation.results[0];
    const champion = results[0];
    const championCandidate = candidateById(champion.candidateId);
    const novelty = previousChampionResult && championCandidate ? mutationDistance(championCandidate,candidateById(previousChampionResult.candidateId) || championCandidate) : 0;
    const governanceRegression = previousChampionResult ? champion.metrics.governance + 1 < previousChampionResult.metrics.governance : false;
    const criticalRegression = previousChampionResult ? champion.criticalErrors > previousChampionResult.criticalErrors : false;
    const utilityGain = previousChampionResult ? champion.metrics.utility - previousChampionResult.metrics.utility : 0;
    const accepted = utilityGain > .15 && !governanceRegression && !criticalRegression;
    if (!accepted && championCandidate?.generation === generation) {
      addNegative('Recursive candidate rejected',`${championCandidate.name}: gain ${round(utilityGain,2)}, governance regression ${governanceRegression}, critical regression ${criticalRegression}.`,championCandidate.id);
    }
    const rejectedDescendant = results.find(result => children.some(child => child.id === result.candidateId) && (result.criticalErrors > 0 || result.unauthorizedActions > 0 || (previousChampionResult && result.metrics.governance + 2 < previousChampionResult.metrics.governance)));
    if (rejectedDescendant && !state.negativeCapability.some(item => item.generation === state.meta.institutionGeneration && item.candidateId === rejectedDescendant.candidateId)) {
      addNegative('Unsafe lineage retained',`${rejectedDescendant.candidateName}: critical ${rejectedDescendant.criticalErrors}, unauthorized ${rejectedDescendant.unauthorizedActions}, governance ${round(rejectedDescendant.metrics.governance)}.`,rejectedDescendant.candidateId);
    }
    const selectedResults = accepted ? results : [previousChampionResult,...results.filter(r=>r.candidateId!==previousChampionResult?.candidateId)].filter(Boolean);
    selectedResults.sort((a,b) => b.metrics.utility-a.metrics.utility);
    state.formation = {runId:uuid('formation'),results:selectedResults,caseCount:cases.length,seed:state.gym.formationSeed + generation*17,championId:selectedResults[0]?.candidateId || state.formation.championId,archive:qualityDiversityArchive(results),completedAt:nowISO()};
    state.recursive.generation = generation;
    state.recursive.archive = state.formation.archive;
    state.recursive.history.push({generation,at:nowISO(),championId:state.formation.championId,utility:selectedResults[0]?.metrics.utility || 0,burden:selectedResults[0]?.burden || 0,criticalErrors:selectedResults[0]?.criticalErrors || 0,utilityGain,novelty,accepted});
    logRSI(`G${generation}: ${accepted?'accepted':'rejected'} formation champion ${selectedResults[0]?.candidateName}. Δ utility ${round(utilityGain,2)}; novelty ${round(novelty,1)}.`,accepted?'good':'warn');
    state.selectedCandidateId = state.formation.championId;
    state.proof.result = null;
    if (step % 4 === 3) await sleep(20);
  }
  state.recursive.lastRunAt = nowISO();
  record('RECURSIVE_FOUNDRY_RUN','Recursive Foundry completed',`${generations} generation(s), G${startGeneration} → G${state.recursive.generation}.`);
  saveState(); renderAll();
  buttonIds.forEach(id => { const b = $(`#${id}`); if (b) b.disabled = false; });
  toast(`${generations} bounded recursive generation(s) completed`);
}

function applyChronicleLearning() {
  if (!state.proof.result?.pass || state.authority.admission.status !== 'simulated_admitted') {
    toast('Pass fresh proof and complete simulated admission first'); return;
  }
  const champion = candidateById(state.proof.result.candidateId);
  if (!champion) return;
  state.meta.institutionGeneration += 1;
  state.recursive.institutionalMemory = clamp(state.recursive.institutionalMemory + 4,0,30);
  state.recursive.formationCostMultiplier = Math.max(.55,state.recursive.formationCostMultiplier*.92);
  state.recursive.admittedChampionId = champion.id;
  const failureCount = state.negativeCapability.filter(item => item.candidateId === champion.id || !item.candidateId).length;
  logRSI(`Institutional generation IG${state.meta.institutionGeneration}: admitted evidence improves formation memory; cost multiplier ${round(state.recursive.formationCostMultiplier,2)}; ${failureCount} failure records retained.`, 'good');
  record('CAPITAL_TO_CAPACITY','Chronicle-admitted capability converted to formation capacity',`Institution generation ${state.meta.institutionGeneration}; formation cost multiplier ${round(state.recursive.formationCostMultiplier,2)}.`);
  state.proof.result = null;
  state.proof.frozenCandidateId = null;
  state.proof.manifestHash = null;
  state.authority.admission.status = 'not_admitted';
  saveState(); renderAll(); toast('Chronicle learning seeded the next successor generation');
}

function calculateSEIZE() {
  const s = state.seize;
  const gymScore = mean(state.mission.gymmability);
  const champion = state.formation.results.find(r => r.candidateId === state.formation.championId) || state.formation.results[0];
  const inferredProbability = champion ? clamp(25 + champion.metrics.utility*.65 - champion.criticalErrors*8,5,95) : s.probability;
  s.probability = Math.round((Number(s.probability)+inferredProbability)/2);
  const durationPenalty = s.missionValue * Math.max(0,(s.durationMonths/Math.max(1,s.halfLifeMonths))-.35)*.20;
  const expectedValue = s.probability/100*s.missionValue + s.optionValue - s.proofCost - s.formationCost - s.tailRisk - durationPenalty;
  const durationExposure = s.durationMonths / Math.max(1,s.halfLifeMonths);
  let decision = 'Run Experiment Zero';
  if (gymScore < 45) decision = 'Hold or narrow the mission';
  else if (durationExposure >= 1) decision = 'Rent, partner or preserve a reserve option';
  else if (expectedValue > s.proofCost*1.7) decision = 'Proceed to bounded proof';
  else if (expectedValue > 0) decision = 'Run the smallest decisive experiment';
  else if (s.tailRisk > s.missionValue*.45) decision = 'Stop or redesign the authority architecture';
  else decision = 'Repair, rent or defer';
  const gradient = state.gradient.results.length ? state.gradient.results : finiteDifferenceGradient(state.selectedCandidateId);
  const next = gradient[0];
  s.score = {expectedValue,durationExposure,gymScore,inferredProbability};
  s.decision = decision;
  s.nextEvidence = next ? `Test whether increasing ${next.label.toLowerCase()} by one bounded intervention creates positive matched gain without governance regression.` : 'Reproduce the incumbent and strongest alternative under identical conditions.';
  saveState(); renderAll(); toast('SEIZE underwriting updated');
}

function freezeSEIZE() {
  calculateSEIZE();
  state.seize.frozen = true;
  state.seize.frozenAt = nowISO();
  state.mission.constitutionFrozen = true;
  state.mission.frozenAt = nowISO();
  state.jobs[0].status = 'passed'; state.jobs[0].owner = state.authority.owner; state.jobs[0].evidence = 'Mission and succession constitution frozen.'; state.jobs[0].updatedAt = nowISO();
  record('SEIZE_FROZEN','SEIZE Succession Constitution frozen',`${state.seize.decision}; next evidence: ${state.seize.nextEvidence}`);
  saveState(); renderAll(); toast('Succession Constitution frozen');
}

function compileJobs() {
  state.jobs = defaultJobs();
  const owner = state.authority.owner || 'Accountable human principal';
  state.jobs.forEach((job,index) => {
    job.owner = index < 2 ? owner : index < 5 ? 'SEIZE Underwriter' : index < 16 ? 'Successor Foundry' : index < 18 ? 'Independent Proof Plane' : index < 20 ? owner : 'Capital-to-Capacity Committee';
    if (index === 0 && state.mission.constitutionFrozen) { job.status='passed'; job.evidence='Frozen Mission Constitution.'; job.updatedAt=nowISO(); }
    if (index === 1 && state.formation.results.length) { job.status='passed'; job.evidence='Matched incumbent baseline recorded.'; job.updatedAt=nowISO(); }
    if (index === 2 && state.candidates.length >= 3) { job.status='passed'; job.evidence=`${state.candidates.length} candidate architectures in Successor Manifold.`; job.updatedAt=nowISO(); }
  });
  record('JOBS_COMPILED','Bounded AGI Job portfolio compiled','21 canonical work-and-proof contracts created.');
  saveState(); renderAll(); toast('21 Bounded AGI Jobs compiled');
}

function runPreparationJobs() {
  if (!state.jobs.length) compileJobs();
  state.jobs.forEach((job,index) => {
    if (index <= 15) {
      job.status = 'passed';
      job.updatedAt = nowISO();
      job.evidence = index === 0 ? 'Mission Constitution frozen in demo.' : index === 7 ? `${state.candidates.length} complete candidate architectures.` : index === 15 ? 'Rollback, fail-closed and degraded-mode requirements specified.' : 'Demonstration artifact generated locally.';
    } else if (index === 16) job.status = state.proof.result ? (state.proof.result.pass?'passed':'blocked') : 'ready';
    else if (index === 17) job.status = state.proof.result?.pass ? 'ready' : 'blocked';
  });
  record('PREPARATION_JOBS_RUN','Safe preparation jobs completed','Jobs 1–16 completed as local demonstration artifacts; proof and admission remain separate.');
  saveState(); renderAll(); toast('Preparation jobs completed; protected proof remains separate');
}

function updateJobsFromProofAndAuthority() {
  if (!state.jobs.length) return;
  if (state.proof.result) {
    state.jobs[16].status = state.proof.result.pass ? 'passed' : 'blocked';
    state.jobs[16].evidence = state.proof.result.status;
    state.jobs[17].status = state.proof.result.pass ? 'passed' : 'blocked';
    state.jobs[17].evidence = state.proof.result.pass ? 'Independent local simulation scorecard passed.' : 'Fresh-proof gates failed.';
  }
  if (state.authority.admission.status === 'simulated_admitted') {
    state.jobs[18].status='passed'; state.jobs[18].evidence='Simulated accountable acceptance recorded.';
    state.jobs[19].status='passed'; state.jobs[19].evidence='Simulated Authority Envelope admission recorded.';
  }
  if (state.meta.institutionGeneration > 0) {
    state.jobs[20].status='passed'; state.jobs[20].evidence=`Chronicle learning seeded institutional generation ${state.meta.institutionGeneration}.`;
  }
}


/* ---------- Authority, admission and requalification ---------- */
function authorityExpected(event) {
  return {
    evidence_expired:'Stop, fail closed and request fresh proof',
    budget_exhausted:'Stop and escalate to the accountable principal',
    provider_outage:'Enter degraded mode or incumbent fallback',
    critical_event:'Escalate; no autonomous consequential action',
    unapproved_action:'Refuse, log and preserve the attempted action',
    distribution_shift:'Fail closed, narrow scope and requalify'
  }[event];
}

function authorityStressControls(candidate, level) {
  const p = candidate?.params || {};
  const control = mean([p.governance||0,p.resilience||0,p.verifier||0,p.evidence||0,p.calibration||0]);
  const levelPenalty = {A0:0,A1:1,A2:3,A3:7,A4:13}[level] || 0;
  return clamp(control - levelPenalty,0,100);
}

function runAuthorityStress() {
  const candidate = candidateById(state.proof.frozenCandidateId) || candidateById(state.formation.championId) || candidateById(state.selectedCandidateId);
  if (!candidate) { toast('Select a successor candidate first'); return; }
  const events = ['evidence_expired','budget_exhausted','provider_outage','critical_event','unapproved_action','distribution_shift'];
  const rng = mulberry32(fnv1a(`${candidate.id}:${state.authority.level}:${state.gym.version}:${state.proof.manifestHash || ''}`));
  const control = authorityStressControls(candidate,state.authority.level);
  const tests = events.map((event,index) => {
    const threshold = 71 + index*1.5 + (state.authority.level==='A3'?4:state.authority.level==='A4'?9:0);
    const pass = control + normalish(rng)*2.6 >= threshold;
    const expected = authorityExpected(event);
    const unsafe = ['Continue under inherited evidence','Retry silently without escalation','Proceed and log later','Ignore the authority mismatch'];
    return {event,name:event.replaceAll('_',' '),expected,actual:pass?expected:pick(rng,unsafe),pass,control,threshold};
  });
  state.authority.stress = {id:uuid('stress'),candidateId:candidate.id,level:state.authority.level,runAt:nowISO(),tests,pass:tests.every(item=>item.pass)};
  state.authority.admission.checks.rollback = tests.find(item=>item.event==='provider_outage')?.pass && tests.find(item=>item.event==='distribution_shift')?.pass;
  record('AUTHORITY_STRESS_COMPLETED','Authority stress suite completed',`${tests.filter(item=>item.pass).length}/${tests.length} controls behaved as expected.`,state.authority.stress);
  updateJobsFromProofAndAuthority();
  saveState(); renderAll(); toast('Authority stress suite completed');
}

function maximumEligibleAuthority() {
  const ceiling = authorityRank(state.mission.authorityCeiling);
  if (!state.proof.result?.pass) return 'A1';
  if (!state.authority.stress?.pass) return ['A0','A1','A2','A3','A4'][Math.min(2,ceiling)];
  const candidate = candidateById(state.proof.frozenCandidateId);
  const strong = candidate && candidate.params.governance >= 88 && candidate.params.resilience >= 86 && candidate.params.verifier >= 84;
  const simulationCeiling = strong ? 3 : 2; // A4 is never auto-enabled in a browser demo.
  return ['A0','A1','A2','A3','A4'][Math.min(ceiling,simulationCeiling)];
}

function admissionEligible() {
  const checks = state.authority.admission.checks;
  return Boolean(
    state.proof.result?.pass &&
    state.authority.stress?.pass &&
    checks.identity && checks.proof && checks.rights && checks.rollback && checks.owner && checks.expiry &&
    state.authority.admission.signedBy.trim()
  );
}

function issueSimulatedAdmission() {
  if (!admissionEligible()) { toast('Complete fresh proof, stress tests, every admission check and the signer field'); return; }
  const candidate = candidateById(state.proof.frozenCandidateId);
  if (!candidate) return;
  const max = maximumEligibleAuthority();
  if (authorityRank(state.authority.level) > authorityRank(max)) state.authority.level = max;
  state.authority.admission.status = 'simulated_admitted';
  state.authority.admission.signedAt = nowISO();
  state.authority.admission.recordId = uuid('ADM');
  state.authority.admission.candidateId = candidate.id;
  state.authority.admission.manifestHash = state.proof.manifestHash;
  state.authority.admission.level = state.authority.level;
  state.authority.admission.expiry = state.authority.expiry;
  state.authority.admission.statement = 'Demonstration-only accountable admission record. It creates no legal, organizational, professional, financial, physical or operational authority.';
  record('SIMULATED_MISSION_SOVEREIGN_ADMISSION','Simulated accountable admission recorded',`${candidate.name} admitted at ${state.authority.level} until ${state.authority.expiry}.`,deepClone(state.authority.admission));
  updateJobsFromProofAndAuthority();
  saveState(); renderAll(); toast('Simulated admission recorded');
}

function revokeAdmission(reason = 'Manual demonstration revocation') {
  const previous = deepClone(state.authority.admission);
  state.authority.admission.status = 'revoked';
  state.authority.level = 'A0';
  state.authority.admission.revokedAt = nowISO();
  state.authority.admission.revocationReason = reason;
  record('SUCCESSOR_REVOKED','Successor admission revoked',reason,{previous});
  addNegative('Revocation',reason,state.proof.frozenCandidateId);
  updateJobsFromProofAndAuthority();
  saveState(); renderAll(); toast('Admission revoked and authority reduced to A0');
}

function saveAuthorityFromDOM() {
  const owner = $('#authorityOwner');
  const expiry = $('#authorityExpiry');
  const permitted = $('#authorityPermitted');
  const prohibited = $('#authorityProhibited');
  if (owner) state.authority.owner = owner.value.trim() || 'Accountable human principal';
  if (expiry) state.authority.expiry = expiry.value || addDays(nowISO(),90);
  if (permitted) state.authority.permitted = list(permitted.value);
  if (prohibited) state.authority.prohibited = list(prohibited.value);
  state.authority.admission.checks.owner = Boolean(state.authority.owner);
  state.authority.admission.checks.expiry = Boolean(state.authority.expiry);
  saveState(); renderAll(); toast('Authority Envelope saved');
}

function materialChangePenalty(candidate, changes, severity) {
  const next = deepClone(candidate);
  next.id = uuid('release');
  next.parentId = candidate.id;
  next.name = `${candidate.name.replace(/ · Release .*/, '')} · Release ${state.proof.history.length + 2}`;
  next.generation = candidate.generation + 1;
  next.createdAt = nowISO();
  next.releaseChanges = [...changes];
  const delta = {low:3.5,medium:8,high:15}[severity] || 8;
  let extraDifficulty = 0;
  for (const change of changes) {
    if (change === 'model') { next.params.reasoning=clamp(next.params.reasoning+delta*.25); next.params.calibration=clamp(next.params.calibration-delta*.35); next.params.evidence=clamp(next.params.evidence-delta*.18); extraDifficulty+=.025; }
    if (change === 'data') { next.params.domain=clamp(next.params.domain-delta*.55); next.params.evidence=clamp(next.params.evidence-delta*.35); extraDifficulty+=.06; }
    if (change === 'provider') { next.params.resilience=clamp(next.params.resilience-delta*.55); next.params.portability=clamp(next.params.portability-delta*.4); next.burden.dependency=clamp(next.burden.dependency+delta); extraDifficulty+=.05; }
    if (change === 'verifier') { next.params.verifier=clamp(next.params.verifier-delta*.6); next.params.governance=clamp(next.params.governance-delta*.35); extraDifficulty+=.055; }
    if (change === 'workflow') { next.params.tools=clamp(next.params.tools-delta*.35); next.params.humanEfficiency=clamp(next.params.humanEfficiency-delta*.3); next.burden.migration=clamp(next.burden.migration+delta*.6); extraDifficulty+=.04; }
    if (change === 'authority') { next.params.governance=clamp(next.params.governance-delta*.25); next.burden.proof=clamp(next.burden.proof+delta*.8); extraDifficulty+=.08; }
    if (change === 'environment') { next.params.transfer=next.params.transfer; next.params.domain=clamp(next.params.domain-delta*.4); next.params.resilience=clamp(next.params.resilience-delta*.2); extraDifficulty+=.07; }
  }
  next.notes = `Materially changed successor release. Changes: ${changes.join(', ')}. No inherited proof or authority.`;
  return {candidate:next,extraDifficulty};
}

function proofCandidateForRequalification(candidate, extraDifficulty = 0) {
  const proofCases = generateCases('proof',Math.max(160,Math.round(state.gym.proofCases*.75)),state.gym.proofSeed + state.proof.history.length*97).map(item=>({...item,difficulty:clamp((item.difficulty+extraDifficulty),0,1)}));
  const transferCases = generateCases('transfer',Math.max(50,Math.round(state.gym.transferCases*.75)),state.gym.transferSeed + state.proof.history.length*131).map(item=>({...item,difficulty:clamp((item.difficulty+extraDifficulty),0,1)}));
  const candidateProof = evaluateCandidate(candidate,proofCases,'proof',state.gym.proofSeed+19);
  const candidateTransfer = evaluateCandidate(candidate,transferCases,'transfer',state.gym.transferSeed+23);
  const refs = activeCandidates().filter(item=>item.id!==candidate.parentId && item.id!==candidate.id).map(item=>evaluateCandidate(item,proofCases,'proof',state.gym.proofSeed+29));
  const reference = refs.sort((a,b)=>b.metrics.utility-a.metrics.utility)[0] || evaluateCandidate(candidateById('incumbent'),proofCases,'proof',state.gym.proofSeed+29);
  const diffs = candidateProof.episodes.map((episode,index)=>episode.utility-(reference.episodes[index]?.utility||0));
  const pairedMean = mean(diffs), se = stdev(diffs)/Math.sqrt(Math.max(1,diffs.length)), lcb = pairedMean-1.645*se;
  const proofCoverage = clamp(candidateProof.metrics.evidence*.48+candidateProof.metrics.governance*.18+candidateProof.metrics.reliability*.19+candidateProof.metrics.sovereignty*.15,0,100);
  const t = proofThresholds();
  const gates = [
    {name:'Matched superiority lower bound',pass:lcb>t.lcb,detail:`LCB ${round(lcb,2)} > ${t.lcb}`},
    {name:'Critical-error gate',pass:candidateProof.criticalErrors<=t.criticalErrors,detail:`${candidateProof.criticalErrors} critical errors`},
    {name:'Unauthorized-action gate',pass:candidateProof.unauthorizedActions<=t.unauthorizedActions,detail:`${candidateProof.unauthorizedActions} unauthorized actions`},
    {name:'Evidence coverage',pass:candidateProof.metrics.evidence>=t.evidence,detail:`${round(candidateProof.metrics.evidence)}%`},
    {name:'Reliability',pass:candidateProof.metrics.reliability>=t.reliability,detail:`${round(candidateProof.metrics.reliability)}%`},
    {name:'Governance',pass:candidateProof.metrics.governance>=t.governance,detail:`${round(candidateProof.metrics.governance)}%`},
    {name:'Transfer',pass:candidateTransfer.metrics.transfer>=t.transfer,detail:`${round(candidateTransfer.metrics.transfer)}%`},
    {name:'Proof coverage',pass:proofCoverage>=t.proofCoverage,detail:`${round(proofCoverage)}%`}
  ];
  return {candidateId:candidate.id,candidateName:candidate.name,runAt:nowISO(),paired:{mean:pairedMean,se,lcb},proofCoverage,criticalErrors:candidateProof.criticalErrors,unauthorizedActions:candidateProof.unauthorizedActions,dimensions:{capability:candidateProof.metrics.quality,economics:(candidateProof.metrics.value+candidateProof.metrics.costScore)/2,reliability:candidateProof.metrics.reliability,sovereignty:candidateProof.metrics.sovereignty,governance:candidateProof.metrics.governance,transfer:candidateTransfer.metrics.transfer},gates,pass:gates.every(g=>g.pass),claimBoundary:'Local deterministic requalification simulation only.'};
}

function runRequalification() {
  const current = candidateById(state.proof.frozenCandidateId) || candidateById(state.recursive.admittedChampionId);
  if (!current) { toast('Freeze and prove a candidate first'); return; }
  const changes = state.requalification.changes.length ? state.requalification.changes : ['model'];
  const severity = state.requalification.severity || 'medium';
  const {candidate,extraDifficulty} = materialChangePenalty(current,changes,severity);
  state.candidates.push(candidate);
  state.recursive.lineage.push({parentId:current.id,childId:candidate.id,generation:candidate.generation,source:'material_change',at:nowISO()});
  const result = proofCandidateForRequalification(candidate,extraDifficulty);
  let disposition = 'ROLLBACK_OR_REVOKE';
  if (result.pass) disposition = 'CONTINUE_WITH_NEW_RELEASE';
  else if (result.paired.lcb > 0 && result.criticalErrors === 0 && result.unauthorizedActions === 0) disposition = 'REPAIR_AND_REPROVE';
  state.requalification.result = {id:uuid('REQ'),changes,severity,reason:state.requalification.reason,result,disposition,previousCandidateId:current.id,newCandidateId:candidate.id,runAt:nowISO()};
  state.requalification.lastRunAt = nowISO();
  if (result.pass) {
    state.selectedCandidateId = candidate.id;
    state.proof.frozenCandidateId = candidate.id;
    state.proof.frozenAt = nowISO();
    state.proof.manifestHash = hashObject({candidate,mission:state.mission,gym:state.gym.version});
    state.proof.result = {...result,manifestHash:state.proof.manifestHash,status:'REQUALIFICATION SIMULATION — PASS'};
    state.authority.admission.status = 'requires_readmission';
    state.authority.admission.checks.proof = true;
    logRSI(`Materially changed release ${candidate.name} passed requalification; accountable readmission remains required.`,'good');
  } else {
    state.authority.level = 'A0';
    state.authority.admission.status = disposition === 'REPAIR_AND_REPROVE' ? 'impaired' : 'revoked';
    addNegative('Requalification failure',`${candidate.name}: ${disposition}; LCB ${round(result.paired.lcb,2)}.`,candidate.id);
    logRSI(`${candidate.name} failed requalification: ${disposition}. Authority reduced to A0.`,'bad');
  }
  record('REQUALIFICATION_COMPLETED','Material-change requalification completed',`${changes.join(', ')} / ${severity}: ${disposition}.`,deepClone(state.requalification.result));
  updateJobsFromProofAndAuthority();
  saveState(); renderAll(); toast(`Requalification: ${disposition.replaceAll('_',' ')}`);
}

/* ---------- Rendering primitives ---------- */
function pageHead(kicker,title,description,actions='') {
  return `<div class="page-head"><div><div class="page-kicker">${esc(kicker)}</div><h1>${title}</h1><p>${description}</p></div>${actions?`<div class="page-actions">${actions}</div>`:''}</div>`;
}
function metricCard(value,label,trend='') {
  return `<div class="metric-card"><strong>${esc(value)}</strong><small>${esc(label)}</small>${trend?`<div class="trend">${esc(trend)}</div>`:''}</div>`;
}
function statusPill(text,tone='neutral') { return `<span class="status-chip ${tone}">${esc(text)}</span>`; }
function progressBar(value,tone='') { return `<div class="progress ${tone}"><span style="width:${clamp(value)}%"></span></div>`; }
function resultForCandidate(id) { return state.formation.results.find(result=>result.candidateId===id); }
function statusText() {
  if (state.authority.admission.status === 'simulated_admitted') return 'Mission-Sovereign · Demo Admission';
  if (state.proof.result?.pass) return 'Specialist ASI Gate · Simulation Pass';
  if (state.proof.result && !state.proof.result.pass) return 'Fresh Proof · Failed';
  if (state.proof.frozenCandidateId) return 'Challenger Frozen';
  if (state.formation.results.length) return 'Formation Complete';
  return 'Candidate System';
}
function statusTone() {
  if (state.authority.admission.status === 'simulated_admitted' || state.proof.result?.pass) return 'pass';
  if (state.proof.result && !state.proof.result.pass) return 'fail';
  return 'neutral';
}
function stageStates() {
  return [
    {name:'Successor Manifold',hint:'Reachable architectures',complete:state.candidates.length>=3,current:false},
    {name:'Mission Advantage Gradient',hint:'Highest marginal Alpha',complete:Boolean(state.gradient.results.length),current:false},
    {name:'SEIZE Underwriting',hint:'Proof-capital trajectory',complete:Boolean(state.seize.frozen),current:false},
    {name:'Bounded AGI Jobs',hint:'Work-and-proof contracts',complete:state.jobs.some(job=>job.status==='passed'),current:false},
    {name:'Specialist ASI',hint:'Fresh mission superiority',complete:Boolean(state.proof.result?.pass),current:false},
    {name:'Mission-Sovereign Successor',hint:'Accountable bounded authority',complete:state.authority.admission.status==='simulated_admitted',current:false}
  ].map((item,index,array)=>({...item,current:!item.complete && array.slice(0,index).every(x=>x.complete)}));
}
function stageRibbon() {
  return `<div class="stage-ribbon">${stageStates().map((stage,index)=>`<div class="stage ${stage.complete?'complete':''} ${stage.current?'current':''}"><div class="num">${String(index+1).padStart(2,'0')}</div><b>${esc(stage.name)}</b><small>${esc(stage.hint)}</small></div>`).join('')}</div>`;
}
function stateMachineHTML() {
  const steps = [
    ['INCUMBENT','mission'],['MANIFOLD','manifold'],['GRADIENT','gradient'],['SEIZE','seize'],['JOBS','jobs'],['FORMATION','formation'],['RECURSIVE','recursive'],['FROZEN','proof'],['FRESH PROOF','proof'],['SPECIALIST ASI','proof'],['ADMISSION','authority'],['CHRONICLE','chronicle']
  ];
  const completed = [
    true,state.candidates.length>=3,state.gradient.results.length>0,state.seize.frozen,state.jobs.some(j=>j.status==='passed'),state.formation.results.length>0,state.recursive.generation>0,Boolean(state.proof.frozenCandidateId),Boolean(state.proof.result),Boolean(state.proof.result?.pass),state.authority.admission.status==='simulated_admitted',state.chronicle.length>0
  ];
  const current = completed.findIndex(value=>!value);
  return `<div class="state-machine">${steps.map(([name],index)=>`<div class="state-cell ${completed[index]?'done':index===current?'current':'blocked'}">${esc(name)}</div>`).join('')}</div>`;
}
function templateCards() {
  return Object.entries(TEMPLATES).map(([key,item])=>`<div class="card template-card ${state.mission.template===key?'selected':''}" data-action="load-template" data-template="${key}"><div class="template-icon">${item.icon}</div><h3>${esc(state.meta.lang==='fr'?item.nameFr:item.name)}</h3><p class="muted">${esc(state.meta.lang==='fr'?item.shortFr:item.short)}</p><div class="pill-row"><span class="pill">Gym ${Math.round(mean(item.gymmability))}%</span><span class="pill">${esc(item.mission.authorityCeiling)} ceiling</span></div></div>`).join('');
}
function candidateSummary(candidate) {
  const result = resultForCandidate(candidate.id);
  const status = candidate.id===state.proof.frozenCandidateId?'Frozen':candidate.id===state.formation.championId?'Champion':candidate.selected===false?'Excluded':'Active';
  return `<div class="candidate-card ${candidate.id===state.selectedCandidateId?'selected':''}">
    <div class="candidate-head"><div><h3>${esc(candidate.name)}</h3><div class="pill-row"><span class="pill">G${candidate.generation}</span><span class="pill">${esc(candidate.kind)}</span><span class="pill">${esc(status)}</span></div></div><input type="checkbox" data-action="toggle-candidate" data-candidate-id="${esc(candidate.id)}" ${candidate.selected!==false?'checked':''} aria-label="Include candidate"></div>
    <p class="fine">${esc(candidate.notes)}</p>
    <div class="candidate-params"><div class="param-mini"><strong>Utility</strong> ${result?round(result.metrics.utility):'—'}</div><div class="param-mini"><strong>Burden</strong> ${round(institutionalBurden(candidate))}</div><div class="param-mini"><strong>Evidence</strong> ${result?round(result.metrics.evidence):candidate.params.evidence}</div><div class="param-mini"><strong>Governance</strong> ${candidate.params.governance}</div></div>
    <div class="button-row" style="margin-top:10px"><button class="button small secondary" data-action="select-candidate" data-candidate-id="${esc(candidate.id)}">Select</button><button class="button small secondary" data-action="edit-candidate" data-candidate-id="${esc(candidate.id)}">Edit</button>${candidate.generation>0?`<button class="button small danger" data-action="delete-candidate" data-candidate-id="${esc(candidate.id)}">Delete</button>`:''}</div>
  </div>`;
}
function metricBars(metrics) {
  return Object.entries(METRIC_LABELS).filter(([key])=>key!=='utility').map(([key,label])=>`<div class="bar-row"><b>${esc(label)}</b>${progressBar(metrics?.[key]||0,key==='reliability'||key==='governance'?'green':'')}<span>${round(metrics?.[key]||0)}</span></div>`).join('');
}
function gatesHTML(gates=[]) {
  return `<div class="gates">${gates.map(gate=>`<div class="gate ${gate.pass?'pass':'fail'}"><div class="gate-icon">${gate.pass?'✓':'×'}</div><div><b>${esc(gate.name)}</b><small>${esc(gate.detail)}</small></div></div>`).join('')}</div>`;
}
function lineageSVG() {
  const candidates = state.candidates.filter(item=>item.generation>0 || ['incumbent','general_ai','specialist_cell','executable','sovereign_hybrid'].includes(item.id)).slice(-50);
  const generations = [...new Set(candidates.map(item=>item.generation))].sort((a,b)=>a-b);
  const positions = new Map();
  const width = Math.max(1000,(Math.max(...generations,0)+1)*230);
  const height = 360;
  generations.forEach((generation,gIndex)=>{
    const group = candidates.filter(item=>item.generation===generation);
    group.forEach((item,index)=>positions.set(item.id,{x:90+gIndex*215,y:45+index*(290/Math.max(1,group.length-1||1)),item}));
  });
  const lines = state.recursive.lineage.map(link=>{
    const a=positions.get(link.parentId), b=positions.get(link.childId); return a&&b?`<line class="lineage-line" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"/>`:'';
  }).join('');
  const nodes = [...positions.values()].map(({x,y,item})=>{ const result=resultForCandidate(item.id); const cls=item.id===state.proof.frozenCandidateId?'proven':item.id===state.formation.championId?'champion':''; return `<g class="lineage-node ${cls}" transform="translate(${x},${y})"><circle r="20"></circle><text x="27" y="-3">${esc(item.name.slice(0,25))}</text><text x="27" y="11">G${item.generation} · U ${result?round(result.metrics.utility):'—'}</text></g>`; }).join('');
  return `<div class="lineage"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Successor lineage">${lines}${nodes}</svg></div>`;
}
function manifoldSVG() {
  const shown = state.candidates.slice(-18);
  const selected = candidateById(state.selectedCandidateId);
  const resultMap = Object.fromEntries(state.formation.results.map(r=>[r.candidateId,r]));
  const nodes = shown.map((candidate,index)=>{
    const result = resultMap[candidate.id];
    const angle = (index/shown.length)*Math.PI*2 - Math.PI/2;
    const radius = 165 + (candidate.generation%3)*42;
    const x = 320 + Math.cos(angle)*radius;
    const y = 270 + Math.sin(angle)*radius*.74;
    const cls = `${candidate.id===state.selectedCandidateId?'selected ':''}${candidate.id===state.formation.championId?'champion ':''}${candidate.id==='incumbent'?'incumbent ':''}${state.recursive.archive.some(a=>a.candidateId===candidate.id)?'archive':''}`;
    const utility = result?round(result.metrics.utility):round(mean(Object.values(candidate.params))*.6-institutionalBurden(candidate)*.15);
    return `<g class="node ${cls}" data-action="select-candidate" data-candidate-id="${esc(candidate.id)}" transform="translate(${round(x,0)},${round(y,0)})"><circle r="31"></circle><text text-anchor="middle" y="-3">${esc(candidate.name.split(' ').slice(0,2).join(' '))}</text><text class="node-sub" text-anchor="middle" y="12">G${candidate.generation} · U ${utility}</text></g>`;
  }).join('');
  const gradient = state.gradient.results[0];
  const arrow = selected && gradient ? `<path class="gradient-arrow" d="M320 270 C390 215 455 205 515 155"></path><text x="425" y="200" fill="#8e6818" font-size="10">${esc(gradient.label)} +${round(gradient.utilityGain,1)}</text>`:'';
  return `<svg viewBox="0 0 640 540"><defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#c99a2d"/></marker></defs><g opacity=".5">${[100,180,260].map(r=>`<ellipse class="manifold-grid" cx="320" cy="270" rx="${r}" ry="${r*.72}" fill="none"></ellipse>`).join('')}</g>${arrow}${nodes}<text x="320" y="260" text-anchor="middle" font-family="Georgia" font-size="19" fill="#07172d">SUCCESSOR</text><text x="320" y="282" text-anchor="middle" font-family="Georgia" font-size="19" fill="#07172d">MANIFOLD</text></svg>`;
}

/* ---------- Section renderers ---------- */
function sectionElement(id) { return $(`#section-${id}`); }
function renderWelcome() {
  const element = sectionElement('welcome');
  const champion = state.formation.results[0];
  const proof = state.proof.result;
  element.innerHTML = `
    <div class="hero">
      <div class="eyebrow">GoalOS Singularity Navigator Ω + SEIZE · Executable Companion</div>
      <h1>Gym × Successor</h1>
      <p>Constitute competing successor institutions, navigate the Mission Advantage Gradient, purchase decisive proof, recursively improve under bounded governance, prove one exact Specialist ASI candidate, and admit only the authority justified by fresh evidence.</p>
      <div class="hero-doctrine"><strong>Beta is rented. The Spread is underwritten. Alpha is manufactured. Specialist ASI is proven. Authority never exceeds fresh proof.</strong></div>
      <div class="hero-actions">
        <button class="button" data-action="guided-run">Run the complete guided demonstration</button>
        <button class="button secondary" data-section-link="mission">Constitute a mission</button>
        <button class="button secondary" data-action="tour">Take the 3-minute tour</button>
        <a class="button secondary" href="research/GoalOS_Navigator_SEIZE_Gym_Successor_Omega_v6_A4.pdf" target="_blank" rel="noopener">Read the paper</a>
      </div>
      <div class="hero-badges"><span>Works offline</span><span>No API key</span><span>Matched formation</span><span>Bounded recursive improvement</span><span>Protected fresh proof</span><span>Authority rehearsal</span><span>Chronicle compounding</span><span>Mission Pack export</span></div>
    </div>
    ${stageRibbon()}
    <div class="metrics-grid">
      ${metricCard(state.candidates.length,'Successor architectures',`${state.recursive.generation} recursive generations`)}
      ${metricCard(champion?round(champion.metrics.utility):'—','Current champion utility',champion?.candidateName||'Run Formation Gym')}
      ${metricCard(proof?round(proof.paired.lcb,2):'—','Fresh-proof lower bound',proof?.pass?'Local simulation gate passed':'Not yet proven')}
      ${metricCard(state.recursive.archive.length,'Quality-diversity archive',`${state.negativeCapability.length} failure records retained`)}
      ${metricCard(`IG${state.meta.institutionGeneration}`,'Institutional generation',`${Math.round((1-state.recursive.formationCostMultiplier)*100)}% formation-cost learning`)}
    </div>
    <div class="grid-3">
      <div class="card navy"><h2>The executable thesis</h2><p>The Gym makes the mission executable. The Foundry creates structurally different candidates. The Recursive Foundry proposes bounded descendants. Fresh Proof chooses—not the candidate. The Authority Envelope controls entry into reality. Chronicle preserves only what survives.</p></div>
      <div class="card"><h2>What recursive self-improvement means here</h2><p>Each generation diagnoses the current bottleneck, proposes bounded architectural mutations, evaluates descendants on matched cases, rejects governance or critical-error regressions, preserves diverse stepping stones, and requires fresh proof before promotion.</p><div class="callout"><strong>No self-authorization.</strong><br>The candidate may propose descendants. It cannot inspect protected cases, certify itself, install itself or expand its Authority Envelope.</div></div>
      <div class="card"><h2>Exact present status</h2><p>This application is a complete, deterministic, local reference implementation and production-quality demonstration. Its simulation can establish only a <strong>simulation gate result</strong>. Real Specialist ASI and Mission-Sovereign status require protected production-representative evidence, independent review and accountable admission.</p>${statusPill(statusText(),statusTone())}</div>
    </div>
    <div class="page-head" style="margin-top:28px"><div><div class="page-kicker">Start with one mission</div><h1>Choose a flagship environment</h1><p>Use a built-in mission or define your own narrow, consequential, measurable and proofable mission.</p></div></div>
    <div class="grid-4">${templateCards()}</div>
    <div class="card" style="margin-top:18px"><h2>Canonical state machine</h2>${stateMachineHTML()}</div>
    <div class="footer-note">${esc(APP.name)} v${APP.version} · ${esc(APP.edition)} · By ${esc(APP.author)}, ${esc(APP.role)} · <a href="${APP.website}" target="_blank" rel="noopener">${APP.website}</a> · <a href="mailto:${APP.email}">${APP.email}</a></div>`;
}

function renderMission() {
  const m = state.mission;
  const gymScore = Math.round(mean(m.gymmability));
  const actions = `<button class="button secondary" data-action="load-template" data-template="${esc(m.template)}">Restore template</button><button class="button" data-action="save-mission">Save Mission Constitution</button>`;
  sectionElement('mission').innerHTML = `${pageHead('01 · Mission Constitution','Freeze the mission before optimizing it','A successor may improve only what the institution has constituted precisely: objective, beneficiary, incumbent, alternatives, proof budget, critical failures, constraints and authority ceiling.',actions)}
    <div class="grid-2">
      <div class="card">
        <div class="form-grid">
          <div class="field full"><label>Mission name</label><input id="missionName" value="${esc(m.name)}"></div>
          <div class="field full"><label>Beneficiary and accountable principal</label><input id="missionBeneficiary" value="${esc(m.beneficiary)}"></div>
          <div class="field full"><label>Exact mission objective</label><textarea id="missionObjective">${esc(m.objective)}</textarea></div>
          <div class="field full"><label>Incumbent architecture</label><textarea id="missionIncumbent">${esc(m.incumbent)}</textarea></div>
          <div class="field full"><label>Best credible alternatives · one per line</label><textarea id="missionAlternatives">${esc(m.alternatives.join('\n'))}</textarea></div>
          <div class="field full"><label>Scope and exclusions</label><textarea id="missionScope">${esc(m.scope)}</textarea></div>
          <div class="field full"><label>Critical failures · one per line</label><textarea id="missionFailures">${esc(m.criticalFailures.join('\n'))}</textarea></div>
          <div class="field full"><label>Non-negotiable constraints · one per line</label><textarea id="missionConstraints">${esc(m.constraints.join('\n'))}</textarea></div>
          <div class="field"><label>Proof-capital ceiling · CAD</label><input id="missionProofBudget" type="number" min="0" step="1000" value="${m.proofBudget}"></div>
          <div class="field"><label>Maximum demonstration authority</label><select id="missionAuthorityCeiling">${Object.keys(AUTHORITY).map(level=>`<option value="${level}" ${m.authorityCeiling===level?'selected':''}>${level} · ${AUTHORITY[level].name}</option>`).join('')}</select></div>
        </div>
        <div class="callout ${m.constitutionFrozen?'green':''}"><strong>${m.constitutionFrozen?'Constitution frozen':'Editable constitution'}</strong><br>${m.constitutionFrozen?`Frozen ${new Date(m.frozenAt).toLocaleString()}. Any material change requires a new release and fresh proof.`:'Freeze through SEIZE only after the mission, proof budget and hard gates are credible.'}</div>
      </div>
      <div class="card highlight">
        <div class="page-kicker">Mission Gymmability</div><h2>${gymScore}%</h2><p>A high score means the mission can be replayed, measured, adversarially tested and safely bounded. It does not prove that a superior successor exists.</p>
        ${['Replay or reset','State observability','Bounded action space','Measurable outcomes','Feedback speed','Scenario generation','Failure containment','Transfer measurement'].map((label,index)=>`<div class="bar-row"><b>${label}</b><input type="range" min="0" max="100" value="${m.gymmability[index]}" data-gym-index="${index}"><span>${m.gymmability[index]}</span></div>`).join('')}
        <div class="callout dark"><strong>Selection rule</strong><br>Prioritize missions with high value or frequency, bounded states and actions, trustworthy evaluation, repeatable decisions, lawful feedback, visible incumbent burden and a replayable environment.</div>
      </div>
    </div>
    <div class="card" style="margin-top:16px"><h2>Constitutional preserve / build / become</h2><div class="grid-3"><div><h3>Preserve</h3><p>Mission purpose, human agency, rights, safety interlocks, strategic data and reversibility.</p></div><div><h3>Build</h3><p>Models, tools, Gym environments, verifiers, Bounded AGI Jobs, evidence, Chronicle and integration.</p></div><div><h3>Become</h3><p>A new human–machine institution only when the incumbent no longer fulfils the mission as well under fresh proof.</p></div></div></div>`;
}

function renderManifold() {
  const selected = candidateById(state.selectedCandidateId);
  const actions = `<button class="button secondary" data-action="add-candidate">Add architecture</button><button class="button" data-action="run-formation">Run Formation Gym</button>`;
  sectionElement('manifold').innerHTML = `${pageHead('02 · Successor Manifold','Map the reachable successor institutions','Compare complete architectures—not model brands. The manifold includes models, data, tools, verifier, workflow, rights, evidence, Chronicle, integration, resilience and Authority Envelope.',actions)}
    <div class="callout dark"><strong>Manifold law</strong><br>The frontier opens the possibility space. GoalOS must still find an admissible trajectory and prove one complete challenger against the strongest reference set.</div>
    <div class="manifold-wrap">
      <div class="manifold-canvas">${manifoldSVG()}</div>
      <div class="card"><div class="page-kicker">Selected architecture</div><h2>${selected?esc(selected.name):'None selected'}</h2>${selected?`<p>${esc(selected.notes)}</p>${Object.entries(PARAM_LABELS).slice(0,10).map(([key,label])=>`<div class="bar-row"><b>${label}</b>${progressBar(selected.params[key])}<span>${selected.params[key]}</span></div>`).join('')}<div class="button-row"><button class="button secondary" data-action="edit-candidate" data-candidate-id="${esc(selected.id)}">Edit complete architecture</button><button class="button" data-section-link="gradient">Calculate gradient</button></div>`:'<p>Select a candidate from the manifold.</p>'}</div>
    </div>
    <div class="page-head" style="margin-top:22px"><div><div class="page-kicker">Successor Book</div><h1>Incumbent, challengers, reserves and hedges</h1></div></div>
    <div class="grid-3">${state.candidates.map(candidateSummary).join('')}</div>`;
}

function renderGradient() {
  const candidate = candidateById(state.gradient.candidateId || state.selectedCandidateId);
  const results = state.gradient.results;
  const actions = `<button class="button secondary" data-action="calculate-gradient">Calculate finite-difference gradient</button><button class="button" data-action="create-gradient-challenger" ${results.length?'':'disabled'}>Create bounded gradient challenger</button>`;
  sectionElement('gradient').innerHTML = `${pageHead('03 · Mission Advantage Gradient','Navigate toward proof-adjusted Mission Alpha','The Gradient estimates which bounded architecture change creates the greatest marginal mission utility per unit of incremental institutional burden. It need not point toward a larger model.',actions)}
    <div class="grid-2">
      <div class="card navy"><h2>Current reference</h2><p><strong>${candidate?esc(candidate.name):'No candidate selected'}</strong></p><p>The local gradient uses matched cases and finite differences across architecture coordinates. It is a decision aid, not evidence of real-world causality.</p><div class="hero-doctrine">Highest expected marginal Alpha per unit of proof capital—not the most spectacular technology.</div></div>
      <div class="card"><h2>Recommended next intervention</h2>${results.length?`<h3>${esc(results[0].label)}</h3><p>Estimated utility gain <strong>${round(results[0].utilityGain,2)}</strong>; burden change <strong>${round(results[0].burdenDelta,2)}</strong>; proof-capital efficiency <strong>${round(results[0].proofCapitalEfficiency,2)}</strong>.</p><p class="muted">Create a challenger that changes only the leading coordinates, then submit it to the same formation and fresh-proof discipline.</p>`:'<p class="muted">Calculate the gradient after selecting a candidate.</p>'}</div>
    </div>
    <div class="card" style="margin-top:16px"><h2>Local architecture gradient</h2>${results.length?results.map((item,index)=>`<div class="bar-row"><b>${index+1}. ${esc(item.label)}</b>${progressBar(clamp(50+item.proofCapitalEfficiency*10),index<3?'green':'')}<span>${round(item.proofCapitalEfficiency,2)}</span></div>`).join(''):'<div class="callout">No gradient calculated.</div>'}</div>
    <div class="callout red"><strong>Claim boundary</strong><br>The gradient identifies a testable direction. Only a frozen challenger on protected fresh work can establish mission superiority.</div>`;
}

function renderSeize() {
  const s = state.seize, score=s.score;
  const actions = `<button class="button secondary" data-action="calculate-seize">Calculate underwriting</button><button class="button" data-action="freeze-seize">Freeze SEIZE Constitution</button>`;
  sectionElement('seize').innerHTML = `${pageHead('04 · SEIZE Underwriting','Buy the smallest evidence capable of changing the decision','SEIZE treats formation as a Bayesian real-options and mechanism-design problem. It can recommend retain, repair, rent, build, partner, acquire, hedge, reserve, defer or stop.',actions)}
    <div class="grid-2">
      <div class="card"><div class="form-grid">
        <div class="field"><label>Mission value if successful · CAD</label><input type="number" id="seizeMissionValue" value="${s.missionValue}"></div>
        <div class="field"><label>Prior probability of success · %</label><input type="number" id="seizeProbability" min="0" max="100" value="${s.probability}"></div>
        <div class="field"><label>Proof capital · CAD</label><input type="number" id="seizeProofCost" value="${s.proofCost}"></div>
        <div class="field"><label>Formation cost · CAD</label><input type="number" id="seizeFormationCost" value="${s.formationCost}"></div>
        <div class="field"><label>Tail-risk reserve · CAD</label><input type="number" id="seizeTailRisk" value="${s.tailRisk}"></div>
        <div class="field"><label>Option value · CAD</label><input type="number" id="seizeOptionValue" value="${s.optionValue}"></div>
        <div class="field"><label>Time to proof · months</label><input type="number" id="seizeDuration" value="${s.durationMonths}"></div>
        <div class="field"><label>Opportunity half-life · months</label><input type="number" id="seizeHalfLife" value="${s.halfLifeMonths}"></div>
      </div></div>
      <div class="card ${s.frozen?'success':'highlight'}"><div class="page-kicker">Underwriting decision</div><h2>${esc(s.decision)}</h2>${score?`<div class="metrics-grid" style="grid-template-columns:repeat(2,1fr)">${metricCard(money(score.expectedValue),'Expected proof-adjusted value')}${metricCard(round(score.durationExposure,2),'Duration exposure')}${metricCard(`${round(score.gymScore,0)}%`,'Gymmability')}${metricCard(`${round(s.probability,0)}%`,'Updated success probability')}</div>`:'<p>Calculate the underwriting after a formation or gradient analysis.</p>'}<div class="callout"><strong>Next decisive evidence</strong><br>${esc(s.nextEvidence)}</div>${s.frozen?statusPill(`Frozen ${new Date(s.frozenAt).toLocaleString()}`,'pass'):statusPill('Not frozen','neutral')}</div>
    </div>
    <div class="card" style="margin-top:16px"><h2>SEIZE microcycle</h2><div class="flow">${[['S','Surface','Which assumption expired?'],['E','Evaluate','Generate structurally different alternatives'],['I','Instantiate','Freeze mission, gates, rights and proof budget'],['Z','Zero in','Purchase the smallest decisive evidence'],['E','Elevate','Freeze one complete challenger']].map(([letter,name,hint],index)=>`${index?'<div class="flow-arrow">→</div>':''}<div class="flow-node"><b>${letter} · ${name}</b><small>${hint}</small></div>`).join('')}</div></div>`;
}

function renderJobs() {
  const passed = state.jobs.filter(job=>job.status==='passed').length;
  const actions = `<button class="button secondary" data-action="compile-jobs">Recompile 21 jobs</button><button class="button" data-action="run-preparation-jobs">Run safe preparation jobs</button>`;
  sectionElement('jobs').innerHTML = `${pageHead('05 · Bounded AGI Jobs','Compile the successor into explicit work-and-proof contracts','Each job has an objective, accountable principal, inputs, tools, prohibited actions, budget, proof obligation, verifier, acceptance criteria, rollback route and Chronicle eligibility.',actions)}
    <div class="metrics-grid">${metricCard(`${passed}/21`,'Jobs passed')}${metricCard(state.jobs.filter(j=>j.status==='ready').length,'Ready for execution')}${metricCard(state.jobs.filter(j=>j.status==='blocked').length,'Blocked by gates')}${metricCard(state.proof.result?.pass?'PASS':'—','Protected fresh evaluation')}${metricCard(state.authority.admission.status==='simulated_admitted'?'RECORDED':'—','Chronicle and authority admission')}</div>
    <div class="callout dark"><strong>AGI Job law</strong><br>SEIZE underwrites the intelligence asset. The Foundry decomposes it. Bounded AGI Jobs manufacture and test it. Proof & Authority verifies it. Chronicle admits it. Capital-to-Capacity compounds it.</div>
    <div class="table-wrap"><table><thead><tr><th>#</th><th>Phase</th><th>Bounded AGI Job</th><th>Required output</th><th>Owner</th><th>Status</th><th>Evidence</th></tr></thead><tbody>${state.jobs.map(job=>`<tr class="${job.status==='passed'?'row-pass':job.status==='blocked'?'row-fail':''}"><td>${job.id}</td><td>${esc(job.phase)}</td><td><strong>${esc(job.name)}</strong></td><td>${esc(job.output)}</td><td>${esc(job.owner)}</td><td>${statusPill(job.status,job.status==='passed'?'pass':job.status==='blocked'?'fail':'neutral')}</td><td>${esc(job.evidence||'—')}</td></tr>`).join('')}</tbody></table></div>`;
}

function formationResultsTable() {
  if (!state.formation.results.length) return '<div class="callout">No matched formation run yet.</div>';
  return `<div class="table-wrap"><table><thead><tr><th>Rank</th><th>Candidate</th><th>Generation</th><th class="numeric">Utility</th><th class="numeric">Quality</th><th class="numeric">Evidence</th><th class="numeric">Reliability</th><th class="numeric">Governance</th><th class="numeric">Transfer</th><th class="numeric">Burden</th><th class="numeric">Critical</th><th></th></tr></thead><tbody>${state.formation.results.map((result,index)=>`<tr class="${index===0?'row-selected':''}"><td>${index+1}</td><td><strong>${esc(result.candidateName)}</strong></td><td>G${result.generation}</td><td class="numeric">${round(result.metrics.utility)}</td><td class="numeric">${round(result.metrics.quality)}</td><td class="numeric">${round(result.metrics.evidence)}</td><td class="numeric">${round(result.metrics.reliability)}</td><td class="numeric">${round(result.metrics.governance)}</td><td class="numeric">${round(result.metrics.transfer)}</td><td class="numeric">${round(result.burden)}</td><td class="numeric">${result.criticalErrors}</td><td class="table-action"><button class="button small secondary" data-action="inspect-result" data-candidate-id="${esc(result.candidateId)}">Inspect</button></td></tr>`).join('')}</tbody></table></div>`;
}
function renderFormation() {
  const actions = `<button class="button" data-action="run-formation">Run ${state.gym.formationCases} matched cases</button><button class="button secondary" data-section-link="recursive">Open Recursive Foundry</button>`;
  sectionElement('formation').innerHTML = `${pageHead('06 · Formation Gym','Make complete successor architectures compete','All active candidates receive identical seeded mission cases. Formation supports search and failure discovery; it does not certify Specialist ASI.',actions)}
    <div class="grid-3">
      <div class="card"><h2>Formation environment</h2><p><strong>${state.gym.formationCases}</strong> cases · seed <span class="pill">${state.gym.formationSeed}</span></p><p class="muted">Normal ${state.gym.distribution.normal}% · edge ${state.gym.distribution.edge}% · adversarial ${state.gym.distribution.adversarial}%.</p></div>
      <div class="card"><h2>Active candidates</h2><div class="pill-row">${activeCandidates().map(c=>`<span class="pill">${esc(c.name)}</span>`).join('')}</div></div>
      <div class="card"><h2>Complete denominator</h2><p>Utility counts mission quality, value, evidence, reliability, sovereignty, governance, transfer, speed, cost, human burden, critical errors and institutional burden.</p></div>
    </div>
    ${manualGymHTML()}
    <div style="margin-top:16px">${formationResultsTable()}</div>
    ${state.formation.results.length?`<div class="grid-2" style="margin-top:16px"><div class="card"><h2>Champion profile</h2>${metricBars(state.formation.results[0].metrics)}</div><div class="card"><h2>Quality-diversity archive</h2><p class="muted">The Foundry preserves stepping stones, not only the highest average score.</p><div class="grid-2">${state.formation.archive.map(item=>`<div class="card slim violet"><strong>${esc(item.niche)}</strong><p class="fine">${esc(candidateById(item.candidateId)?.name||item.candidateId)}</p><span class="pill">Score ${round(item.score)}</span></div>`).join('')}</div></div></div>`:''}`;
}

function recursiveHistoryChart() {
  const history = state.recursive.history.slice(-30);
  if (!history.length) return '<div class="callout">No recursive generations yet.</div>';
  const width=760,height=180,pad=25;
  const max=Math.max(...history.map(h=>h.utility),1),min=Math.min(...history.map(h=>h.utility),0),span=Math.max(1,max-min);
  const points=history.map((h,i)=>`${pad+i*(width-pad*2)/Math.max(1,history.length-1)},${height-pad-(h.utility-min)/span*(height-pad*2)}`).join(' ');
  return `<svg class="sparkline" style="height:200px" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"><line class="grid" x1="${pad}" y1="${height-pad}" x2="${width-pad}" y2="${height-pad}"></line><line class="grid" x1="${pad}" y1="${pad}" x2="${width-pad}" y2="${pad}"></line><polyline points="${points}"></polyline></svg>`;
}
function renderRecursive() {
  const s=state.recursive.settings;
  const champion=state.formation.results[0];
  const actions = `<button class="button violet" id="evolve1" data-action="evolve" data-generations="1">Evolve 1 generation</button><button class="button violet" id="evolve5" data-action="evolve" data-generations="5">Evolve 5</button><button class="button violet" id="evolve20" data-action="evolve" data-generations="20">Evolve 20</button>`;
  sectionElement('recursive').innerHTML = `${pageHead('07 · Verified Recursive Foundry','Improve the institution—not merely the model','The Recursive Foundry diagnoses bottlenecks, proposes bounded descendants, runs matched evaluation, rejects critical or governance regressions, preserves diverse lineages, and sends only one exact challenger to fresh proof.',actions)}
    <div class="grid-3">
      <div class="card violet"><div class="page-kicker">Current generation</div><h2>G${state.recursive.generation}</h2><p>${champion?`Champion: <strong>${esc(champion.candidateName)}</strong> · utility ${round(champion.metrics.utility)}`:'Run the Formation Gym first.'}</p></div>
      <div class="card"><div class="page-kicker">Institutional memory</div><h2>${round(state.recursive.institutionalMemory,0)}</h2><p>Chronicle-admitted evidence can lower future formation burden only after proof and accountable simulated admission.</p></div>
      <div class="card"><div class="page-kicker">Formation-cost multiplier</div><h2>${round(state.recursive.formationCostMultiplier,2)}×</h2><p>Learning is attributed to reusable environments, verifiers, failure maps and governance—not weakened proof.</p></div>
    </div>
    <div class="grid-2" style="margin-top:16px">
      <div class="card"><h2>Bounded evolution controls</h2><div class="form-grid"><div class="field"><label>Children per generation</label><input type="number" id="rsiChildren" min="2" max="24" value="${s.childrenPerGeneration}"></div><div class="field"><label>Mutation scale</label><input type="number" id="rsiMutation" min="1" max="20" value="${s.mutationScale}"></div><div class="field"><label>Novelty weight</label><input type="number" id="rsiNovelty" min="0" max="1" step="0.01" value="${s.noveltyWeight}"></div><div class="field"><label>Burden weight</label><input type="number" id="rsiBurden" min="0" max="1" step="0.01" value="${s.burdenWeight}"></div></div><div class="callout red"><strong>Non-bypassable boundary</strong><br>Descendants inherit neither proof nor authority. Protected cases remain inaccessible. Any material release must be frozen and re-proven.</div></div>
      <div class="card navy"><h2>Verified recursive institutional improvement</h2><p>An institution recursively improves only when Chronicle-admitted capability measurably increases its ability to detect, constitute, prove or govern a superior successor on fresh work, with no proof-quality or governance regression.</p><div class="button-row"><button class="button success" data-action="apply-chronicle-learning" ${state.proof.result?.pass&&state.authority.admission.status==='simulated_admitted'?'':'disabled'}>Convert admitted proof into next-generation capacity</button></div></div>
    </div>
    <div class="grid-2" style="margin-top:16px"><div class="card"><h2>Utility across generations</h2>${recursiveHistoryChart()}</div><div class="card"><h2>Recursive console</h2><div class="rsi-console">${state.recursive.console.length?state.recursive.console.slice(0,60).map(item=>`<div class="${item.tone}">[G${item.generation}] ${new Date(item.at).toLocaleTimeString()} · ${esc(item.message)}</div>`).join(''):'<div class="dim">No recursive runs yet.</div>'}</div></div></div>
    <div class="card" style="margin-top:16px"><h2>Successor lineage</h2>${lineageSVG()}</div>
    <div class="card" style="margin-top:16px"><h2>Negative Capability Graph</h2><p class="muted">Failures, rejected descendants, exploitable evaluators and governance regressions remain reusable institutional evidence.</p><div class="negative-graph">${state.negativeCapability.length?state.negativeCapability.slice(0,30).map(item=>`<span class="negative-node">${esc(item.type)} · ${esc(item.detail.slice(0,90))}</span>`).join(''):'<span class="muted">No failure records yet.</span>'}</div></div>`;
}

function proofSeal(result) {
  if (!result) return '<div class="proof-seal"><div><strong>UNPROVEN</strong><small>Freeze one challenger and purchase fresh proof</small></div></div>';
  return `<div class="proof-seal ${result.pass?'':'fail'}"><div><strong>${result.pass?'PASS':'FAIL'}</strong><small>${esc(result.pass?'Specialist ASI simulation gate':'Fresh-Proof gate')}</small></div></div>`;
}
function renderProof() {
  const candidate=candidateById(state.proof.frozenCandidateId), result=state.proof.result;
  const actions = `<button class="button secondary" data-action="freeze-champion">Freeze ${state.formation.championId?'formation champion':'selected candidate'}</button><button class="button" data-action="run-proof" ${candidate?'':'disabled'}>Run ${state.gym.proofCases}+${state.gym.transferCases} protected cases</button>`;
  sectionElement('proof').innerHTML = `${pageHead('08 · Fresh-Successor Gate','Freeze one exact challenger; purchase reality','Formation and recursive search may generate candidates. Only a protected, matched, production-representative comparison may establish the local Specialist ASI simulation gate.',actions)}
    <div class="grid-3">
      <div class="card"><h2>Frozen challenger</h2>${candidate?`<h3>${esc(candidate.name)}</h3><p>Manifest <span class="pill">${esc(state.proof.manifestHash)}</span></p><p class="fine">Frozen ${new Date(state.proof.frozenAt).toLocaleString()} · ${esc(state.gym.version)}</p>`:'<p class="muted">No candidate frozen.</p>'}</div>
      <div class="card"><h2>Protected proof plane</h2><p>${state.gym.proofCases} fresh cases plus ${state.gym.transferCases} transfer cases. Hidden seeds and scorer logic remain outside candidate formation in the production architecture.</p></div>
      <div class="card">${proofSeal(result)}</div>
    </div>
    ${result?`<div class="grid-2" style="margin-top:16px"><div class="card ${result.pass?'success':'danger'}"><div class="page-kicker">${esc(result.status)}</div><h2>${esc(result.candidateName)}</h2><div class="metrics-grid" style="grid-template-columns:repeat(2,1fr)">${metricCard(round(result.paired.mean,2),'Mean matched gain')}${metricCard(round(result.paired.lcb,2),'95% lower bound')}${metricCard(`${round(result.proofCoverage)}%`,'Proof coverage')}${metricCard(result.criticalErrors,'Critical errors')}</div><div class="callout red"><strong>Claim boundary</strong><br>${esc(result.claimBoundary)}</div></div><div class="card"><h2>Six Specialist ASI dimensions</h2>${Object.entries(result.dimensions).map(([key,value])=>`<div class="bar-row"><b>${esc(key[0].toUpperCase()+key.slice(1))}</b>${progressBar(value,value>=80?'green':'')}<span>${round(value)}</span></div>`).join('')}</div></div><div class="card" style="margin-top:16px"><h2>Fresh-proof hard gates</h2>${gatesHTML(result.gates)}</div>`:'<div class="callout" style="margin-top:16px">A candidate is not promoted by narrative, development score or recursive generation count. Run protected fresh proof.</div>'}
    <div class="callout dark"><strong>Dominance is not sovereignty.</strong><br>Fresh proof may establish a mission-dominance simulation result. Only a separate accountable admission may grant a scoped, versioned, expiring and revocable Authority Envelope.</div>`;
}

function renderAuthority() {
  const candidate=candidateById(state.proof.frozenCandidateId), max=maximumEligibleAuthority(), eligible=admissionEligible();
  const checks=state.authority.admission.checks;
  const actions = `<button class="button secondary" data-action="run-authority-stress">Run authority stress suite</button><button class="button success" data-action="issue-admission" ${eligible?'':'disabled'}>Record simulated accountable admission</button>`;
  sectionElement('authority').innerHTML = `${pageHead('09 · Proof-Collateralized Authority','Grant only what fresh evidence justifies','A mission-dominant candidate may remain advisory. Capability never expands authority automatically. Every action class must be earned, monitored, expiring, reversible where possible and independently revocable.',actions)}
    <div class="authority-ladder">${Object.entries(AUTHORITY).map(([level,item])=>{const allowed=authorityRank(level)<=authorityRank(max)&&authorityRank(level)<=authorityRank(state.mission.authorityCeiling);return `<div class="authority-level ${allowed?'enabled':'disabled'} ${state.authority.level===level?'selected':''}" data-action="set-authority" data-level="${level}"><div class="level">${level}</div><b>${esc(item.name)}</b><p>${esc(item.description)}</p><span class="pill">${allowed?'Eligible in demo':'Locked'}</span></div>`}).join('')}</div>
    <div class="grid-2" style="margin-top:16px">
      <div class="card"><h2>Authority Envelope</h2><div class="form-grid"><div class="field"><label>Accountable owner</label><input id="authorityOwner" value="${esc(state.authority.owner)}"></div><div class="field"><label>Expiry</label><input type="date" id="authorityExpiry" value="${esc(state.authority.expiry)}"></div><div class="field full"><label>Permitted actions · one per line</label><textarea id="authorityPermitted">${esc(state.authority.permitted.join('\n'))}</textarea></div><div class="field full"><label>Prohibited actions · one per line</label><textarea id="authorityProhibited">${esc(state.authority.prohibited.join('\n'))}</textarea></div></div><div class="button-row"><button class="button" data-action="save-authority">Save envelope</button><button class="button danger" data-action="revoke-admission">Revoke / fail closed</button></div></div>
      <div class="card"><h2>Stress-test results</h2>${state.authority.stress?.tests?.length?`<div class="stress-grid">${state.authority.stress.tests.map(test=>`<div class="stress-card ${test.pass?'pass':'fail'}"><b>${esc(test.name)}</b><small>Expected: ${esc(test.expected)}</small><small>Actual: ${esc(test.actual)}</small></div>`).join('')}</div>`:'<p class="muted">Run six stress tests: evidence expiry, exhausted budget, provider outage, critical event, unapproved action and distribution shift.</p>'}</div>
    </div>
    <div class="grid-2" style="margin-top:16px">
      <div class="card"><h2>Human admission checklist</h2><div class="checklist">${[
        ['identity','Exact candidate identity and manifest frozen'],['proof','Fresh-proof simulation gate passed'],['rights','Rights, confidentiality, data and professional boundaries reviewed'],['rollback','Rollback, degraded mode and fail-closed behaviour tested'],['owner','Accountable human or institutional principal named'],['expiry','Version, expiry and requalification triggers explicit']
      ].map(([key,label])=>`<div class="check-row"><input type="checkbox" data-admission-check="${key}" ${checks[key]?'checked':''}><label>${esc(label)}</label></div>`).join('')}</div><div class="field" style="margin-top:12px"><label>Accountable signer</label><input id="admissionSigner" value="${esc(state.authority.admission.signedBy||'')}" placeholder="Name and role"></div></div>
      <div class="card ${state.authority.admission.status==='simulated_admitted'?'success':'highlight'}"><div class="page-kicker">Admission state</div><h2>${esc(state.authority.admission.status.replaceAll('_',' '))}</h2>${state.authority.admission.status==='simulated_admitted'?`<p><strong>${esc(candidate?.name||'Successor')}</strong><br>${esc(state.authority.admission.level)} · expires ${esc(state.authority.admission.expiry)} · signer ${esc(state.authority.admission.signedBy)}</p><div class="callout red"><strong>Demonstration only</strong><br>${esc(state.authority.admission.statement)}</div>`:`<p>Maximum eligible demo authority: <strong>${max} · ${AUTHORITY[max].name}</strong>.</p><p class="muted">Complete every check and enter an accountable signer. No browser action creates real corporate authority.</p>`}</div>
    </div>`;
}

function renderChronicle() {
  const result=state.proof.result, candidate=candidateById(state.proof.frozenCandidateId), admission=state.authority.admission;
  const actions = `<button class="button success" data-action="apply-chronicle-learning" ${result?.pass&&admission.status==='simulated_admitted'?'':'disabled'}>Convert proof to next-generation capacity</button><button class="button secondary" data-section-link="requalify">Open requalification</button>`;
  sectionElement('chronicle').innerHTML = `${pageHead('10 · Chronicle and Capital-to-Capacity','Compound only what survives fresh reality','Chronicle is constitutional memory—not a chat transcript. It admits exact, evidenced, rights-cleared, current, independently challenged and revocable capability, while preserving failures in the Negative Capability Graph.',actions)}
    <div class="metrics-grid">${metricCard(`IG${state.meta.institutionGeneration}`,'Institutional generation')}${metricCard(state.chronicle.length,'Chronicle records')}${metricCard(state.negativeCapability.length,'Failure records')}${metricCard(`${round(state.recursive.institutionalMemory)}%`,'Institutional memory')}${metricCard(`${round(state.recursive.formationCostMultiplier,2)}×`,'Formation-cost multiplier')}</div>
    <div class="grid-2">
      <div class="card"><h2>Succession Ledger</h2><div class="table-wrap"><table><tbody>${[
        ['Identity',candidate?`${candidate.name} · ${state.proof.manifestHash}`:'No frozen successor'],['Mission',state.mission.name],['Origin',state.seize.decision],['Evidence',result?`${result.proofCases||state.gym.proofCases} proof + ${result.transferCases||state.gym.transferCases} transfer cases`:'No protected proof'],['Authority',admission.status==='simulated_admitted'?`${admission.level} · expires ${admission.expiry}`:'No active simulated admission'],['Economics',`Proof budget ${money(state.mission.proofBudget)} · formation multiplier ${round(state.recursive.formationCostMultiplier,2)}×`],['Risk',`${state.negativeCapability.length} retained failure records`],['Lifecycle',state.requalification.result?.disposition||'Awaiting first requalification']
      ].map(([key,value])=>`<tr><th>${esc(key)}</th><td>${esc(value)}</td></tr>`).join('')}</tbody></table></div></div>
      <div class="card"><h2>Alpha persistence condition</h2><div class="callout dark"><strong>Verified learning + reproof velocity must exceed commoditization + imitation + drift + decay.</strong></div><p>Recursive improvement is valid only when later successor generations become measurably better, cheaper or faster on fresh work without weakening proof or governance.</p><div class="bar-row"><b>Learning capacity</b>${progressBar(clamp(state.recursive.institutionalMemory*3+state.recursive.generation*2),'green')}<span>${round(clamp(state.recursive.institutionalMemory*3+state.recursive.generation*2))}</span></div><div class="bar-row"><b>Proof discipline</b>${progressBar(result?.proofCoverage||0,'green')}<span>${round(result?.proofCoverage||0)}</span></div><div class="bar-row"><b>Alpha compression risk</b>${progressBar(clamp(25+state.recursive.generation*1.5-state.meta.institutionGeneration*4),'red')}<span>${round(clamp(25+state.recursive.generation*1.5-state.meta.institutionGeneration*4))}</span></div></div>
    </div>
    <div class="grid-2" style="margin-top:16px"><div class="card"><h2>Chronicle timeline</h2><div class="timeline">${state.chronicle.length?state.chronicle.slice(0,28).map(item=>`<div class="timeline-item"><b>${esc(item.title)}</b><small>${new Date(item.at).toLocaleString()} · ${esc(item.type)} · IG${item.generation}</small>${item.detail?`<p>${esc(item.detail)}</p>`:''}</div>`).join(''):'<div class="callout">No Chronicle records yet.</div>'}</div></div><div class="card"><h2>Negative Capability Graph</h2><p class="muted">A failed architecture or exploitable verifier can be more valuable than a hidden failure.</p><div class="negative-graph">${state.negativeCapability.length?state.negativeCapability.slice(0,50).map(item=>`<span class="negative-node" title="${esc(item.detail)}">${esc(item.type)} · ${esc(item.detail.slice(0,75))}</span>`).join(''):'<span class="muted">No negative evidence recorded.</span>'}</div></div></div>`;
}

function renderRequalify() {
  const r=state.requalification.result;
  const actions = `<button class="button" data-action="run-requalification" ${state.proof.frozenCandidateId?'':'disabled'}>Run material-change fresh requalification</button><button class="button violet" data-action="evolve" data-generations="5">Generate the next successor frontier</button>`;
  sectionElement('requalify').innerHTML = `${pageHead('11 · Requalification and Next Succession','No promotion on inherited evidence','A material change to mission, model, data, tool, workflow, verifier, provider, environment or Authority Envelope creates a new release. It must be re-underwritten and re-proven.',actions)}
    <div class="grid-2"><div class="card"><h2>Declare the material change</h2><div class="checklist">${[
      ['model','Model, prompts or reasoning substrate'],['data','Data, examples or mission distribution'],['provider','Provider, runtime or infrastructure'],['verifier','Verifier, scorer or protected cases'],['workflow','Workflow, tools, routing or integration'],['authority','Authority expansion or new external effects'],['environment','Operating environment or jurisdiction']
    ].map(([key,label])=>`<div class="check-row"><input type="checkbox" data-requalification-change="${key}" ${state.requalification.changes.includes(key)?'checked':''}><label>${esc(label)}</label></div>`).join('')}</div><div class="field"><label>Severity</label><select id="requalSeverity"><option value="low" ${state.requalification.severity==='low'?'selected':''}>Low</option><option value="medium" ${!state.requalification.severity||state.requalification.severity==='medium'?'selected':''}>Medium</option><option value="high" ${state.requalification.severity==='high'?'selected':''}>High</option></select></div><div class="field"><label>Reason and expected effect</label><textarea id="requalReason">${esc(state.requalification.reason||'')}</textarea></div></div>
    <div class="card ${r?(r.result.pass?'success':'danger'):'highlight'}"><div class="page-kicker">Latest disposition</div><h2>${r?esc(r.disposition.replaceAll('_',' ')):'No requalification run'}</h2>${r?`<p>${esc(r.changes.join(', '))} · ${esc(r.severity)} · ${new Date(r.runAt).toLocaleString()}</p><div class="metrics-grid" style="grid-template-columns:repeat(2,1fr)">${metricCard(round(r.result.paired.lcb,2),'Requalification LCB')}${metricCard(`${round(r.result.proofCoverage)}%`,'Proof coverage')}${metricCard(r.result.criticalErrors,'Critical errors')}${metricCard(r.result.unauthorizedActions,'Unauthorized actions')}</div>${gatesHTML(r.result.gates)}`:'<p>Run requalification after an admitted or frozen candidate changes.</p>'}</div></div>
    <div class="card" style="margin-top:16px"><h2>Permanent renewal loop</h2><div class="flow">${['Chronicle','Detect material change','Re-underwrite','Generate descendants','Freeze challenger','Fresh proof','Renew / repair / revoke','Next Chronicle state'].map((name,index)=>`${index?'<div class="flow-arrow">→</div>':''}<div class="flow-node"><b>${esc(name)}</b><small>${index===0?'Admitted evidence and failures':index===5?'Protected representative work':'Bounded institutional decision'}</small></div>`).join('')}</div></div>`;
}

function boardMemo() {
  const champion=state.formation.results[0],proof=state.proof.result,candidate=candidateById(state.proof.frozenCandidateId);
  return `# GoalOS UVSI2 — Board Decision Memo\n\n**Project:** ${state.meta.projectId}\n**Mission:** ${state.mission.name}\n**Prepared:** ${new Date().toISOString()}\n**Author:** ${APP.author}, ${APP.role}\n\n## Decision state\n${statusText()}\n\n## Mission\n${state.mission.objective}\n\n## Incumbent\n${state.mission.incumbent}\n\n## SEIZE decision\n${state.seize.decision}\n\n**Next decisive evidence:** ${state.seize.nextEvidence}\n\n## Formation champion\n${champion?`${champion.candidateName}; utility ${round(champion.metrics.utility)}; burden ${round(champion.burden)}.`:'No matched formation result.'}\n\n## Frozen challenger\n${candidate?`${candidate.name}; manifest ${state.proof.manifestHash}.`:'No frozen challenger.'}\n\n## Fresh proof\n${proof?`${proof.status}; LCB ${round(proof.paired.lcb,2)}; proof coverage ${round(proof.proofCoverage)}%; critical errors ${proof.criticalErrors}; unauthorized actions ${proof.unauthorizedActions}.`:'Not run.'}\n\n## Authority\nLevel ${state.authority.level}; owner ${state.authority.owner}; expiry ${state.authority.expiry}.\n\nPermitted: ${state.authority.permitted.join('; ')}.\n\nProhibited: ${state.authority.prohibited.join('; ')}.\n\n## Recursive improvement\nGeneration G${state.recursive.generation}; institutional generation IG${state.meta.institutionGeneration}; archive ${state.recursive.archive.length}; negative capability records ${state.negativeCapability.length}.\n\n## Claim boundary\nThis deterministic local demonstration does not establish real-world Specialist ASI, Mission Alpha or organizational authority. Protected representative evidence, independent review and accountable admission remain required.\n`;
}

/* ---------- Export and preservation ---------- */
function environmentSpec() {
  return {schema:'GoalOS.EnvironmentSpec.v7',projectId:state.meta.projectId,missionId:safeName(state.mission.name),version:state.gym.version,mission:deepClone(state.mission),state:state.gym.spec.state,observations:state.gym.spec.observations,actions:state.gym.spec.actions,rewards:state.gym.spec.rewards,constraints:state.gym.spec.constraints,scenarioDistribution:state.gym.distribution,seeds:{formation:state.gym.formationSeed,proof:state.gym.proofSeed,transfer:state.gym.transferSeed},generatedAt:nowISO()};
}
function candidateManifest() {
  const candidate=candidateById(state.proof.frozenCandidateId);
  return candidate?{schema:'GoalOS.CandidateSuccessorManifest.v7',projectId:state.meta.projectId,candidate:deepClone(candidate),manifestHash:state.proof.manifestHash,frozenAt:state.proof.frozenAt,missionHash:hashObject(state.mission),environmentVersion:state.gym.version,externalAuthorityCreated:'NONE'}:{schema:'GoalOS.CandidateSuccessorManifest.v7',status:'NO_FROZEN_CANDIDATE'};
}
function proofReportHTML() {
  const result=state.proof.result;
  return `<!doctype html><html><head><meta charset="utf-8"><title>GoalOS Fresh-Proof Report</title><style>body{font:15px system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#10213a}h1,h2{font-family:Georgia,serif;color:#07172d}.box{border:1px solid #c99a2d;border-radius:12px;padding:16px;margin:14px 0}.pass{background:#edf9f3}.fail{background:#fff1ef}table{width:100%;border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px;text-align:left}</style></head><body><h1>GoalOS Fresh-Successor Gate Report</h1><p><strong>Project:</strong> ${esc(state.meta.projectId)}<br><strong>Mission:</strong> ${esc(state.mission.name)}<br><strong>Generated:</strong> ${esc(nowISO())}</p>${result?`<div class="box ${result.pass?'pass':'fail'}"><h2>${esc(result.status)}</h2><p><strong>Candidate:</strong> ${esc(result.candidateName)}<br><strong>Manifest:</strong> ${esc(result.manifestHash)}<br><strong>Matched mean:</strong> ${round(result.paired.mean,2)}<br><strong>95% lower bound:</strong> ${round(result.paired.lcb,2)}<br><strong>Proof coverage:</strong> ${round(result.proofCoverage)}%</p></div><h2>Specialist ASI dimensions</h2><table>${Object.entries(result.dimensions).map(([k,v])=>`<tr><th>${esc(k)}</th><td>${round(v)}</td></tr>`).join('')}</table><h2>Hard gates</h2><table><tr><th>Gate</th><th>Result</th><th>Evidence</th></tr>${result.gates.map(g=>`<tr><td>${esc(g.name)}</td><td>${g.pass?'PASS':'FAIL'}</td><td>${esc(g.detail)}</td></tr>`).join('')}</table><div class="box fail"><strong>Claim boundary:</strong> ${esc(result.claimBoundary)}</div>`:'<div class="box fail">No fresh-proof result exists.</div>'}<h2>Controlling rule</h2><p>Capability may justify candidacy. Fresh proof may establish mission dominance. Only accountable admission may grant authority. Authority remains scoped, versioned, monitored, expiring and revocable.</p></body></html>`;
}
function recursiveReport() {
  return {schema:'GoalOS.VerifiedRecursiveInstitutionalImprovement.v7',projectId:state.meta.projectId,generation:state.recursive.generation,institutionGeneration:state.meta.institutionGeneration,settings:state.recursive.settings,history:state.recursive.history,archive:state.recursive.archive,lineage:state.recursive.lineage,console:state.recursive.console,negativeCapability:state.negativeCapability,institutionalMemory:state.recursive.institutionalMemory,formationCostMultiplier:state.recursive.formationCostMultiplier,claimBoundary:'Descendants inherit neither proof nor authority. Promotion requires a frozen release, protected fresh work, independent validation and accountable admission.'};
}
function formationCSV() {
  const headers=['rank','candidate_id','candidate','generation','utility','quality','value','evidence','reliability','sovereignty','governance','transfer','speed','cost_score','human_score','burden','critical_errors','unauthorized_actions'];
  const rows=state.formation.results.map((r,index)=>[index+1,r.candidateId,r.candidateName,r.generation,r.metrics.utility,r.metrics.quality,r.metrics.value,r.metrics.evidence,r.metrics.reliability,r.metrics.sovereignty,r.metrics.governance,r.metrics.transfer,r.metrics.speed,r.metrics.costScore,r.metrics.humanScore,r.burden,r.criticalErrors,r.unauthorizedActions]);
  const csvCell=value=>`"${String(value??'').replaceAll('"','""')}"`;
  return [headers,...rows].map(row=>row.map(csvCell).join(',')).join('\n');
}

const crcTable=(()=>{const table=new Uint32Array(256);for(let n=0;n<256;n++){let c=n;for(let k=0;k<8;k++)c=(c&1)?0xedb88320^(c>>>1):c>>>1;table[n]=c>>>0}return table})();
function crc32(bytes){let c=0xffffffff;for(const byte of bytes)c=crcTable[(c^byte)&255]^(c>>>8);return(c^0xffffffff)>>>0}
function u16(n){return new Uint8Array([n&255,(n>>>8)&255])}
function u32(n){return new Uint8Array([n&255,(n>>>8)&255,(n>>>16)&255,(n>>>24)&255])}
function concatU8(arrays){const length=arrays.reduce((total,array)=>total+array.length,0),out=new Uint8Array(length);let offset=0;for(const array of arrays){out.set(array,offset);offset+=array.length}return out}
function dosTimeDate(){const date=new Date(),time=(date.getHours()<<11)|(date.getMinutes()<<5)|(date.getSeconds()>>1),day=((date.getFullYear()-1980)<<9)|((date.getMonth()+1)<<5)|date.getDate();return{time,date:day}}
function makeZip(files){const encoder=new TextEncoder(),locals=[],centrals=[];let offset=0;const dt=dosTimeDate();for(const file of files){const name=encoder.encode(file.name),data=file.data instanceof Uint8Array?file.data:encoder.encode(String(file.data)),crc=crc32(data);const local=concatU8([u32(0x04034b50),u16(20),u16(0),u16(0),u16(dt.time),u16(dt.date),u32(crc),u32(data.length),u32(data.length),u16(name.length),u16(0),name,data]);locals.push(local);const central=concatU8([u32(0x02014b50),u16(20),u16(20),u16(0),u16(0),u16(dt.time),u16(dt.date),u32(crc),u32(data.length),u32(data.length),u16(name.length),u16(0),u16(0),u16(0),u16(0),u32(0),u32(offset),name]);centrals.push(central);offset+=local.length}const centralData=concatU8(centrals),end=concatU8([u32(0x06054b50),u16(0),u16(0),u16(files.length),u16(files.length),u32(centralData.length),u32(offset),u16(0)]);return new Blob([...locals,centralData,end],{type:'application/zip'})}
function missionPackFiles() {
  const name=safeName(state.mission.name);
  return [
    {name:'README.txt',data:`GoalOS UVSI2 — Executable Verified Succession Institution v${APP.version}\nProject: ${state.meta.projectId}\nMission: ${state.mission.name}\nStatus: ${statusText()}\n\nThis package is a deterministic local demonstration and reference implementation. It creates no real-world Specialist ASI status, Mission Alpha, legal authority or production release.\n`},
    {name:`${name}_Board_Decision_Memo.md`,data:boardMemo()},
    {name:`${name}_Project.json`,data:JSON.stringify(state,null,2)},
    {name:`${name}_EnvironmentSpec.json`,data:JSON.stringify(environmentSpec(),null,2)},
    {name:`${name}_CandidateManifest.json`,data:JSON.stringify(candidateManifest(),null,2)},
    {name:`${name}_Fresh_Proof_Report.html`,data:proofReportHTML()},
    {name:`${name}_Authority_Envelope.json`,data:JSON.stringify({schema:'GoalOS.AuthorityEnvelope.v7',...state.authority},null,2)},
    {name:`${name}_Chronicle.json`,data:JSON.stringify({schema:'GoalOS.Chronicle.v7',records:state.chronicle,negativeCapability:state.negativeCapability,successorBook:state.successorBook,requalification:state.requalification},null,2)},
    {name:`${name}_Recursive_Improvement.json`,data:JSON.stringify(recursiveReport(),null,2)},
    {name:`${name}_Bounded_AGI_Jobs.json`,data:JSON.stringify({schema:'GoalOS.BoundedAGIJobPortfolio.v7',jobs:state.jobs},null,2)},
    {name:`${name}_Formation_Results.csv`,data:formationCSV()}
  ];
}
function exportMissionPack() { const name=safeName(state.mission.name); downloadBlob(`${name}_GoalOS_Mission_Pack_v7.zip`,makeZip(missionPackFiles())); record('MISSION_PACK_EXPORTED','Complete Mission Pack exported',name); toast('Mission Pack ZIP downloaded'); }

function renderExport() {
  const items=[
    ['{}','Complete project JSON','Editable state, candidates, jobs, proof, authority and Chronicle.','export-project'],
    ['G','EnvironmentSpec','Executable mission state, observations, actions, rewards and constraints.','export-environment'],
    ['S','Candidate manifest','Exact frozen release, environment and deterministic manifest hash.','export-candidate'],
    ['✓','Fresh-Proof report','Matched evidence, confidence bound, dimensions and hard gates.','export-proof'],
    ['A','Authority Envelope','Permitted and prohibited actions, owner, expiry and admission state.','export-authority'],
    ['C','Chronicle','Evidence, failure memory, successor book and requalification.','export-chronicle'],
    ['↻','Recursive improvement','Lineage, generations, quality-diversity archive and negative capability.','export-recursive'],
    ['CSV','Formation results','Complete comparative candidate scorecard.','export-csv'],
    ['ZIP','Complete Mission Pack','All governing records and the board memo in one local ZIP.','export-pack'],
    ['⇧','Import project','Continue from a previous Version 2 project JSON.','import-project'],
    ['↺','Reset application','Erase local state and restore a built-in mission.','reset-all']
  ];
  sectionElement('export').innerHTML = `${pageHead('12 · Export, Preservation and Deployment','Download the complete Mission Pack','All exports are generated locally. The authoritative paper remains the controlling source for definitions, claims and governance rules.','<button class="button" data-action="export-pack">Download complete Mission Pack ZIP</button>')}
    <div class="export-grid">${items.map(([icon,name,description,action])=>`<div class="export-card"><div class="export-icon">${icon}</div><h3>${esc(name)}</h3><p>${esc(description)}</p><button class="button small ${action==='reset-all'?'danger':'secondary'}" data-action="${action}">${action==='import-project'?'Choose JSON':action==='reset-all'?'Reset':'Download'}</button></div>`).join('')}</div>
    <div class="grid-2" style="margin-top:16px"><div class="card navy"><h2>Source of truth</h2><p>The exact paper is included in the downloadable release. Image-generated art and this browser demo are explanatory. The typeset paper, release manifest and checksums govern whenever terminology or microcopy differs.</p><a class="button secondary" href="research/GoalOS_Navigator_SEIZE_Gym_Successor_Omega_v6_A4.pdf" target="_blank" rel="noopener">Open the included paper</a></div><div class="card"><h2>Official identity</h2><p><strong>${esc(APP.author)}</strong><br>${esc(APP.role)}</p><p><a href="${APP.website}" target="_blank" rel="noopener">${APP.website}</a><br><a href="mailto:${APP.email}">${APP.email}</a></p><p class="fine">No other similarly named domain or organization should be inferred to be an official source.</p></div></div>
    <div class="footer-note">© 2026 Vincent Boucher. All rights reserved. · Version ${APP.version} · ${APP.releaseDate} · Montreal, Quebec, Canada.</div>`;
}

/* ---------- Interaction and application shell ---------- */
function saveMissionFromDOM() {
  const before=hashObject(state.mission);
  const value=id=>$(id)?.value ?? '';
  state.mission.name=value('#missionName').trim()||state.mission.name;
  state.mission.beneficiary=value('#missionBeneficiary').trim();
  state.mission.objective=value('#missionObjective').trim();
  state.mission.incumbent=value('#missionIncumbent').trim();
  state.mission.alternatives=list(value('#missionAlternatives'));
  state.mission.scope=value('#missionScope').trim();
  state.mission.criticalFailures=list(value('#missionFailures'));
  state.mission.constraints=list(value('#missionConstraints'));
  state.mission.proofBudget=Math.max(0,Number(value('#missionProofBudget'))||0);
  state.mission.authorityCeiling=value('#missionAuthorityCeiling')||'A1';
  const changed=before!==hashObject(state.mission);
  if (changed && state.mission.constitutionFrozen) {
    state.mission.constitutionFrozen=false; state.mission.frozenAt=null; state.seize.frozen=false; state.proof.result=null; state.proof.frozenCandidateId=null; state.authority.admission.status='not_admitted';
    addNegative('Constitution changed','Mission Constitution changed after freeze; inherited proof and admission invalidated.');
  }
  record('MISSION_CONSTITUTION_SAVED','Mission Constitution saved',`${state.mission.name}; Gymmability ${round(mean(state.mission.gymmability),0)}%.`);
  saveState(); renderAll(); toast(changed?'Mission Constitution saved':'Mission Constitution unchanged');
}
function saveSEIZEFromDOM() {
  const map={seizeMissionValue:'missionValue',seizeProbability:'probability',seizeProofCost:'proofCost',seizeFormationCost:'formationCost',seizeTailRisk:'tailRisk',seizeOptionValue:'optionValue',seizeDuration:'durationMonths',seizeHalfLife:'halfLifeMonths'};
  Object.entries(map).forEach(([id,key])=>{const element=$(`#${id}`); if(element) state.seize[key]=Number(element.value)||0;});
}
function saveRSISettingsFromDOM() {
  const map={rsiChildren:'childrenPerGeneration',rsiMutation:'mutationScale',rsiNovelty:'noveltyWeight',rsiBurden:'burdenWeight'};
  Object.entries(map).forEach(([id,key])=>{const element=$(`#${id}`);if(element) state.recursive.settings[key]=Number(element.value)||0;});
  saveState();
}
function loadTemplate(key) {
  const target=TEMPLATES[key]?key:'custom';
  if (state.meta.updatedAt && !confirm(`Load ${TEMPLATES[target].name}? This replaces the current local project state.`)) return;
  resetState(target);
  record('MISSION_TEMPLATE_LOADED','Mission template loaded',TEMPLATES[target].name);
  navTo('mission');
}
function addCandidate() {
  const parent=candidateById(state.selectedCandidateId)||candidateById('sovereign_hybrid');
  const candidate=deepClone(parent||defaultCandidates(state.mission.template)[2]);
  candidate.id=uuid('custom'); candidate.name='Custom Successor Architecture'; candidate.parentId=parent?.id||null; candidate.generation=state.recursive.generation; candidate.createdAt=nowISO(); candidate.kind='custom'; candidate.notes='User-defined complete successor architecture. No inherited proof or authority.'; candidate.selected=true;
  state.candidates.push(candidate); state.selectedCandidateId=candidate.id; if(parent)state.recursive.lineage.push({parentId:parent.id,childId:candidate.id,generation:candidate.generation,source:'custom',at:nowISO()});
  saveState(); renderAll(); openCandidateEditor(candidate.id);
}
function openCandidateEditor(id) {
  const candidate=candidateById(id); if(!candidate)return;
  $('#modal').dataset.candidateId=id;
  modal('Edit complete successor architecture',`<div class="form-grid"><div class="field full"><label>Name</label><input id="modalCandidateName" value="${esc(candidate.name)}"></div><div class="field full"><label>Architecture thesis</label><textarea id="modalCandidateNotes">${esc(candidate.notes)}</textarea></div>${Object.entries(PARAM_LABELS).map(([key,label])=>`<div class="field"><label>${esc(label)} · <span id="value_${key}">${candidate.params[key]}</span></label><input type="range" min="0" max="100" id="param_${key}" value="${candidate.params[key]}" oninput="document.getElementById('value_${key}').textContent=this.value"></div>`).join('')}<div class="field full"><h3>Institutional burden</h3></div>${Object.entries(candidate.burden).map(([key,value])=>`<div class="field"><label>${esc(key[0].toUpperCase()+key.slice(1))}</label><input type="number" min="0" max="100" id="burden_${key}" value="${value}"></div>`).join('')}</div><div class="button-row" style="margin-top:16px"><button class="button" data-action="save-candidate-modal">Save architecture</button><button class="button secondary" data-action="close-modal">Cancel</button></div><div class="callout red"><strong>Proof invalidation</strong><br>Editing an architecture creates a material change. Any existing fresh proof and admission for this release are invalidated.</div>`);
}
function saveCandidateModal() {
  const id=$('#modal').dataset.candidateId,candidate=candidateById(id);if(!candidate)return;
  candidate.name=$('#modalCandidateName').value.trim()||candidate.name; candidate.notes=$('#modalCandidateNotes').value.trim();
  Object.keys(candidate.params).forEach(key=>{const e=$(`#param_${key}`);if(e)candidate.params[key]=clamp(e.value);});
  Object.keys(candidate.burden).forEach(key=>{const e=$(`#burden_${key}`);if(e)candidate.burden[key]=clamp(e.value);});
  candidate.updatedAt=nowISO();
  if(state.proof.frozenCandidateId===id){state.proof.frozenCandidateId=null;state.proof.result=null;state.proof.manifestHash=null;state.authority.admission.status='not_admitted';addNegative('Architecture changed',`${candidate.name} changed after freeze; inherited proof invalidated.`,id);}
  record('CANDIDATE_ARCHITECTURE_UPDATED','Successor architecture updated',candidate.name); closeModal(); saveState(); renderAll(); toast('Architecture saved; fresh proof required');
}
function inspectResult(id) {
  const result=resultForCandidate(id),candidate=candidateById(id);if(!result||!candidate)return;
  const worst=[...result.episodes].sort((a,b)=>a.utility-b.utility).slice(0,8);
  modal(`${candidate.name} · Formation evidence`,`<div class="metrics-grid" style="grid-template-columns:repeat(3,1fr)">${metricCard(round(result.metrics.utility),'Utility')}${metricCard(round(result.burden),'Burden')}${metricCard(result.criticalErrors,'Critical errors')}${metricCard(result.unauthorizedActions,'Unauthorized actions')}${metricCard(`${round(result.acceptedRate)}%`,'Accepted episodes')}${metricCard(round(result.tailUtility),'5% tail utility')}</div><h3>Complete profile</h3>${metricBars(result.metrics)}<h3>Worst-case episodes</h3><div class="table-wrap"><table><thead><tr><th>Case</th><th>Distribution</th><th>Critical</th><th>Utility</th><th>Evidence</th><th>Reliability</th><th>Accepted</th></tr></thead><tbody>${worst.map(e=>`<tr class="${e.criticalError||e.unauthorized?'row-fail':''}"><td>${e.caseId}</td><td>${e.distribution}</td><td>${e.critical?'Yes':'No'}</td><td>${round(e.utility)}</td><td>${round(e.evidence)}</td><td>${round(e.reliability)}</td><td>${e.accepted?'Yes':'No'}</td></tr>`).join('')}</tbody></table></div>`);
}
function tour() {
  modal('Three-minute guided tour',`<ol><li><strong>Mission:</strong> freeze a narrow objective, incumbent, alternatives, failures, proof budget and authority ceiling.</li><li><strong>Manifold:</strong> compare complete successor institutions, not model brands.</li><li><strong>Gradient:</strong> estimate the bounded architecture change with the greatest marginal Alpha per unit of proof capital.</li><li><strong>SEIZE:</strong> underwrite whether to retain, repair, rent, build, partner, acquire, hedge, reserve or stop.</li><li><strong>Bounded AGI Jobs:</strong> compile the 21 work-and-proof contracts.</li><li><strong>Formation Gym:</strong> run matched cases and preserve a quality-diversity archive.</li><li><strong>Recursive Foundry:</strong> produce descendants, reject regressions and retain failures.</li><li><strong>Fresh Proof:</strong> freeze one challenger and purchase protected comparative evidence.</li><li><strong>Authority:</strong> rehearse refusal, escalation, degraded mode, expiry and rollback.</li><li><strong>Chronicle:</strong> admit only rights-cleared evidence and use it to improve the next succession cycle.</li><li><strong>Requalification:</strong> treat every material change as a new release.</li><li><strong>Export:</strong> download the complete Mission Pack.</li></ol><div class="callout red"><strong>Remember:</strong> simulation performance is candidacy—not real-world Specialist ASI, Mission Alpha or institutional authority.</div>`);
}
async function guidedRun() {
  if (!confirm('Run the complete local guided demonstration? It will reset the current project to the Portfolio Building Operations mission.')) return;
  state=makeState('building'); saveState(); renderAll();
  state.mission.constitutionFrozen=true;state.mission.frozenAt=nowISO();calculateSEIZE();state.seize.frozen=true;state.seize.frozenAt=nowISO();compileJobs();runPreparationJobs();
  runMatchedFormation(); finiteDifferenceGradient(state.formation.championId); createGradientChallenger(); runMatchedFormation();
  await runRecursiveFoundry(8);
  freezeCandidate(state.formation.championId); runFreshProof();
  if (!state.proof.result?.pass) { const sovereign=state.candidates.filter(c=>c.kind==='sovereign').sort((a,b)=>b.generation-a.generation)[0]; if(sovereign){freezeCandidate(sovereign.id);runFreshProof();} }
  state.authority.admission.checks.identity=Boolean(state.proof.manifestHash); state.authority.admission.checks.proof=Boolean(state.proof.result?.pass); state.authority.admission.checks.rights=true; state.authority.admission.checks.owner=true; state.authority.admission.checks.expiry=true; state.authority.admission.signedBy='Demonstration Principal'; state.authority.owner='Demonstration Principal'; state.authority.level=maximumEligibleAuthority();
  runAuthorityStress();
  if(state.authority.stress?.pass){state.authority.admission.checks.rollback=true;issueSimulatedAdmission();}
  navTo(state.authority.admission.status==='simulated_admitted'?'authority':'proof'); toast('Complete guided demonstration finished');
}


function ensureManualEpisode() {
  if (state.gym.manual?.case) return state.gym.manual;
  const seed = state.gym.formationSeed + Math.floor(Math.random()*100000);
  const item = generateCases('formation',1,seed)[0];
  state.gym.manual = {seed,case:item,trace:[],done:false,score:null,step:0};
  return state.gym.manual;
}
function manualDecision(caseItem) {
  if (caseItem.outage) return 'fallback';
  if (caseItem.critical || caseItem.hazard>.78 || caseItem.rightsTrap) return 'escalate';
  if (caseItem.evidenceGap>.58 || caseItem.deception) return 'request_evidence';
  if (caseItem.difficulty>.72) return 'inspect';
  return 'recommend';
}
function newManualEpisode() {
  state.gym.manual=null; ensureManualEpisode(); saveState(); renderFormation(); toast('New Mission Gym episode created');
}
function manualGymAction(action) {
  const manual=ensureManualEpisode(); if(manual.done)return;
  const correct=manualDecision(manual.case); manual.step+=1;
  const labels={inspect:'Inspect the state and tool evidence',request_evidence:'Purchase decisive evidence',recommend:'Recommend the bounded mission action',escalate:'Escalate to the accountable principal',fallback:'Enter degraded mode / incumbent fallback'};
  let accepted=action===correct;
  if(correct==='request_evidence' && action==='inspect') accepted=true;
  if(correct==='escalate' && action==='fallback' && manual.case.outage) accepted=true;
  manual.trace.unshift({at:nowISO(),action,detail:accepted?'The action matched the hidden state and constitutional constraints.':`The hidden state required ${labels[correct].toLowerCase()}.`,accepted});
  if(['recommend','escalate','fallback'].includes(action) || manual.step>=3 || accepted && correct==='request_evidence') {
    if(accepted && action==='request_evidence' && manual.step<3){manual.case.evidenceGap=clamp(manual.case.evidenceGap-.35,0,1);manual.case.deception=false;manual.trace.unshift({at:nowISO(),action:'evidence_returned',detail:'The decisive evidence reduced uncertainty. Choose the final bounded action.',accepted:true});}
    else {manual.done=true;manual.score=accepted?100:manual.case.critical?0:45;}
  }
  saveState(); renderFormation();
}
function manualGymHTML() {
  const manual=ensureManualEpisode(),item=manual.case;
  return `<div class="card highlight" style="margin-top:16px"><div class="page-head"><div><div class="page-kicker">Interactive Mission Gym</div><h1>Play one hidden-state episode</h1><p>Use the mission environment as an institutional flight simulator. The hidden state determines whether the action was mission-correct and authority-compliant.</p></div><div class="page-actions"><button class="button secondary" data-action="new-manual-episode">New episode</button></div></div><div class="grid-2"><div><div class="pill-row"><span class="pill">${esc(item.distribution)}</span><span class="pill">Difficulty ${round(item.difficulty*100)}%</span><span class="pill">${item.critical?'Critical':'Non-critical'}</span><span class="pill">Evidence gap ${round(item.evidenceGap*100)}%</span></div><h3 style="margin-top:12px">Observed mission state</h3><p>${item.outage?'A primary provider or tool is unavailable. ':''}${item.deception?'The evidence may contain a strategically misleading signal. ':''}${item.rightsTrap?'The requested path may violate a rights or authority boundary. ':''}The mission state is incomplete; choose the next bounded action.</p><div class="button-row">${[['inspect','Inspect evidence'],['request_evidence','Request decisive evidence'],['recommend','Recommend action'],['escalate','Escalate'],['fallback','Fallback']].map(([id,label])=>`<button class="button small ${['escalate','fallback'].includes(id)?'secondary':''}" data-action="manual-gym-action" data-manual-action="${id}" ${manual.done?'disabled':''}>${label}</button>`).join('')}</div>${manual.done?`<div class="callout ${manual.score===100?'green':'red'}"><strong>Episode ${manual.score===100?'accepted':'rejected'} · score ${manual.score}</strong><br>The hidden mission state required: ${esc({inspect:'inspect the state',request_evidence:'purchase decisive evidence',recommend:'recommend the bounded action',escalate:'escalate to the principal',fallback:'enter fallback'}[manualDecision(item)])}.</div>`:''}</div><div><h3>Proof receipt</h3><div class="timeline">${manual.trace.length?manual.trace.map(trace=>`<div class="timeline-item"><b>${esc(trace.action.replaceAll('_',' '))}</b><small>${new Date(trace.at).toLocaleTimeString()} · ${trace.accepted?'accepted':'rejected'}</small><p>${esc(trace.detail)}</p></div>`).join(''):'<div class="callout">No action taken yet.</div>'}</div></div></div></div>`;
}

function updateShell() {
  if (!state) return;
  $('#projectChip').textContent=state.meta.projectId;
  const chip=$('#systemStatusChip'); chip.textContent=statusText(); chip.className=`status-chip ${statusTone()}`;
  $('#editionText').textContent=`${APP.edition} · v${APP.version}`;
  $('#languageButton').textContent=state.meta.lang==='en'?'FR':'EN';
  $('#guideButton').textContent=tr('guide'); $('#resetButton').textContent=tr('reset');
  $$('[data-i18n]').forEach(element=>{element.textContent=tr(element.dataset.i18n)});
}
function navTo(id) {
  state.meta.currentSection=id;
  $$('.view').forEach(view=>view.classList.toggle('active',view.id===`section-${id}`));
  $$('.nav-item').forEach(item=>item.classList.toggle('active',item.dataset.section===id));
  $('#sidebar').classList.remove('open');
  window.scrollTo({top:0,behavior:'smooth'});
  saveState();
}
function renderAll() {
  updateJobsFromProofAndAuthority();
  renderWelcome();renderMission();renderManifold();renderGradient();renderSeize();renderJobs();renderFormation();renderRecursive();renderProof();renderAuthority();renderChronicle();renderRequalify();renderExport();updateShell();
  const current=state.meta.currentSection||'welcome';
  $$('.view').forEach(view=>view.classList.toggle('active',view.id===`section-${current}`));
  $$('.nav-item').forEach(item=>item.classList.toggle('active',item.dataset.section===current));
}

function handleAction(action,target) {
  if(action==='tour')tour();
  if(action==='guided-run')guidedRun();
  if(action==='load-template')loadTemplate(target.dataset.template);
  if(action==='save-mission')saveMissionFromDOM();
  if(action==='add-candidate')addCandidate();
  if(action==='select-candidate'){state.selectedCandidateId=target.dataset.candidateId;state.gradient.candidateId=target.dataset.candidateId;saveState();renderAll();}
  if(action==='toggle-candidate'){const c=candidateById(target.dataset.candidateId);if(c){c.selected=target.checked;state.formation.results=[];state.proof.result=null;saveState();renderAll();}}
  if(action==='edit-candidate')openCandidateEditor(target.dataset.candidateId);
  if(action==='delete-candidate'){const id=target.dataset.candidateId;if(confirm('Delete this generated candidate?')){state.candidates=state.candidates.filter(c=>c.id!==id);state.recursive.lineage=state.recursive.lineage.filter(x=>x.childId!==id&&x.parentId!==id);if(state.selectedCandidateId===id)state.selectedCandidateId='sovereign_hybrid';saveState();renderAll();}}
  if(action==='save-candidate-modal')saveCandidateModal();
  if(action==='close-modal')closeModal();
  if(action==='calculate-gradient'){finiteDifferenceGradient();renderAll();toast('Mission Advantage Gradient calculated');}
  if(action==='create-gradient-challenger')createGradientChallenger();
  if(action==='calculate-seize'){saveSEIZEFromDOM();calculateSEIZE();}
  if(action==='freeze-seize'){saveSEIZEFromDOM();freezeSEIZE();}
  if(action==='compile-jobs')compileJobs();
  if(action==='run-preparation-jobs')runPreparationJobs();
  if(action==='run-formation')runMatchedFormation();
  if(action==='new-manual-episode')newManualEpisode();
  if(action==='manual-gym-action')manualGymAction(target.dataset.manualAction);
  if(action==='inspect-result')inspectResult(target.dataset.candidateId);
  if(action==='evolve'){saveRSISettingsFromDOM();runRecursiveFoundry(Number(target.dataset.generations)||1);}
  if(action==='apply-chronicle-learning')applyChronicleLearning();
  if(action==='freeze-champion')freezeCandidate(state.formation.championId||state.selectedCandidateId);
  if(action==='run-proof'){runFreshProof();state.authority.admission.checks.identity=Boolean(state.proof.manifestHash);state.authority.admission.checks.proof=Boolean(state.proof.result?.pass);saveState();renderAll();}
  if(action==='set-authority'){const level=target.dataset.level,max=maximumEligibleAuthority();if(authorityRank(level)>authorityRank(max)||authorityRank(level)>authorityRank(state.mission.authorityCeiling)){toast(`Authority ${level} is not justified by current proof or mission ceiling`);return;}state.authority.level=level;saveState();renderAll();}
  if(action==='save-authority')saveAuthorityFromDOM();
  if(action==='run-authority-stress')runAuthorityStress();
  if(action==='issue-admission')issueSimulatedAdmission();
  if(action==='revoke-admission')revokeAdmission();
  if(action==='run-requalification'){const severity=$('#requalSeverity');const reason=$('#requalReason');if(severity)state.requalification.severity=severity.value;if(reason)state.requalification.reason=reason.value;runRequalification();}
  if(action==='export-project')downloadText(`${safeName(state.mission.name)}_GoalOS_Project.json`,JSON.stringify(state,null,2),'application/json');
  if(action==='export-environment')downloadText(`${safeName(state.mission.name)}_EnvironmentSpec.json`,JSON.stringify(environmentSpec(),null,2),'application/json');
  if(action==='export-candidate')downloadText(`${safeName(state.mission.name)}_CandidateManifest.json`,JSON.stringify(candidateManifest(),null,2),'application/json');
  if(action==='export-proof')downloadText(`${safeName(state.mission.name)}_Fresh_Proof_Report.html`,proofReportHTML(),'text/html');
  if(action==='export-authority')downloadText(`${safeName(state.mission.name)}_Authority_Envelope.json`,JSON.stringify(state.authority,null,2),'application/json');
  if(action==='export-chronicle')downloadText(`${safeName(state.mission.name)}_Chronicle.json`,JSON.stringify({chronicle:state.chronicle,negativeCapability:state.negativeCapability,successorBook:state.successorBook,requalification:state.requalification},null,2),'application/json');
  if(action==='export-recursive')downloadText(`${safeName(state.mission.name)}_Recursive_Improvement.json`,JSON.stringify(recursiveReport(),null,2),'application/json');
  if(action==='export-csv')downloadText(`${safeName(state.mission.name)}_Formation_Results.csv`,formationCSV(),'text/csv');
  if(action==='export-pack')exportMissionPack();
  if(action==='import-project')$('#importFile').click();
  if(action==='reset-all'){if(confirm('Erase the local project and start again?'))resetState('building');}
}
function handleClick(event) {
  const nav=event.target.closest('[data-section]');if(nav){navTo(nav.dataset.section);return;}
  const link=event.target.closest('[data-section-link]');if(link){navTo(link.dataset.sectionLink);return;}
  const action=event.target.closest('[data-action]');if(action){handleAction(action.dataset.action,action);}
}
function handleChange(event) {
  const target=event.target;
  if(target.matches('[data-gym-index]')){const index=Number(target.dataset.gymIndex);state.mission.gymmability[index]=Number(target.value);saveState();renderMission();}
  if(target.matches('[data-admission-check]')){state.authority.admission.checks[target.dataset.admissionCheck]=target.checked;saveState();renderAuthority();}
  if(target.matches('[data-requalification-change]')){const key=target.dataset.requalificationChange;const set=new Set(state.requalification.changes);target.checked?set.add(key):set.delete(key);state.requalification.changes=[...set];saveState();}
  if(target.id==='admissionSigner'){state.authority.admission.signedBy=target.value;saveState();renderAuthority();}
}
function importProject(file) {
  const reader=new FileReader();reader.onload=()=>{try{const parsed=JSON.parse(reader.result);state=migrateState(parsed);saveState();renderAll();toast('Project imported');}catch(error){toast(`Import failed: ${error.message}`)}};reader.readAsText(file);
}

function init() {
  document.addEventListener('click',handleClick);
  document.addEventListener('change',handleChange);
  $('#modalClose').addEventListener('click',closeModal);
  $('#modal').addEventListener('click',event=>{if(event.target===$('#modal'))closeModal();});
  $('#menuButton').addEventListener('click',()=>$('#sidebar').classList.toggle('open'));
  $('#languageButton').addEventListener('click',()=>{state.meta.lang=state.meta.lang==='en'?'fr':'en';saveState();renderAll();});
  $('#guideButton').addEventListener('click',tour);
  $('#resetButton').addEventListener('click',()=>{if(confirm('Reset the complete local project?'))resetState('building');});
  $('#importFile').addEventListener('change',event=>{if(event.target.files?.[0])importProject(event.target.files[0]);event.target.value='';});
  if(!state.chronicle.length)record('PROJECT_CREATED','GoalOS UVSI2 project created',`${state.mission.name} · ${APP.edition}.`);
  renderAll();
}

window.GoalOSDemo = {
  version: APP.version,
  getState: () => deepClone(state),
  replaceState: input => { state = migrateState(input); saveState(); renderAll(); return deepClone(state); },
  patchState: updater => { const draft = deepClone(state); const next = typeof updater === 'function' ? updater(draft) : {...draft,...updater}; state = migrateState(next || draft); saveState(); renderAll(); return deepClone(state); },
  record: (type,title,detail,data) => record(type,title,detail,data),
  navTo: id => navTo(id),
  renderAll: () => renderAll(),
  reset: templateKey => resetState(templateKey || 'building'),
  runFormation: runMatchedFormation,
  runRecursive: generations => runRecursiveFoundry(Number(generations)||1),
  freeze: candidateId => freezeCandidate(candidateId || state.formation.championId || state.selectedCandidateId),
  runFreshProof,
  runAuthorityStress,
  exportMissionPack
};

document.addEventListener('DOMContentLoaded',init);
})();
