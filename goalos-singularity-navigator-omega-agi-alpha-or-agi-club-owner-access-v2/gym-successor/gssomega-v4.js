(() => {
'use strict';

const RELEASE = {
  version: '5.0.0-UVSI1',
  id: 'v5.0.0-UVSI1',
  name: 'Unified Verified Succession Institution · Sovereign Mission Gym × Specialist ASI × Successor Ω',
  storageKey: 'goalos_gssomega_v4_0_0',
  legacyKeys: ['goalos_gssomega_v4_overlay'],
  date: '2026-08-10',
  claimBoundary: 'Browser-local deterministic institutional simulation and reference implementation. No real-world Specialist ASI, Mission Alpha, customer value, professional authority or production authority is established.'
};

const q = (s, r=document) => r.querySelector(s);
const qa = (s, r=document) => [...r.querySelectorAll(s)];
const esc = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const clamp = (value, min=0, max=100) => Math.max(min, Math.min(max, Number(value)||0));
const mean = a => a?.length ? a.reduce((x,y)=>x+Number(y||0),0)/a.length : 0;
const round = (v,d=1) => Number(v||0).toFixed(d);
const now = () => new Date().toISOString();
const deep = o => JSON.parse(JSON.stringify(o));
const authorityRank = level => ['A0','A1','A2','A3','A4'].indexOf(level);
const safe = value => String(value||'GoalOS').normalize('NFKD').replace(/[^a-zA-Z0-9_-]+/g,'_').replace(/^_+|_+$/g,'').slice(0,90)||'GoalOS';

function canonical(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  return `{${Object.keys(value).sort().map(k=>`${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`;
}
function hash(textOrObject) {
  const text = typeof textOrObject === 'string' ? textOrObject : canonical(textOrObject);
  let h=2166136261;
  for(let i=0;i<text.length;i++){h^=text.charCodeAt(i);h=Math.imul(h,16777619)}
  return (h>>>0).toString(16).padStart(8,'0').toUpperCase();
}
function core() { try { return window.GoalOSDemo?.getState?.() || {}; } catch { return {}; } }
function lang() { return core()?.meta?.lang === 'fr' ? 'fr' : 'en'; }
function t(en,fr) { return lang()==='fr' ? fr : en; }
function candidateById(c,id){ return c?.candidates?.find(x=>x.id===id); }
function formationResult(c,id){ return c?.formation?.results?.find(x=>x.candidateId===id); }
function proofCandidate(c){ return candidateById(c,c?.proof?.frozenCandidateId) || candidateById(c,c?.formation?.championId) || candidateById(c,c?.selectedCandidateId); }
function coreStatus(c){
  if(c?.authority?.admission?.status==='simulated_admitted') return 'MISSION-SOVEREIGN · DEMO ADMISSION';
  if(c?.proof?.result?.pass) return 'MISSION-DOMINANT SPECIALIST ASI · SIMULATION PASS';
  if(c?.proof?.frozenCandidateId) return 'FROZEN CHALLENGER';
  if(c?.formation?.results?.length) return 'FORMATION EVIDENCE AVAILABLE';
  return 'CANDIDATE INSTITUTION';
}

const FAMILY_BLUEPRINTS = [
  {id:'formation',name:'Formation Gym',nameFr:'Gym de formation',purpose:'Build, replay, repair and search candidate architectures.',purposeFr:'Construire, rejouer, réparer et rechercher des architectures candidates.',visibility:'Visible to formation roles; repeated access permitted.',visibilityFr:'Visible aux rôles de formation; accès répété permis.',custodian:'Formation Custodian',promotionWeight:0},
  {id:'adversarial',name:'Adversarial Gym',nameFr:'Gym adversarial',purpose:'Attack rewards, evidence, tools, verifier, authority and resilience.',purposeFr:'Attaquer les récompenses, la preuve, les outils, le vérificateur, l’autorité et la résilience.',visibility:'Visible or semi-blind; designed to expose failure.',visibilityFr:'Visible ou semi-aveugle; conçu pour exposer les échecs.',custodian:'Independent Red-Team Custodian',promotionWeight:0},
  {id:'transfer',name:'Transfer Gym',nameFr:'Gym de transfert',purpose:'Test new populations, regimes, structures and failure clusters.',purposeFr:'Tester de nouvelles populations, régimes, structures et grappes d’échec.',visibility:'Partially protected; prevents benchmark-only success.',visibilityFr:'Partiellement protégé; empêche le succès limité au banc d’essai.',custodian:'Transfer Evaluation Custodian',promotionWeight:.25},
  {id:'fresh',name:'Fresh-Proof Gym',nameFr:'Gym de preuve fraîche',purpose:'Evaluate one frozen challenger on unseen representative work.',purposeFr:'Évaluer un challenger gelé sur du travail inédit et représentatif.',visibility:'Hidden from candidate and formation roles until proof completion.',visibilityFr:'Caché au candidat et aux rôles de formation jusqu’à la fin de la preuve.',custodian:'Independent Proof Custodian',promotionWeight:1},
  {id:'canary',name:'Canary & Requalification Gym',nameFr:'Gym canari et requalification',purpose:'Connect shadow, sandbox, reversible canary and monitored operation.',purposeFr:'Relier le mode fantôme, le bac à sable, le canari réversible et l’exploitation surveillée.',visibility:'Production-connected, least privilege, fail closed.',visibilityFr:'Connecté à la production, moindre privilège, fermeture sûre.',custodian:'Operational Assurance Custodian',promotionWeight:.5}
];
const NESTED_GYMS = [
  {id:'mission',name:'Mission Gym',nameFr:'Mission Gym',question:'Can the complete architecture perform the mission?',questionFr:'L’architecture complète peut-elle accomplir la mission?'},
  {id:'institutional',name:'Institutional Gym',nameFr:'Gym institutionnel',question:'Can the human–machine institution preserve evidence, control and recovery?',questionFr:'L’institution humain–machine peut-elle préserver la preuve, le contrôle et la reprise?'},
  {id:'succession',name:'Succession Gym',nameFr:'Gym de succession',question:'Can the institution select, admit, impair, replace and renew successors legitimately?',questionFr:'L’institution peut-elle sélectionner, admettre, déprécier, remplacer et renouveler les successeurs légitimement?'}
];
const DECISIONS = [
  {id:'underwrite',name:'UNDERWRITE',nameFr:'SOUSCRIRE',owner:'SEIZE Investment Committee',ownerFr:'Comité d’investissement SEIZE',meaning:'Authorize the evidence programme and proof-capital budget.',meaningFr:'Autoriser le programme de preuve et le budget de capital-preuve.'},
  {id:'execute',name:'EXECUTE',nameFr:'EXÉCUTER',owner:'Mission Authority',ownerFr:'Autorité de mission',meaning:'Authorize one bounded job or episode to begin.',meaningFr:'Autoriser le démarrage d’une tâche ou d’un épisode borné.'},
  {id:'accept',name:'ACCEPT',nameFr:'ACCEPTER',owner:'Accountable Principal',ownerFr:'Principal responsable',meaning:'Accept a result for one exact purpose after independent evidence.',meaningFr:'Accepter un résultat pour une fin exacte après preuve indépendante.'},
  {id:'admit',name:'ADMIT',nameFr:'ADMETTRE',owner:'Institutional Governor',ownerFr:'Gouverneur institutionnel',meaning:'Allow the proven capability to influence future missions under limits.',meaningFr:'Permettre à la capacité prouvée d’influencer les missions futures sous limites.'},
  {id:'allocate',name:'ALLOCATE',nameFr:'ALLOUER',owner:'Capital-to-Capacity Committee',ownerFr:'Comité Capital-vers-Capacité',meaning:'Permit verified value to finance the next successor frontier.',meaningFr:'Permettre à la valeur vérifiée de financer la prochaine frontière successorale.'}
];
const AUTHORITY_ACTIONS = [
  {id:'analysis',name:'Internal analysis',min:'A0',reversible:true,consequential:false},
  {id:'evidence',name:'Evidence acquisition',min:'A0',reversible:true,consequential:false},
  {id:'recommendation',name:'Evidenced recommendation',min:'A1',reversible:true,consequential:false},
  {id:'sandbox',name:'Sealed-sandbox execution',min:'A2',reversible:true,consequential:false},
  {id:'bounded_external',name:'Reversible bounded external action',min:'A3',reversible:true,consequential:true},
  {id:'consequential',name:'Consequential external action',min:'A4',reversible:false,consequential:true}
];

function defaultOverlay(){
  return {
    meta:{release:RELEASE.version,id:RELEASE.id,createdAt:now(),updatedAt:now(),sourcePapers:['GoalOS Singularity Navigator Ω + SEIZE · Gym, Specialist ASI & Successor Ω — Unified Verified Succession Institution v6.0.0']},
    nested:Object.fromEntries(NESTED_GYMS.map(g=>[g.id,{status:'not_run',score:null,pass:false,runAt:null,evidence:[],failures:[]} ])),
    families:FAMILY_BLUEPRINTS.map((f,i)=>({...f,version:`${f.id.toUpperCase()}-1.0.0`,releaseHash:null,status:'not_conformed',checks:[],lastRunAt:null,independentCustody:f.id==='fresh'||f.id==='adversarial',protectedSeeds:f.id==='fresh'||f.id==='transfer',productionConnected:f.id==='canary'})),
    physics:{scenarioFidelity:82,protectedCaseIndependence:92,evaluatorUncertainty:8,canaryCalibration:0,observabilityDebt:0,controllabilityDebt:0,sensorMismatch:0,actuationMismatch:0,viability:0,stability:0,basisRisk:0,basisReserve:0,lastCalculatedAt:null},
    economics:{aiBeta:0,missionBeta:0,grossSpread:0,sovereigntySpread:0,missionAlpha:null,alphaState:'UNPROVEN',factors:{provider:0,data:0,verifier:0,human:0,integration:0,regime:0},charges:{completeBurden:0,beta:0,basis:0,proof:0,authority:0},lastCalculatedAt:null},
    decisions:Object.fromEntries(DECISIONS.map(d=>[d.id,{status:'unsigned',owner:d.owner,signedBy:'',signedAt:null,scope:'',receipt:null,prerequisites:[]} ])),
    authorityLattice:{dimensions:{identity:100,missionScope:100,systems:72,data:80,actionClass:25,amountLimit:100,time:90,reversibility:95,evidenceFreshness:0,professionalAuthority:45,dualControl:65,incidentCapacity:75},actionReceipts:[],lastEvaluatedAt:null},
    successorOmega:{status:'candidate',identity:null,version:null,qualifiedAt:null,admittedAt:null,omegaAt:null,proofState:'unproven',authorityState:'none',recursionFirewall:{protectedCasesExternal:true,selfCertificationProhibited:true,selfInstallationProhibited:true,authorityInheritanceProhibited:true,constitutionRewriteProhibited:true},qualification:[],claimBoundary:RELEASE.claimBoundary},
    handover:{status:'not_started',steps:[
      {id:'freeze_incumbent',name:'Freeze incumbent known-good state',status:'pending'},
      {id:'shadow',name:'Run challenger in matched shadow mode',status:'pending'},
      {id:'minimum_state',name:'Transfer minimum canary state',status:'pending'},
      {id:'fallback',name:'Assign fallback owner and response time',status:'pending'},
      {id:'canary',name:'Execute one reversible bounded action',status:'pending'},
      {id:'compare',name:'Compare real-like outcomes with Gym prediction',status:'pending'},
      {id:'basis_update',name:'Update Gym basis-risk estimate',status:'pending'},
      {id:'decision',name:'Expand, hold, repair, rollback or revoke',status:'pending'},
      {id:'chronicle',name:'Preserve handover evidence in Chronicle',status:'pending'}
    ],fallbackOwner:'Accountable human principal',responseMinutes:15,predictedGain:null,observedGain:null,basisGap:null,decision:null,runAt:null,receipt:null},
    ledger:{gymId:'GSSO-MISSION-GYM',owner:'Accountable institution',rightsCleared:false,portableAPI:true,independentCustody:true,protectedCases:true,providerNeutral:true,chronicleIntegrated:true,requalificationEnabled:true,eventSourced:true,environmentVersions:[],proofHistory:[],changeLog:[],maintenanceCost:25000,requalificationCost:15000,substitutionValue:150000,holdUpReduction:175000,learningReuseValue:225000,maturity:'G2',sovereigntyPremium:0,lastEvaluatedAt:null},
    receipts:[],lastCoreFingerprint:null
  };
}
let overlay = loadOverlay();
function loadOverlay(){
  try{
    let raw=localStorage.getItem(RELEASE.storageKey);
    if(!raw) for(const k of RELEASE.legacyKeys){raw=localStorage.getItem(k);if(raw)break}
    if(!raw)return defaultOverlay();
    const input=JSON.parse(raw),base=defaultOverlay();
    return {
      ...base,...input,
      meta:{...base.meta,...input.meta,release:RELEASE.version,id:RELEASE.id},
      nested:{...base.nested,...input.nested},
      families:Array.isArray(input.families)?FAMILY_BLUEPRINTS.map(f=>({...base.families.find(x=>x.id===f.id),...input.families.find(x=>x.id===f.id)})):base.families,
      physics:{...base.physics,...input.physics},economics:{...base.economics,...input.economics,factors:{...base.economics.factors,...input.economics?.factors},charges:{...base.economics.charges,...input.economics?.charges}},
      decisions:{...base.decisions,...input.decisions},authorityLattice:{...base.authorityLattice,...input.authorityLattice,dimensions:{...base.authorityLattice.dimensions,...input.authorityLattice?.dimensions}},
      successorOmega:{...base.successorOmega,...input.successorOmega,recursionFirewall:{...base.successorOmega.recursionFirewall,...input.successorOmega?.recursionFirewall}},
      handover:{...base.handover,...input.handover},ledger:{...base.ledger,...input.ledger}
    };
  }catch(e){console.warn('GSSO overlay recovery failed',e);return defaultOverlay()}
}
function save(){overlay.meta.updatedAt=now();try{localStorage.setItem(RELEASE.storageKey,JSON.stringify(overlay))}catch(e){console.warn(e)} }
function receipt(type,data){const r={id:`GSSO-${type}-${Date.now().toString(36).toUpperCase()}`,type,at:now(),release:RELEASE.id,coreFingerprint:fingerprintCore(),data:deep(data)};r.hash=hash(r);overlay.receipts.unshift(r);overlay.receipts=overlay.receipts.slice(0,300);save();return r}
function fingerprintCore(){const c=core();return hash({release:c?.meta?.release,project:c?.meta?.projectId,mission:c?.mission?.name,gym:c?.gym?.version,candidate:c?.proof?.frozenCandidateId,proof:c?.proof?.result?.id,authority:c?.authority?.admission?.status,generation:c?.meta?.institutionGeneration})}

function derivePhysics(){
  const c=core(),gym=c?.mission?.gymmability||[50,50,50,50,50,50,50,50],cand=proofCandidate(c),fr=formationResult(c,cand?.id),proof=c?.proof?.result;
  const observability=Number(gym[1]||50),bounded=Number(gym[2]||50),containment=Number(gym[6]||50),transfer=proof?.dimensions?.transfer ?? fr?.metrics?.transfer ?? Number(gym[7]||50);
  const governance=proof?.dimensions?.governance ?? fr?.metrics?.governance ?? cand?.params?.governance ?? 50;
  const reliability=proof?.dimensions?.reliability ?? fr?.metrics?.reliability ?? cand?.params?.resilience ?? 50;
  const evidence=fr?.metrics?.evidence ?? cand?.params?.evidence ?? 50;
  overlay.physics.observabilityDebt=clamp(100-observability);
  overlay.physics.controllabilityDebt=clamp(100-bounded);
  overlay.physics.sensorMismatch=clamp((100-observability)*.55+(100-evidence)*.25+(100-overlay.physics.scenarioFidelity)*.20);
  overlay.physics.actuationMismatch=clamp((100-bounded)*.45+(100-containment)*.30+Math.max(0,authorityRank(c?.authority?.level||'A0')-1)*8);
  overlay.physics.viability=clamp(governance*.42+reliability*.35+containment*.23-(proof?.criticalErrors||0)*30-(proof?.unauthorizedActions||0)*40);
  overlay.physics.stability=clamp(reliability*.45+(cand?.params?.resilience||50)*.25+governance*.20+(c?.authority?.stress?.pass?10:0));
  const canaryPenalty=overlay.physics.canaryCalibration>0?100-overlay.physics.canaryCalibration:35;
  overlay.physics.basisRisk=clamp((100-transfer)*.25+overlay.physics.sensorMismatch*.20+overlay.physics.actuationMismatch*.15+(100-overlay.physics.scenarioFidelity)*.15+(100-overlay.physics.protectedCaseIndependence)*.10+overlay.physics.evaluatorUncertainty*.10+canaryPenalty*.05);
  overlay.physics.basisReserve=clamp(overlay.physics.basisRisk*.55+(100-overlay.physics.viability)*.20+(100-overlay.physics.stability)*.10);
  overlay.physics.lastCalculatedAt=now();
  return overlay.physics;
}
function deriveEconomics(){
  const c=core(),cand=proofCandidate(c),candidateResult=formationResult(c,cand?.id),inc=formationResult(c,'incumbent'),beta=formationResult(c,'general_ai'),proof=c?.proof?.result;
  const candidateUtility=candidateResult?.metrics?.utility ?? mean(Object.values(cand?.params||{}))*.55;
  const incUtility=inc?.metrics?.utility ?? 40,betaUtility=beta?.metrics?.utility ?? 55,best=Math.max(incUtility,betaUtility);
  const factors={
    provider:clamp(cand?.burden?.dependency ?? 50),
    data:clamp(100-(candidateResult?.metrics?.evidence ?? cand?.params?.evidence ?? 50)),
    verifier:clamp(100-(cand?.params?.verifier ?? 50)),
    human:clamp(100-(candidateResult?.metrics?.humanScore ?? cand?.params?.humanEfficiency ?? 50)),
    integration:clamp(cand?.burden?.migration ?? 50),
    regime:clamp(100-(proof?.dimensions?.transfer ?? candidateResult?.metrics?.transfer ?? 50))
  };
  const missionBeta=factors.provider*.22+factors.data*.16+factors.verifier*.16+factors.human*.13+factors.integration*.15+factors.regime*.18;
  const aiBeta=betaUtility;
  const grossSpread=candidateUtility-best;
  const physics=derivePhysics();
  const completeBurden=(candidateResult?.burden ?? 50)*.055;
  const betaCharge=missionBeta*.035;
  const basisCharge=physics.basisReserve*.045;
  const proofCharge=proof?.pass?Math.max(0,1-(proof?.paired?.lcb||0))*.8:Math.max(2,Math.abs(Math.min(0,proof?.paired?.lcb||0))*.35+3);
  const authorityCharge=Math.max(0,authorityRank(c?.authority?.level||'A0'))*.48;
  const sovereigntySpread=grossSpread-completeBurden-betaCharge-basisCharge-proofCharge-authorityCharge;
  const missionAlpha=proof?.pass?Math.min(proof.paired.lcb,sovereigntySpread):null;
  overlay.economics={...overlay.economics,aiBeta,missionBeta,grossSpread,sovereigntySpread,missionAlpha,alphaState:proof?.pass&&missionAlpha>0?'PROVEN IN SIMULATION':'UNPROVEN',factors,charges:{completeBurden,beta:betaCharge,basis:basisCharge,proof:proofCharge,authority:authorityCharge},lastCalculatedAt:now()};
  return overlay.economics;
}
function nestedEvidence(kind){
  const c=core(),p=derivePhysics(),e=deriveEconomics(),proof=c?.proof?.result,stress=c?.authority?.stress;
  if(kind==='mission') return {score:clamp(mean(c?.mission?.gymmability||[])*.45+(proof?.dimensions?.capability??formationResult(c,c?.formation?.championId)?.metrics?.quality??50)*.35+(proof?.dimensions?.transfer??50)*.20),evidence:['Gymmability vector','Matched formation','Fresh transfer result'],failures:[p.observabilityDebt>35?'Observability debt exceeds 35':null,p.controllabilityDebt>35?'Controllability debt exceeds 35':null].filter(Boolean)};
  if(kind==='institutional') return {score:clamp((proof?.dimensions?.governance??50)*.25+(proof?.dimensions?.reliability??50)*.22+(proof?.proofCoverage??50)*.18+(stress?.pass?18:0)+p.stability*.17),evidence:['Producer–critic–verifier separation','Authority stress','Evidence custody','Degraded mode'],failures:[!stress?.pass?'Authority stress not passed':null,p.stability<72?'Operational stability below 72':null].filter(Boolean)};
  const signed=Object.values(overlay.decisions).filter(d=>d.status==='signed').length;
  return {score:clamp((c?.recursive?.generation||0)*6+Math.min(24,(c?.negativeCapability?.length||0)*2)+(c?.requalification?.result?12:0)+signed*7+(c?.authority?.admission?.status==='simulated_admitted'?13:0)+(overlay.handover.decision?10:0)),evidence:['Quality-diversity archive','Negative Capability Graph','Decision separation','Requalification','Handover'],failures:[signed<4?'Fewer than four institutional decisions signed':null,!c?.recursive?.generation?'No successor generation completed':null].filter(Boolean)};
}
function runNested(kind){const r=nestedEvidence(kind);overlay.nested[kind]={status:r.score>=75&&!r.failures.length?'passed':'needs_work',score:r.score,pass:r.score>=75&&!r.failures.length,runAt:now(),evidence:r.evidence,failures:r.failures};receipt('NESTED_GYM',{kind,result:overlay.nested[kind]});save();render();}
function runAllNested(){NESTED_GYMS.forEach(g=>runNested(g.id));}

function familyChecks(f){
  const c=core(),proof=c?.proof?.result;
  const base=[{name:'Exact environment release',pass:Boolean(c?.gym?.version),detail:c?.gym?.version||'missing'},{name:'Typed custody boundary',pass:Boolean(f.custodian),detail:f.custodian}];
  if(f.id==='formation')base.push({name:'Matched candidate evidence',pass:Boolean(c?.formation?.results?.length),detail:`${c?.formation?.results?.length||0} candidates`});
  if(f.id==='adversarial')base.push({name:'Failure-preserving memory',pass:(c?.negativeCapability?.length||0)>0,detail:`${c?.negativeCapability?.length||0} records`},{name:'Authority challenge',pass:Boolean(c?.authority?.stress),detail:c?.authority?.stress?.pass?'passed':'not passed'});
  if(f.id==='transfer')base.push({name:'Transfer evidence',pass:(proof?.dimensions?.transfer||0)>=70,detail:`${round(proof?.dimensions?.transfer||0)}%`});
  if(f.id==='fresh')base.push({name:'Frozen exact challenger',pass:Boolean(c?.proof?.manifestHash),detail:c?.proof?.manifestHash||'none'},{name:'Protected proof completed',pass:Boolean(proof),detail:proof?.status||'not run'},{name:'Formation roles excluded',pass:Boolean(c?.proof?.protectedLocked),detail:c?.proof?.protectedLocked?'locked':'unlocked'});
  if(f.id==='canary')base.push({name:'Accountable admission exists',pass:c?.authority?.admission?.status==='simulated_admitted',detail:c?.authority?.admission?.status||'none'},{name:'Reversible handover executed',pass:Boolean(overlay.handover.decision),detail:overlay.handover.decision||'not run'});
  return base;
}
function runFamily(id){const f=overlay.families.find(x=>x.id===id);if(!f)return;f.checks=familyChecks(f);f.status=f.checks.every(x=>x.pass)?'conformed':'needs_work';f.lastRunAt=now();f.releaseHash=hash({id:f.id,version:f.version,custodian:f.custodian,checks:f.checks,core:fingerprintCore()});receipt('GYM_FAMILY_RELEASE',{id:f.id,version:f.version,status:f.status,hash:f.releaseHash});save();deriveLedger();render();}
function runAllFamilies(){overlay.families.forEach(f=>runFamily(f.id));}

function decisionPrerequisites(id){
  const c=core();
  const map={
    underwrite:[['SEIZE has a decision',c?.seize?.decision&&c.seize.decision!=='Not underwritten'],['Mission Constitution frozen',Boolean(c?.mission?.constitutionFrozen)]],
    execute:[['Mission Constitution frozen',Boolean(c?.mission?.constitutionFrozen)],['Bounded AGI Jobs compiled',(c?.jobs||[]).some(j=>j.status!=='not_started')]],
    accept:[['Fresh proof passed',Boolean(c?.proof?.result?.pass)],['Exact challenger manifest',Boolean(c?.proof?.manifestHash)]],
    admit:[['Accountable demo admission exists',c?.authority?.admission?.status==='simulated_admitted'],['Authority stress passed',Boolean(c?.authority?.stress?.pass)]],
    allocate:[['ADMIT decision signed',overlay.decisions.admit.status==='signed'],['Accepted learning or successor generation',(c?.meta?.institutionGeneration||0)>0||(c?.recursive?.generation||0)>0]]
  };
  return map[id]||[];
}
function signDecision(id){const d=overlay.decisions[id];if(!d)return;const prereq=decisionPrerequisites(id);d.prerequisites=prereq.map(([name,pass])=>({name,pass:Boolean(pass)}));if(d.prerequisites.some(x=>!x.pass)){toastLocal(t('Prerequisites are not satisfied.','Les préalables ne sont pas satisfaits.'));render();return}const signer=q('#gssDecisionSigner')?.value?.trim()||'Demonstration Principal';d.status='signed';d.signedBy=signer;d.signedAt=now();d.scope=q(`#decisionScope_${id}`)?.value?.trim()||DECISIONS.find(x=>x.id===id)?.meaning;d.receipt=receipt('DECISION',{decision:id,signer,scope:d.scope,prerequisites:d.prerequisites});save();deriveOmega();render();}
function revokeDecision(id){const d=overlay.decisions[id];if(!d)return;d.status='revoked';d.revokedAt=now();receipt('DECISION_REVOKED',{decision:id});save();deriveOmega();render();}

function authorityDimensions(){
  const c=core(),p=derivePhysics(),proof=c?.proof?.result;
  return {
    identity:c?.proof?.manifestHash?100:25,
    missionScope:c?.mission?.constitutionFrozen?100:45,
    systems:clamp((c?.authority?.permitted?.length||0)*14),
    data:clamp((proof?.proofCoverage||50)*.82),
    actionClass:clamp((authorityRank(c?.authority?.level||'A0')+1)*20),
    amountLimit:c?.authority?.level==='A4'?55:c?.authority?.level==='A3'?78:100,
    time:c?.authority?.expiry?90:35,
    reversibility:clamp(p.viability*.55+(c?.authority?.stress?.pass?40:0)),
    evidenceFreshness:proof?.runAt?clamp(100-(Date.now()-new Date(proof.runAt).getTime())/86400000*2):0,
    professionalAuthority:c?.authority?.owner&&c.authority.owner!=='Accountable human principal'?70:45,
    dualControl:overlay.decisions.admit.status==='signed'?85:55,
    incidentCapacity:clamp(p.stability*.7+(c?.authority?.stress?.pass?25:0))
  };
}
function evaluateAuthority(){overlay.authorityLattice.dimensions={...overlay.authorityLattice.dimensions,...authorityDimensions()};overlay.authorityLattice.lastEvaluatedAt=now();receipt('AUTHORITY_LATTICE',{dimensions:overlay.authorityLattice.dimensions,level:core()?.authority?.level});save();render();}
function simulateAction(){
  const c=core(),actionId=q('#gssActionClass')?.value||'analysis',action=AUTHORITY_ACTIONS.find(a=>a.id===actionId),system=q('#gssActionSystem')?.value?.trim()||'Approved mission workspace',dataClass=q('#gssActionData')?.value?.trim()||'Approved non-secret mission data',amount=Number(q('#gssActionAmount')?.value||0),current=Boolean(q('#gssEvidenceCurrent')?.checked),reversible=Boolean(q('#gssActionReversible')?.checked),dims=authorityDimensions();
  const level=c?.authority?.level||'A0',checks=[
    {name:'Authority level',pass:authorityRank(level)>=authorityRank(action.min),detail:`${level} ≥ ${action.min}`},
    {name:'Fresh evidence',pass:current&&dims.evidenceFreshness>=60,detail:`${round(dims.evidenceFreshness)}% freshness`},
    {name:'Mission identity',pass:dims.identity===100&&dims.missionScope===100,detail:`identity ${dims.identity}; mission ${dims.missionScope}`},
    {name:'Reversibility',pass:!action.reversible||reversible,detail:reversible?'declared reversible':'not reversible'},
    {name:'Amount limit',pass:amount<=10000||authorityRank(level)>=3,detail:`${amount}`},
    {name:'Incident capacity',pass:dims.incidentCapacity>=70||!action.consequential,detail:`${round(dims.incidentCapacity)}%`}
  ];
  const allowed=checks.every(x=>x.pass);const r=receipt('PROOF_CARRYING_ACTION',{action:action.id,system,dataClass,amount,reversible,authorityLevel:level,checks,verdict:allowed?'AUTHORIZED_SIMULATION':'BLOCKED'});overlay.authorityLattice.actionReceipts.unshift({...r,allowed,checks,actionName:action.name,system});overlay.authorityLattice.actionReceipts=overlay.authorityLattice.actionReceipts.slice(0,50);save();render();toastLocal(allowed?t('Action passed the simulated external gate.','L’action a franchi la porte externe simulée.'):t('Action blocked by the Authority Lattice.','Action bloquée par la grille d’autorité.'));}

function runHandover(){
  const c=core(),proof=c?.proof?.result;if(!proof?.pass||c?.authority?.admission?.status!=='simulated_admitted'){toastLocal(t('Fresh proof and accountable demo admission are required first.','La preuve fraîche et l’admission responsable de démonstration sont d’abord requises.'));return}
  overlay.handover.steps.forEach(s=>s.status='passed');
  const predicted=Number(proof.paired.lcb||0),seed=parseInt(hash({predicted,core:fingerprintCore()}).slice(0,6),16),variation=((seed%1000)/1000-.5)*5.2,observed=predicted+variation,basisGap=Math.abs(variation);
  const critical=(proof.criticalErrors||0)>0||(proof.unauthorizedActions||0)>0;
  const decision=critical?'rollback_or_revoke':basisGap<=2.5?'expand':basisGap<=5?'hold_and_reprove':'repair';
  if(decision!=='expand')overlay.handover.steps.find(s=>s.id==='decision').status='warning';
  overlay.handover={...overlay.handover,status:'completed',predictedGain:predicted,observedGain:observed,basisGap,decision,runAt:now()};
  overlay.handover.receipt=receipt('SUCCESSOR_HANDOVER',{predicted,observed,basisGap,decision,fallbackOwner:overlay.handover.fallbackOwner,responseMinutes:overlay.handover.responseMinutes});
  overlay.physics.canaryCalibration=clamp(100-basisGap*10);
  derivePhysics();runFamily('canary');deriveLedger();deriveOmega();save();render();
}
function rollbackHandover(){overlay.handover.status='rolled_back';overlay.handover.decision='rollback_or_revoke';overlay.handover.steps.forEach(s=>{if(['canary','compare','basis_update','decision'].includes(s.id))s.status='rolled_back'});receipt('HANDOVER_ROLLBACK',{fallbackOwner:overlay.handover.fallbackOwner});deriveOmega();save();render();}

function deriveLedger(){
  const l=overlay.ledger,c=core(),familyCount=overlay.families.filter(f=>f.status==='conformed').length,nestedCount=Object.values(overlay.nested).filter(x=>x.pass).length;
  const conditions=[true,Boolean(c?.gym?.version),Boolean(c?.gym?.spec?.state?.length),nestedCount>=2,overlay.families.find(f=>f.id==='fresh')?.status==='conformed',Boolean(overlay.handover.decision),l.rightsCleared&&l.portableAPI&&l.independentCustody&&l.chronicleIntegrated&&l.requalificationEnabled,(c?.recursive?.generation||0)>0&&overlay.families.every(f=>f.status==='conformed')];
  let level=0;conditions.forEach((v,i)=>{if(v&&i===level)level=i+1});level=Math.min(7,level);
  l.maturity=`G${level}`;
  l.sovereigntyPremium=l.holdUpReduction+l.substitutionValue+l.learningReuseValue-l.maintenanceCost-l.requalificationCost-derivePhysics().basisReserve*2500;
  l.lastEvaluatedAt=now();
  if(c?.gym?.version&&!l.environmentVersions.some(v=>v.version===c.gym.version))l.environmentVersions.push({version:c.gym.version,addedAt:now(),hash:hash({version:c.gym.version,spec:c.gym.spec})});
  if(c?.proof?.result&&!l.proofHistory.some(p=>p.id===c.proof.result.id))l.proofHistory.push({id:c.proof.result.id,at:c.proof.result.runAt,pass:c.proof.result.pass,candidate:c.proof.result.candidateName,environment:c.proof.result.environmentVersion});
  save();return l;
}
function saveLedger(){const ids=['rightsCleared','portableAPI','independentCustody','protectedCases','providerNeutral','chronicleIntegrated','requalificationEnabled','eventSourced'];ids.forEach(id=>{const e=q(`#ledger_${id}`);if(e)overlay.ledger[id]=e.checked});['maintenanceCost','requalificationCost','substitutionValue','holdUpReduction','learningReuseValue'].forEach(id=>{const e=q(`#ledger_${id}`);if(e)overlay.ledger[id]=Number(e.value||0)});overlay.ledger.changeLog.unshift({at:now(),type:'LEDGER_UPDATED',by:q('#gssDecisionSigner')?.value||'Demonstration Principal',hash:hash(overlay.ledger)});deriveLedger();receipt('SOVEREIGN_GYM_LEDGER',{maturity:overlay.ledger.maturity,premium:overlay.ledger.sovereigntyPremium});render();}

function omegaQualification(){
  const c=core(),proof=c?.proof?.result;
  return [
    {name:'Mission Gym exists as a versioned executable environment',pass:Boolean(c?.gym?.version&&c?.gym?.spec?.state?.length)},
    {name:'Mission ≠ candidate ≠ admitted institution is preserved',pass:true},
    {name:'One exact frozen candidate passed fresh proof',pass:Boolean(proof?.pass&&c?.proof?.manifestHash)},
    {name:'Accountable admission and scoped authority exist',pass:c?.authority?.admission?.status==='simulated_admitted'},
    {name:'Mission, Institutional and Succession Gyms passed',pass:Object.values(overlay.nested).every(x=>x.pass)},
    {name:'Five Gym families are independently conformed',pass:overlay.families.every(f=>f.status==='conformed')},
    {name:'Underwrite, Execute, Accept and Admit are separately signed',pass:['underwrite','execute','accept','admit'].every(id=>overlay.decisions[id].status==='signed')},
    {name:'Authority Lattice has fresh evidence and incident capacity',pass:(overlay.authorityLattice.dimensions.evidenceFreshness||0)>=60&&(overlay.authorityLattice.dimensions.incidentCapacity||0)>=70},
    {name:'Reversible successor handover completed',pass:Boolean(overlay.handover.decision)&&overlay.handover.decision!=='rollback_or_revoke'},
    {name:'Sovereign Gym Ledger reached at least G6',pass:Number((deriveLedger().maturity||'G0').slice(1))>=6},
    {name:'Recursive Foundry produced at least one later generation',pass:(c?.recursive?.generation||0)>0},
    {name:'Recursion firewall remains non-bypassable',pass:Object.values(overlay.successorOmega.recursionFirewall).every(Boolean)}
  ];
}
function deriveOmega(){
  const c=core(),proof=c?.proof?.result,qf=omegaQualification();
  let status='candidate';if(proof?.pass)status='mission_dominant_specialist_asi';if(c?.authority?.admission?.status==='simulated_admitted')status='mission_sovereign_successor';if(overlay.successorOmega.omegaAt&&qf.every(x=>x.pass))status='successor_omega_demo';
  overlay.successorOmega={...overlay.successorOmega,status,identity:c?.proof?.manifestHash||null,version:candidateById(c,c?.proof?.frozenCandidateId)?.generation!=null?`${candidateById(c,c.proof.frozenCandidateId).id}@G${candidateById(c,c.proof.frozenCandidateId).generation}`:null,qualifiedAt:proof?.pass?proof.runAt:null,admittedAt:c?.authority?.admission?.signedAt||null,proofState:proof?.pass?'fresh-proof-pass':'unproven',authorityState:c?.authority?.admission?.status||'none',qualification:qf};save();return overlay.successorOmega;
}
function constituteOmega(){const qf=omegaQualification();if(qf.some(x=>!x.pass)){overlay.successorOmega.qualification=qf;save();render();toastLocal(t('Successor Ω prerequisites are incomplete.','Les préalables du Successeur Ω sont incomplets.'));return}if(!confirm(t('Constitute the local demonstration state Successor Ω? This creates no real-world authority.','Constituer l’état local de démonstration Successeur Ω? Cela ne crée aucune autorité réelle.')))return;overlay.successorOmega.omegaAt=now();overlay.successorOmega.status='successor_omega_demo';receipt('SUCCESSOR_OMEGA_DEMO',{candidate:overlay.successorOmega.identity,qualification:qf});save();render();}
function revokeOmega(){overlay.successorOmega.omegaAt=null;overlay.successorOmega.status='revoked';receipt('SUCCESSOR_OMEGA_REVOKED',{});save();render();}

function updateFromCore(){
  const fp=fingerprintCore();
  if(fp!==overlay.lastCoreFingerprint){overlay.lastCoreFingerprint=fp;derivePhysics();deriveEconomics();evaluateAuthoritySilent();deriveLedger();deriveOmega();save();render();}
}
function evaluateAuthoritySilent(){overlay.authorityLattice.dimensions={...overlay.authorityLattice.dimensions,...authorityDimensions()};overlay.authorityLattice.lastEvaluatedAt=now();}

function metric(value,label,detail=''){return `<div class="gss-metric"><b>${esc(value)}</b><span>${esc(label)}</span>${detail?`<small>${esc(detail)}</small>`:''}</div>`}
function statusBadge(status){const pass=['passed','conformed','signed','successor_omega_demo','completed'].includes(status);const warn=['needs_work','warning','mission_dominant_specialist_asi','mission_sovereign_successor'].includes(status);return `<span class="gss-status ${pass?'pass':warn?'warn':'neutral'}">${esc(String(status||'unknown').replaceAll('_',' '))}</span>`}
function header(kicker,title,copy,actions=''){return `<div class="page-head gss-page-head"><div><div class="page-kicker">${esc(kicker)}</div><h1>${title}</h1><p>${copy}</p></div>${actions?`<div class="page-actions">${actions}</div>`:''}</div>`}
function bar(value,label){return `<div class="gss-bar"><div><span>${esc(label)}</span><b>${round(value)}%</b></div><i><em style="width:${clamp(value)}%"></em></i></div>`}
function toastLocal(message){let e=q('#toast');if(e){e.textContent=message;e.classList.remove('hidden');clearTimeout(toastLocal.timer);toastLocal.timer=setTimeout(()=>e.classList.add('hidden'),3000)}else console.info(message)}
function section(id){return q(`#section-${id}`)}

function renderTriad(){const c=core();section('triad').innerHTML=header('01 · Categorical invariant','Gym ≠ Specialist ASI ≠ Successor Ω','Three different institutional objects answer three different questions. A score cannot collapse environment, candidate and admitted authority into one object.')+`<div class="gss-triad"><article><span>𝒢ₘ</span><h2>MISSION GYM</h2><p>Makes the mission executable, repeatable, adversarial and measurable.</p><b>Can this architecture perform?</b></article><div class="gss-not-equal">≠</div><article><span>𝑥★ₘ</span><h2>SPECIALIST ASI</h2><p>The complete frozen candidate that proves superiority on protected fresh work.</p><b>Which candidate wins?</b></article><div class="gss-not-equal">≠</div><article><span>Ωₘ</span><h2>SUCCESSOR Ω</h2><p>The proven candidate after accountable admission, identity, memory, rollback and scoped authority.</p><b>What authority is justified?</b></article></div><div class="gss-rule"><strong>NO AUTOMATIC AUTHORITY</strong><span>Mission dominance does not imply consequential authority. Training rewards teach. Protected proof decides. Accountable authority admits.</span></div><div class="grid-2"><div class="card"><h2>The complete Successor is larger than the policy</h2><div class="gss-stack">${['Policy portfolio','World model & evidence graph','Tools, retrieval & deterministic systems','Critic & independent verifier','Human roles & workflow integration','Identity, proof-carrying action & rollback','Chronicle, rights, portability & requalification'].map(x=>`<span>${x}</span>`).join('')}</div></div><div class="card navy"><h2>Current local state</h2>${metric(coreStatus(c),'Constitutional state')}${metric(c?.proof?.manifestHash||'Not frozen','Exact candidate identity')}${metric(c?.authority?.level||'A0','Current authority level')}<p class="fine">${RELEASE.claimBoundary}</p></div></div>`}
function renderPhysics(){const p=derivePhysics();section('physics').innerHTML=header('03 · Mission physics','Reality gap, observability, controllability and stability','The Gym is useful only when its state, sensors, actions, dynamics and evaluation remain sufficiently faithful to the real mission.',`<button class="button" data-gss-action="save-physics">Recalculate physics</button>`)+`<div class="metrics-grid gss-metrics-6">${metric(round(p.observabilityDebt),'Observability debt')}${metric(round(p.controllabilityDebt),'Controllability debt')}${metric(round(p.sensorMismatch),'Sensor mismatch')}${metric(round(p.actuationMismatch),'Actuation mismatch')}${metric(round(p.viability),'Viability')}${metric(round(p.stability),'Stability')}</div><div class="grid-2"><div class="card"><h2>Reality-gap assumptions</h2>${[['scenarioFidelity','Scenario fidelity'],['protectedCaseIndependence','Protected-case independence'],['evaluatorUncertainty','Evaluator uncertainty'],['canaryCalibration','Canary calibration']].map(([id,label])=>`<label class="gss-slider"><span>${label}<b id="physicsValue_${id}">${round(p[id])}</b></span><input id="physics_${id}" type="range" min="0" max="100" value="${p[id]}" oninput="document.getElementById('physicsValue_${id}').textContent=this.value"></label>`).join('')}</div><div class="card highlight"><h2>Basis-risk underwriting</h2>${bar(p.basisRisk,'Gym basis risk')}${bar(p.basisReserve,'Basis-risk reserve')}${bar(p.viability,'Viability kernel proxy')}${bar(p.stability,'Lyapunov-style stability proxy')}<div class="callout ${p.basisRisk<=25?'green':p.basisRisk<=45?'':'red'}"><strong>${p.basisRisk<=25?'Controlled basis risk':p.basisRisk<=45?'Material basis risk':'High basis risk'}</strong><br>Demonstrated advantage must remain positive after the Gym-to-reality reserve is charged.</div></div></div><div class="gss-equation">α<sub>reality-LB</sub> = α<sub>fresh proof</sub> − ε<sub>gym basis</sub> − D<sub>proof</sub> − R<sub>authority</sub></div>`}
function renderNested(){section('nested').innerHTML=header('04 · Three nested Gyms','Mission competence is not institutional competence','The inner task, the operating institution and the succession decision fail in different ways.',`<button class="button" data-gss-action="run-all-nested">Run all three Gyms</button>`)+`<div class="gss-nested">${NESTED_GYMS.slice().reverse().map((g,index)=>{const r=overlay.nested[g.id];return `<article class="level-${index}"><div><span>${index===0?'OUTER':index===1?'MIDDLE':'INNER'}</span><h2>${esc(t(g.name,g.nameFr))}</h2><p>${esc(t(g.question,g.questionFr))}</p></div>${statusBadge(r.status)}${r.score!=null?bar(r.score,'Institutional score'):''}<ul>${r.evidence.map(e=>`<li>${esc(e)}</li>`).join('')}${r.failures.map(e=>`<li class="fail">${esc(e)}</li>`).join('')}</ul><button class="button small secondary" data-gss-action="run-nested" data-kind="${g.id}">Run ${esc(t(g.name,g.nameFr))}</button></article>`}).join('')}</div>`}
function renderFamilies(){section('families').innerHTML=header('05 · Five environment families','The Gym that teaches must not be the Gym that certifies','Formation, attack, transfer, fresh proof and production-connected requalification require different visibility and custody.',`<button class="button" data-gss-action="run-all-families">Conformance-check all families</button>`)+`<div class="gss-family-grid">${overlay.families.map((f,i)=>`<article><div class="gss-family-top"><span>0${i+1}</span>${statusBadge(f.status)}</div><h2>${esc(t(f.name,f.nameFr))}</h2><p>${esc(t(f.purpose,f.purposeFr))}</p><dl><dt>Release</dt><dd>${esc(f.version)}</dd><dt>Custodian</dt><dd>${esc(f.custodian)}</dd><dt>Visibility</dt><dd>${esc(t(f.visibility,f.visibilityFr))}</dd><dt>Promotion weight</dt><dd>${f.promotionWeight}</dd></dl>${f.releaseHash?`<code>${f.releaseHash}</code>`:''}<div class="gss-checks">${f.checks.map(x=>`<div class="${x.pass?'pass':'fail'}"><b>${x.pass?'✓':'×'}</b><span>${esc(x.name)}<small>${esc(x.detail)}</small></span></div>`).join('')}</div><button class="button small secondary" data-gss-action="run-family" data-family="${f.id}">Run conformance</button></article>`).join('')}</div>`}
function renderAlpha(){const e=deriveEconomics();section('alpha').innerHTML=header('08 · Alpha, Beta and Spread','From rented intelligence to robust mission advantage','AI Beta is the common market baseline. Mission Beta is systematic dependence. The Sovereignty Spread is the robust net opportunity. Mission Alpha exists only after fresh proof.',`<button class="button" data-gss-action="recalculate-alpha">Recalculate</button>`)+`<div class="gss-alpha-flow"><article><span>β<sub>AI</sub></span><h2>AI BETA</h2><b>${round(e.aiBeta)}</b><p>Broadly rentable capability.</p></article><i>→</i><article><span>Δ</span><h2>MISSION ADVANTAGE SPREAD</h2><b>${round(e.grossSpread,2)}</b><p>Gross reachable opportunity.</p></article><i>→</i><article><span>Σ</span><h2>SOVEREIGNTY SPREAD</h2><b>${round(e.sovereigntySpread,2)}</b><p>Net opportunity after burdens and reserves.</p></article><i>→</i><article class="alpha"><span>αₘ</span><h2>MISSION ALPHA</h2><b>${e.missionAlpha==null?'UNPROVEN':round(e.missionAlpha,2)}</b><p>Fresh-proof residual advantage.</p></article></div><div class="grid-2"><div class="card"><h2>Mission Beta factor exposures</h2>${Object.entries(e.factors).map(([k,v])=>bar(v,k[0].toUpperCase()+k.slice(1))).join('')}<div class="gss-equation">βₘ = systematic sensitivity to providers + data + verifier + humans + integration + regime</div></div><div class="card"><h2>Complete-denominator charges</h2>${Object.entries(e.charges).map(([k,v])=>bar(Math.min(100,v*10),`${k}: ${round(v,2)} utility points`)).join('')}<div class="callout ${e.sovereigntySpread>0?'green':'red'}"><strong>${e.alphaState}</strong><br>Gross Spread ${round(e.grossSpread,2)} → Sovereignty Spread ${round(e.sovereigntySpread,2)} → Mission Alpha ${e.missionAlpha==null?'not established':round(e.missionAlpha,2)}.</div></div></div>`}
function renderDecisions(){section('decisions').innerHTML=header('10 · Institutional decisions','Underwrite ≠ Execute ≠ Accept ≠ Admit ≠ Allocate','Each decision creates a different authority. The candidate owns none of them.',`<label class="gss-inline-input">Signer <input id="gssDecisionSigner" value="Demonstration Principal"></label>`)+`<div class="gss-decision-grid">${DECISIONS.map((d,i)=>{const r=overlay.decisions[d.id],pre=decisionPrerequisites(d.id);return `<article><div class="gss-decision-no">0${i+1}</div><h2>${esc(t(d.name,d.nameFr))}</h2><p>${esc(t(d.meaning,d.meaningFr))}</p><small>${esc(t(d.owner,d.ownerFr))}</small>${statusBadge(r.status)}<textarea id="decisionScope_${d.id}" rows="2" placeholder="Exact scope">${esc(r.scope||'')}</textarea><div class="gss-checks">${pre.map(([name,pass])=>`<div class="${pass?'pass':'fail'}"><b>${pass?'✓':'×'}</b><span>${esc(name)}</span></div>`).join('')}</div><div class="button-row"><button class="button small" data-gss-action="sign-decision" data-decision="${d.id}">Sign</button>${r.status==='signed'?`<button class="button small danger" data-gss-action="revoke-decision" data-decision="${d.id}">Revoke</button>`:''}</div>${r.receipt?`<code>${r.receipt.hash}</code>`:''}</article>`}).join('')}</div>`}
function renderOmega(){const o=deriveOmega(),c=core();section('omega').innerHTML=header('16 · Successor Ω','The proven winner admitted to act, compound and renew','Successor Ω is reserved for an admitted institution that also preserves memory, requalifies itself, generates challengers and prepares its next successor without self-authorizing.',`<button class="button" data-gss-action="constitute-omega">Constitute Successor Ω · Demo</button><button class="button danger" data-gss-action="revoke-omega">Revoke</button>`)+`<div class="gss-omega-state"><div class="omega-core">Ω</div><div><span>${statusBadge(o.status)}</span><h2>${esc(candidateById(c,c?.proof?.frozenCandidateId)?.name||'No frozen candidate')}</h2><p>Identity: <code>${esc(o.identity||'not constituted')}</code></p><p>Version: ${esc(o.version||'—')} · Proof: ${esc(o.proofState)} · Authority: ${esc(o.authorityState)}</p></div></div><div class="gss-state-ladder">${[['Candidate','candidate'],['Verified intelligence','verified'],['Mission-Dominant Specialist ASI','mission_dominant_specialist_asi'],['Mission-Sovereign Successor','mission_sovereign_successor'],['Successor Ω','successor_omega_demo']].map(([label,id],i)=>`<div class="${(['candidate','verified','mission_dominant_specialist_asi','mission_sovereign_successor','successor_omega_demo'].indexOf(o.status)>=i)?'reached':''}"><b>${i+1}</b><span>${label}</span></div>`).join('')}</div><div class="grid-2"><div class="card"><h2>Ω qualification</h2><div class="gss-checks">${o.qualification.map(x=>`<div class="${x.pass?'pass':'fail'}"><b>${x.pass?'✓':'×'}</b><span>${esc(x.name)}</span></div>`).join('')}</div></div><div class="card navy"><h2>Recursion firewall</h2>${Object.entries(o.recursionFirewall).map(([k,v])=>`<div class="gss-firewall"><b>${v?'LOCKED':'OPEN'}</b><span>${esc(k.replace(/([A-Z])/g,' $1'))}</span></div>`).join('')}<div class="callout red"><strong>No self-authorization</strong><br>The successor may propose descendants. It may not inspect protected proof, certify itself, install itself, inherit authority or rewrite the Mission Constitution.</div></div></div>`}
function renderAuthorityLatticeEnhancement(){const host=section('authority');if(!host)return;const existing=host.querySelector('.gss-authority-extension');if(existing)existing.remove();const d=authorityDimensions();overlay.authorityLattice.dimensions={...overlay.authorityLattice.dimensions,...d};const box=document.createElement('div');box.className='gss-authority-extension';box.innerHTML=`<div class="section-divider"><span>v4 Authority Lattice & proof-carrying action</span></div><div class="grid-2"><div class="card"><h2>Authority is multidimensional</h2>${Object.entries(d).map(([k,v])=>bar(v,k.replace(/([A-Z])/g,' $1'))).join('')}<button class="button secondary" data-gss-action="evaluate-authority">Re-evaluate lattice</button></div><div class="card highlight"><h2>Simulate one proof-carrying action</h2><div class="form-grid"><label class="field"><span>Action class</span><select id="gssActionClass">${AUTHORITY_ACTIONS.map(a=>`<option value="${a.id}">${a.name} · ${a.min}</option>`).join('')}</select></label><label class="field"><span>System</span><input id="gssActionSystem" value="Approved mission workspace"></label><label class="field"><span>Data class</span><input id="gssActionData" value="Approved non-secret mission data"></label><label class="field"><span>Amount / exposure</span><input id="gssActionAmount" type="number" value="0"></label><label class="check"><input id="gssActionReversible" type="checkbox" checked> Reversible</label><label class="check"><input id="gssEvidenceCurrent" type="checkbox" checked> Evidence current</label></div><button class="button" data-gss-action="simulate-action">Run external gate</button></div></div><div class="card" style="margin-top:16px"><h2>Proof-carrying action receipts</h2><div class="table-wrap"><table><thead><tr><th>Time</th><th>Action</th><th>System</th><th>Verdict</th><th>Receipt</th></tr></thead><tbody>${overlay.authorityLattice.actionReceipts.slice(0,10).map(r=>`<tr><td>${new Date(r.at).toLocaleString()}</td><td>${esc(r.actionName)}</td><td>${esc(r.system)}</td><td>${r.allowed?'AUTHORIZED · DEMO':'BLOCKED'}</td><td><code>${r.hash}</code></td></tr>`).join('')||'<tr><td colspan="5">No action receipt yet.</td></tr>'}</tbody></table></div></div>`;host.appendChild(box)}
function renderHandover(){const h=overlay.handover;section('handover').innerHTML=header('17 · Successor-to-successor handover','The incumbent remains part of the safety architecture','Replacement proceeds through known-good freeze, shadow, minimum state, reversible canary, comparison, basis-risk update and explicit disposition.',`<button class="button" data-gss-action="run-handover">Run reversible handover</button><button class="button danger" data-gss-action="rollback-handover">Rollback</button>`)+`<div class="grid-2"><div class="card"><h2>Handover constitution</h2><label class="field"><span>Fallback owner</span><input id="handoverFallbackOwner" value="${esc(h.fallbackOwner)}"></label><label class="field"><span>Response time (minutes)</span><input id="handoverResponse" type="number" value="${h.responseMinutes}"></label><div class="gss-handover">${h.steps.map((s,i)=>`<div class="${s.status}"><b>${i+1}</b><span>${esc(s.name)}</span>${statusBadge(s.status)}</div>`).join('')}</div></div><div class="card highlight"><h2>Canary result</h2>${metric(h.predictedGain==null?'—':round(h.predictedGain,2),'Gym-predicted gain')}${metric(h.observedGain==null?'—':round(h.observedGain,2),'Canary-observed gain')}${metric(h.basisGap==null?'—':round(h.basisGap,2),'Basis gap')}${metric(h.decision||'Not run','Disposition')}<div class="callout ${h.decision==='expand'?'green':h.decision?'red':''}"><strong>${h.status.replaceAll('_',' ')}</strong><br>Authority expands only after the canary, not because the fresh-proof score was attractive.</div></div></div>`}
function renderLedger(){const l=deriveLedger();section('ledger').innerHTML=header('18 · Sovereign Gym Ledger','The executable mission environment is a corporate asset','Models may be rented and replaced. The institution should control the mission constitution, environment versions, protected cases, proof receipts, portability and continuity memory.',`<button class="button" data-gss-action="save-ledger">Save and evaluate ledger</button>`)+`<div class="metrics-grid">${metric(l.maturity,'Gym maturity')}${metric(new Intl.NumberFormat('en-CA',{style:'currency',currency:'CAD',maximumFractionDigits:0}).format(l.sovereigntyPremium),'Sovereignty premium')}${metric(l.environmentVersions.length,'Environment versions')}${metric(l.proofHistory.length,'Proof releases')}</div><div class="grid-2"><div class="card"><h2>Sovereignty controls</h2>${[['rightsCleared','Rights-cleared mission asset'],['portableAPI','Portable environment API'],['independentCustody','Independent proof custody'],['protectedCases','Protected cases and seeds'],['providerNeutral','Provider-neutral architecture'],['chronicleIntegrated','Chronicle integration'],['requalificationEnabled','Requalification enabled'],['eventSourced','Event-sourced lineage']].map(([id,label])=>`<label class="check gss-ledger-check"><input id="ledger_${id}" type="checkbox" ${l[id]?'checked':''}> ${label}</label>`).join('')}</div><div class="card"><h2>Economic asset model</h2>${[['maintenanceCost','Annual maintenance cost'],['requalificationCost','Requalification cost'],['substitutionValue','Provider substitution value'],['holdUpReduction','Vendor hold-up reduction'],['learningReuseValue','Institutional learning reuse']].map(([id,label])=>`<label class="field"><span>${label}</span><input id="ledger_${id}" type="number" value="${l[id]}"></label>`).join('')}</div></div><div class="grid-2" style="margin-top:16px"><div class="card"><h2>Environment lineage</h2><div class="timeline">${l.environmentVersions.slice().reverse().map(v=>`<div class="timeline-item"><b>${esc(v.version)}</b><small>${new Date(v.addedAt).toLocaleString()}</small><p><code>${v.hash}</code></p></div>`).join('')||'<div class="callout">No environment version recorded.</div>'}</div></div><div class="card"><h2>Proof lineage</h2><div class="timeline">${l.proofHistory.slice().reverse().map(v=>`<div class="timeline-item"><b>${esc(v.candidate)} · ${v.pass?'PASS':'FAIL'}</b><small>${new Date(v.at).toLocaleString()}</small><p>${esc(v.environment)}</p></div>`).join('')||'<div class="callout">No proof release recorded.</div>'}</div></div></div>`}
function renderExportEnhancement(){const host=section('export');if(!host)return;const existing=host.querySelector('.gss-export-extension');if(existing)existing.remove();const box=document.createElement('div');box.className='gss-export-extension';box.innerHTML=`<div class="section-divider"><span>v4 Gym · Specialist ASI · Successor Ω records</span></div><div class="export-grid"><div class="export-card"><div class="export-icon">Ω</div><h3>Complete GSSO constitutional record</h3><p>Nested Gyms, Gym families, mission physics, Alpha/Beta/Spread, decisions, Authority Lattice, handover, ledger and Successor Ω.</p><button class="button small" data-gss-action="export-gsso">Download JSON</button></div><div class="export-card"><div class="export-icon">ZIP</div><h3>GSSO v4 Mission Pack</h3><p>Complete v4 records in one locally generated ZIP, additive to the inherited Mission Pack.</p><button class="button small" data-gss-action="export-gsso-pack">Download ZIP</button></div><div class="export-card"><div class="export-icon">P</div><h3>Authoritative architecture paper</h3><p>Gym, Specialist ASI & Successor Ω — authoritative publication edition.</p><a class="button small secondary" href="../research/current/GoalOS_Unified_Verified_Succession_Institution_v6_0_0_WEB.pdf" target="_blank">Open paper</a></div><div class="export-card"><div class="export-icon">F</div><h3>Foundational architecture paper</h3><p>The Executable Succession Institution.</p><a class="button small secondary" href="../research/current/GoalOS_Unified_Verified_Succession_Institution_v6_0_0_WEB.pdf" target="_blank">Open paper</a></div></div>`;host.appendChild(box)}

const FR_TEXT=Object.freeze({
'01 · Categorical invariant':'01 · Invariant catégoriel',
'Three different institutional objects answer three different questions. A score cannot collapse environment, candidate and admitted authority into one object.':'Trois objets institutionnels distincts répondent à trois questions distinctes. Un score ne peut confondre l’environnement, le candidat et l’autorité admise en un seul objet.',
'MISSION GYM':'GYM DE MISSION',
'Makes the mission executable, repeatable, adversarial and measurable.':'Rend la mission exécutable, répétable, adversariale et mesurable.',
'Can this architecture perform?':'Cette architecture peut-elle accomplir la mission?',
'SPECIALIST ASI':'ASI SPÉCIALISTE',
'The complete frozen candidate that proves superiority on protected fresh work.':'Le candidat complet et gelé qui prouve sa supériorité sur du travail frais protégé.',
'Which candidate wins?':'Quel candidat l’emporte?',
'SUCCESSOR Ω':'SUCCESSEUR Ω',
'The proven candidate after accountable admission, identity, memory, rollback and scoped authority.':'Le candidat prouvé après admission responsable, identité, mémoire, retour arrière et autorité circonscrite.',
'What authority is justified?':'Quelle autorité est justifiée?',
'NO AUTOMATIC AUTHORITY':'AUCUNE AUTORITÉ AUTOMATIQUE',
'Mission dominance does not imply consequential authority. Training rewards teach. Protected proof decides. Accountable authority admits.':'La dominance de mission n’implique aucune autorité conséquente. Les récompenses d’entraînement enseignent. La preuve protégée décide. L’autorité responsable admet.',
'The complete Successor is larger than the policy':'Le Successeur complet dépasse la politique',
'Policy portfolio':'Portefeuille de politiques',
'World model & evidence graph':'Modèle du monde et graphe de preuve',
'Tools, retrieval & deterministic systems':'Outils, récupération et systèmes déterministes',
'Critic & independent verifier':'Critique et vérificateur indépendant',
'Human roles & workflow integration':'Rôles humains et intégration du flux de travail',
'Identity, proof-carrying action & rollback':'Identité, action porteuse de preuve et retour arrière',
'Chronicle, rights, portability & requalification':'Chronicle, droits, portabilité et requalification',
'Current local state':'État local actuel',
'Constitutional state':'État constitutionnel',
'Exact candidate identity':'Identité exacte du candidat',
'Current authority level':'Niveau d’autorité actuel',
'Not frozen':'Non gelé',
'CANDIDATE INSTITUTION':'INSTITUTION CANDIDATE',
'MISSION-SOVEREIGN · DEMO ADMISSION':'SOUVERAIN POUR LA MISSION · ADMISSION DE DÉMONSTRATION',
'Browser-local deterministic institutional simulation and reference implementation. No real-world Specialist ASI, Mission Alpha, customer value, professional authority or production authority is established.':'Simulation institutionnelle déterministe locale au navigateur et implémentation de référence. Aucune ASI spécialiste réelle, aucun Alpha de mission, aucune valeur client, autorité professionnelle ou autorité de production n’est établi.',
'03 · Mission physics':'03 · Physique de mission',
'Reality gap, observability, controllability and stability':'Écart à la réalité, observabilité, contrôlabilité et stabilité',
'The Gym is useful only when its state, sensors, actions, dynamics and evaluation remain sufficiently faithful to the real mission.':'Le Gym n’est utile que si son état, ses capteurs, ses actions, sa dynamique et son évaluation demeurent suffisamment fidèles à la mission réelle.',
'Recalculate physics':'Recalculer la physique',
'Observability debt':'Dette d’observabilité',
'Controllability debt':'Dette de contrôlabilité',
'Sensor mismatch':'Écart des capteurs',
'Actuation mismatch':'Écart d’actionnement',
'Viability':'Viabilité',
'Stability':'Stabilité',
'Reality-gap assumptions':'Hypothèses d’écart à la réalité',
'Scenario fidelity':'Fidélité des scénarios',
'Protected-case independence':'Indépendance des cas protégés',
'Evaluator uncertainty':'Incertitude de l’évaluateur',
'Canary calibration':'Étalonnage du canari',
'Basis-risk underwriting':'Souscription du risque de base',
'Gym basis risk':'Risque de base du Gym',
'Basis-risk reserve':'Réserve de risque de base',
'Viability kernel proxy':'Proxy du noyau de viabilité',
'Lyapunov-style stability proxy':'Proxy de stabilité de type Lyapunov',
'Controlled basis risk':'Risque de base maîtrisé',
'Material basis risk':'Risque de base important',
'High basis risk':'Risque de base élevé',
'Demonstrated advantage must remain positive after the Gym-to-reality reserve is charged.':'L’avantage démontré doit demeurer positif après imputation de la réserve Gym-réalité.',
'04 · Three nested Gyms':'04 · Trois Gyms imbriqués',
'Mission competence is not institutional competence':'La compétence de mission n’est pas la compétence institutionnelle',
'The inner task, the operating institution and the succession decision fail in different ways.':'La tâche interne, l’institution opérante et la décision de succession échouent de façons différentes.',
'Run all three Gyms':'Exécuter les trois Gyms',
'OUTER':'EXTERNE',
'MIDDLE':'INTERMÉDIAIRE',
'INNER':'INTERNE',
'Institutional score':'Score institutionnel',
'Quality-diversity archive':'Archive qualité-diversité',
'Negative Capability Graph':'Graphe de capacité négative',
'Decision separation':'Séparation des décisions',
'Requalification':'Requalification',
'Handover':'Passation',
'Fewer than four institutional decisions signed':'Moins de quatre décisions institutionnelles signées',
'Producer–critic–verifier separation':'Séparation producteur–critique–vérificateur',
'Authority stress':'Test de résistance de l’autorité',
'Evidence custody':'Garde de la preuve',
'Degraded mode':'Mode dégradé',
'Gymmability vector':'Vecteur de Gymmabilité',
'Matched formation':'Formation appariée',
'Fresh transfer result':'Résultat frais de transfert',
'05 · Five environment families':'05 · Cinq familles d’environnements',
'The Gym that teaches must not be the Gym that certifies':'Le Gym qui enseigne ne doit pas être celui qui certifie',
'Formation, attack, transfer, fresh proof and production-connected requalification require different visibility and custody.':'La formation, l’attaque, le transfert, la preuve fraîche et la requalification connectée à la production exigent des visibilités et gardes distinctes.',
'Conformance-check all families':'Vérifier la conformité de toutes les familles',
'Release':'Version',
'Custodian':'Gardien',
'Visibility':'Visibilité',
'Promotion weight':'Poids de promotion',
'Run conformance':'Exécuter la conformité',
'Formation Custodian':'Gardien de formation',
'Independent Red-Team Custodian':'Gardien indépendant de l’équipe rouge',
'Transfer Evaluation Custodian':'Gardien de l’évaluation de transfert',
'Independent Proof Custodian':'Gardien indépendant de la preuve',
'Operational Assurance Custodian':'Gardien de l’assurance opérationnelle',
'Exact environment release':'Version exacte de l’environnement',
'Typed custody boundary':'Frontière de garde typée',
'Matched candidate evidence':'Preuve appariée du candidat',
'Failure-preserving memory':'Mémoire préservant les échecs',
'Authority challenge':'Défi d’autorité',
'Transfer evidence':'Preuve de transfert',
'Frozen exact challenger':'Challenger exact gelé',
'Protected proof completed':'Preuve protégée terminée',
'Formation roles excluded':'Rôles de formation exclus',
'Accountable admission exists':'Admission responsable existante',
'Reversible handover executed':'Passation réversible exécutée',
'locked':'verrouillé',
'not run':'non exécuté',
'passed':'réussi',
'conformed':'conforme',
'needs work':'travail requis',
'unsigned':'non signée',
'signed':'signée',
'revoked':'révoquée',
'pending':'en attente',
'completed':'terminée',
'warning':'avertissement',
'not started':'non commencée',
'08 · Alpha, Beta and Spread':'08 · Alpha, Bêta et Spread',
'From rented intelligence to robust mission advantage':'De l’intelligence louée à l’avantage robuste de mission',
'AI Beta is the common market baseline. Mission Beta is systematic dependence. The Sovereignty Spread is the robust net opportunity. Mission Alpha exists only after fresh proof.':'La Bêta IA est la référence commune du marché. La Bêta de mission est la dépendance systématique. Le Spread de souveraineté est l’opportunité nette robuste. L’Alpha de mission n’existe qu’après preuve fraîche.',
'Recalculate':'Recalculer',
'AI BETA':'BÊTA IA',
'Broadly rentable capability.':'Capacité largement louable.',
'MISSION ADVANTAGE SPREAD':'SPREAD D’AVANTAGE DE MISSION',
'Gross reachable opportunity.':'Opportunité brute atteignable.',
'SOVEREIGNTY SPREAD':'SPREAD DE SOUVERAINETÉ',
'Net opportunity after burdens and reserves.':'Opportunité nette après charges et réserves.',
'MISSION ALPHA':'ALPHA DE MISSION',
'UNPROVEN':'NON PROUVÉ',
'Fresh-proof residual advantage.':'Avantage résiduel après preuve fraîche.',
'Mission Beta factor exposures':'Expositions factorielles de la Bêta de mission',
'Provider':'Fournisseur',
'Data':'Données',
'Verifier':'Vérificateur',
'Human':'Humain',
'Integration':'Intégration',
'Regime':'Régime',
'Complete-denominator charges':'Charges du dénominateur complet',
'PROVEN IN SIMULATION':'PROUVÉ EN SIMULATION',
'Gross Spread':'Spread brut',
'Sovereignty Spread':'Spread de souveraineté',
'Mission Alpha':'Alpha de mission',
'not established':'non établi',
'10 · Institutional decisions':'10 · Décisions institutionnelles',
'Underwrite ≠ Execute ≠ Accept ≠ Admit ≠ Allocate':'Souscrire ≠ Exécuter ≠ Accepter ≠ Admettre ≠ Allouer',
'Each decision creates a different authority. The candidate owns none of them.':'Chaque décision crée une autorité distincte. Le candidat n’en contrôle aucune.',
'Signer':'Signataire',
'Demonstration Principal':'Principal de démonstration',
'Exact scope':'Portée exacte',
'Sign':'Signer',
'Revoke':'Révoquer',
'SEIZE has a decision':'SEIZE dispose d’une décision',
'Mission Constitution frozen':'Constitution de mission gelée',
'Bounded AGI Jobs compiled':'Tâches AGI bornées compilées',
'Fresh proof passed':'Preuve fraîche réussie',
'Exact challenger manifest':'Manifeste exact du challenger',
'Accountable demo admission exists':'Admission responsable de démonstration existante',
'Authority stress passed':'Test de résistance de l’autorité réussi',
'ADMIT decision signed':'Décision ADMETTRE signée',
'Accepted learning or successor generation':'Apprentissage accepté ou génération de successeur',
'16 · Successor Ω':'16 · Successeur Ω',
'The proven winner admitted to act, compound and renew':'Le gagnant prouvé admis à agir, composer et se renouveler',
'Successor Ω is reserved for an admitted institution that also preserves memory, requalifies itself, generates challengers and prepares its next successor without self-authorizing.':'Successeur Ω est réservé à une institution admise qui préserve aussi la mémoire, se requalifie, génère des challengers et prépare son prochain successeur sans s’auto-autoriser.',
'Constitute Successor Ω · Demo':'Constituer Successeur Ω · Démo',
'Candidate':'Candidat',
'Verified intelligence':'Intelligence vérifiée',
'Mission-Dominant Specialist ASI':'ASI spécialiste dominante pour la mission',
'Mission-Sovereign Successor':'Successeur souverain pour la mission',
'Ω qualification':'Qualification Ω',
'Mission Gym exists as a versioned executable environment':'Le Gym de mission existe comme environnement exécutable versionné',
'Mission ≠ candidate ≠ admitted institution is preserved':'Mission ≠ candidat ≠ institution admise est préservé',
'One exact frozen candidate passed fresh proof':'Un candidat exact gelé a réussi la preuve fraîche',
'Accountable admission and scoped authority exist':'Une admission responsable et une autorité circonscrite existent',
'Mission, Institutional and Succession Gyms passed':'Les Gyms de mission, institutionnel et de succession ont réussi',
'Five Gym families are independently conformed':'Les cinq familles de Gym sont conformes de façon indépendante',
'Underwrite, Execute, Accept and Admit are separately signed':'Souscrire, Exécuter, Accepter et Admettre sont signés séparément',
'Authority Lattice has fresh evidence and incident capacity':'La grille d’autorité dispose de preuve fraîche et d’une capacité d’incident',
'Reversible successor handover completed':'Passation réversible du successeur terminée',
'Sovereign Gym Ledger reached at least G6':'Le registre du Gym souverain a atteint au moins G6',
'Recursive Foundry produced at least one later generation':'La Fonderie récursive a produit au moins une génération ultérieure',
'Recursion firewall remains non-bypassable':'Le pare-feu de récursion demeure incontournable',
'Recursion firewall':'Pare-feu de récursion',
'LOCKED':'VERROUILLÉ',
'protected Cases External':'cas protégés externes',
'self Certification Prohibited':'auto-certification interdite',
'self Installation Prohibited':'auto-installation interdite',
'authority Inheritance Prohibited':'héritage d’autorité interdit',
'constitution Rewrite Prohibited':'réécriture de la constitution interdite',
'No self-authorization':'Aucune auto-autorisation',
'The successor may propose descendants. It may not inspect protected proof, certify itself, install itself, inherit authority or rewrite the Mission Constitution.':'Le successeur peut proposer des descendants. Il ne peut inspecter la preuve protégée, se certifier, s’installer, hériter de l’autorité ni réécrire la Constitution de mission.',
'17 · Successor-to-successor handover':'17 · Passation de successeur à successeur',
'The incumbent remains part of the safety architecture':'L’incumbent demeure dans l’architecture de sûreté',
'Replacement proceeds through known-good freeze, shadow, minimum state, reversible canary, comparison, basis-risk update and explicit disposition.':'Le remplacement passe par le gel de l’état connu, le mode fantôme, l’état minimal, un canari réversible, la comparaison, la mise à jour du risque de base et une disposition explicite.',
'Run reversible handover':'Exécuter la passation réversible',
'Rollback':'Retour arrière',
'Handover constitution':'Constitution de passation',
'Fallback owner':'Responsable du repli',
'Response time (minutes)':'Délai de réponse (minutes)',
'Freeze incumbent known-good state':'Geler l’état connu et fiable de l’incumbent',
'Run challenger in matched shadow mode':'Exécuter le challenger en mode fantôme apparié',
'Transfer minimum canary state':'Transférer l’état minimal du canari',
'Assign fallback owner and response time':'Assigner le responsable du repli et le délai de réponse',
'Execute one reversible bounded action':'Exécuter une action bornée réversible',
'Compare real-like outcomes with Gym prediction':'Comparer les résultats quasi réels à la prédiction du Gym',
'Update Gym basis-risk estimate':'Mettre à jour l’estimation du risque de base du Gym',
'Expand, hold, repair, rollback or revoke':'Étendre, maintenir, réparer, revenir en arrière ou révoquer',
'Preserve handover evidence in Chronicle':'Préserver la preuve de passation dans Chronicle',
'Canary result':'Résultat canari',
'Gym-predicted gain':'Gain prédit par le Gym',
'Canary-observed gain':'Gain observé par le canari',
'Basis gap':'Écart de base',
'Disposition':'Disposition',
'Authority expands only after the canary, not because the fresh-proof score was attractive.':'L’autorité ne s’étend qu’après le canari, jamais parce que le score de preuve fraîche était attrayant.',
'18 · Sovereign Gym Ledger':'18 · Registre du Gym souverain',
'The executable mission environment is a corporate asset':'L’environnement exécutable de mission est un actif d’entreprise',
'Models may be rented and replaced. The institution should control the mission constitution, environment versions, protected cases, proof receipts, portability and continuity memory.':'Les modèles peuvent être loués et remplacés. L’institution doit contrôler la constitution de mission, les versions d’environnement, les cas protégés, les reçus de preuve, la portabilité et la mémoire de continuité.',
'Save and evaluate ledger':'Enregistrer et évaluer le registre',
'Gym maturity':'Maturité du Gym',
'Sovereignty premium':'Prime de souveraineté',
'Environment versions':'Versions d’environnement',
'Proof releases':'Versions de preuve',
'Sovereignty controls':'Contrôles de souveraineté',
'Rights-cleared mission asset':'Actif de mission aux droits clarifiés',
'Portable environment API':'API d’environnement portable',
'Independent proof custody':'Garde indépendante de la preuve',
'Protected cases and seeds':'Cas et germes protégés',
'Provider-neutral architecture':'Architecture neutre aux fournisseurs',
'Chronicle integration':'Intégration à Chronicle',
'Requalification enabled':'Requalification activée',
'Event-sourced lineage':'Lignée sourcée par événements',
'Economic asset model':'Modèle économique de l’actif',
'Annual maintenance cost':'Coût annuel de maintenance',
'Requalification cost':'Coût de requalification',
'Provider substitution value':'Valeur de substitution du fournisseur',
'Vendor hold-up reduction':'Réduction du risque de prise en otage fournisseur',
'Institutional learning reuse':'Réutilisation de l’apprentissage institutionnel',
'Environment lineage':'Lignée des environnements',
'Proof lineage':'Lignée des preuves',
'Separate':'Séparer',
'Reality':'Réalité',
'Institution':'Institution',
'Custody':'Garde',
'Underwrite':'Souscrire',
'Constitute':'Constituer',
'Transfer':'Transférer',
'Own':'Posséder',
'Bound':'Borner',
'Compound':'Composer',
'Renew':'Renouveler',
'Package':'Emballer',
'v4 Gym · Specialist ASI · Successor Ω records':'Dossiers v4 Gym · ASI spécialiste · Successeur Ω',
'Complete GSSO constitutional record':'Dossier constitutionnel GSSO complet',
'Nested Gyms, Gym families, mission physics, Alpha/Beta/Spread, decisions, Authority Lattice, handover, ledger and Successor Ω.':'Gyms imbriqués, familles de Gym, physique de mission, Alpha/Bêta/Spread, décisions, grille d’autorité, passation, registre et Successeur Ω.',
'Download JSON':'Télécharger le JSON',
'GSSO v4 Mission Pack':'Mission Pack GSSO v4',
'Complete v4 records in one locally generated ZIP, additive to the inherited Mission Pack.':'Dossiers v4 complets dans un ZIP généré localement, ajouté au Mission Pack hérité.',
'Download ZIP':'Télécharger le ZIP',
'Authoritative architecture paper':'Article d’architecture faisant autorité',
'Gym, Specialist ASI & Successor Ω — authoritative publication edition.':'Gym, ASI spécialiste et Successeur Ω — édition de publication faisant autorité.',
'Open paper':'Ouvrir l’article',
'Foundational architecture paper':'Article d’architecture fondateur',
'The Executable Succession Institution.':'L’Institution de succession exécutable.',
'not conformed':'non conforme',
'candidate':'candidat',
'mission sovereign successor':'successeur souverain pour la mission',
'mission dominant specialist asi':'ASI spécialiste dominante pour la mission',
'No frozen candidate':'Aucun candidat gelé',
'not constituted':'non constitué',
'unproven':'non prouvé',
'not admitted':'non admis',
'not_admitted':'non admis',
'fresh proof pass':'preuve fraîche réussie',
'fresh-proof-pass':'preuve fraîche réussie',
'simulated admitted':'admis en simulation',
'simulated_admitted':'admis en simulation',
});
function translateV4(){
  if(lang()!=='fr')return;
  const ids=['triad','physics','nested','families','alpha','decisions','omega','handover','ledger','export'];
  const roots=ids.map(id=>section(id)).filter(Boolean);
  qa('#nav [data-section="triad"],#nav [data-section="physics"],#nav [data-section="nested"],#nav [data-section="families"],#nav [data-section="alpha"],#nav [data-section="decisions"],#nav [data-section="omega"],#nav [data-section="handover"],#nav [data-section="ledger"]').forEach(x=>roots.push(x));
  const translateCore=core=>{
    if(FR_TEXT[core])return FR_TEXT[core];
    let m=core.match(/^Run (.+)$/);if(m)return `Exécuter ${FR_TEXT[m[1]]||m[1]}`;
    m=core.match(/^Identity:\s*$/);if(m)return 'Identité :';
    m=core.match(/^Version:\s*$/);if(m)return 'Version :';
    m=core.match(/^ · Proof:\s*$/);if(m)return ' · Preuve :';
    m=core.match(/^ · Authority:\s*$/);if(m)return ' · Autorité :';
    if(core.includes('Version:')||core.includes('Proof:')||core.includes('Authority:'))return core.replace('Version:','Version :').replace('Proof:','Preuve :').replace('Authority:','Autorité :').replace('not_admitted','non admis').replace('fresh-proof-pass','preuve fraîche réussie').replace('simulated_admitted','admis en simulation');
    m=core.match(/^([A-Za-z]+): ([0-9.]+) utility points$/);if(m){const names={completeBurden:'charge complète',beta:'bêta',basis:'base',proof:'preuve',authority:'autorité'};return `${names[m[1]]||m[1]} : ${m[2]} points d’utilité`}
    m=core.match(/^Gross Spread ([0-9.-]+) → Sovereignty Spread ([0-9.-]+) → Mission Alpha (.+)\.$/);if(m)return `Spread brut ${m[1]} → Spread de souveraineté ${m[2]} → Alpha de mission ${FR_TEXT[m[3]]||m[3]}.`;
    return core;
  };
  for(const root of roots){
    const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT,{acceptNode(n){return n.parentElement&& !['SCRIPT','STYLE','CODE','PRE','TEXTAREA'].includes(n.parentElement.tagName)&&n.nodeValue.trim()?NodeFilter.FILTER_ACCEPT:NodeFilter.FILTER_REJECT}});
    const nodes=[];while(walker.nextNode())nodes.push(walker.currentNode);
    for(const n of nodes){const raw=n.nodeValue,lead=raw.match(/^\s*/)[0],trail=raw.match(/\s*$/)[0],c=raw.trim(),z=translateCore(c);if(z!==c)n.nodeValue=lead+z+trail}
    root.querySelectorAll('[placeholder]').forEach(e=>{if(FR_TEXT[e.placeholder])e.placeholder=FR_TEXT[e.placeholder]});
    root.querySelectorAll('input[type="text"]').forEach(e=>{if(FR_TEXT[e.value])e.value=FR_TEXT[e.value]});
  }
}

function render(){
  if(!section('triad'))return;
  renderTriad();renderPhysics();renderNested();renderFamilies();renderAlpha();renderDecisions();renderOmega();renderAuthorityLatticeEnhancement();renderHandover();renderLedger();renderExportEnhancement();
  updateNavLabels();translateV4();
}
function updateNavLabels(){const map={triad:['Triad','Triade'],physics:['Mission Physics','Physique de mission'],nested:['Nested Gyms','Gyms imbriqués'],families:['Gym Families','Familles de Gym'],alpha:['Alpha · Beta · Spread','Alpha · Bêta · Spread'],decisions:['Five Decisions','Cinq décisions'],omega:['Successor Ω','Successeur Ω'],handover:['Handover','Passation'],ledger:['Gym Ledger','Registre du Gym']};Object.entries(map).forEach(([id,[en,fr]])=>{const e=q(`[data-section="${id}"] b`);if(e)e.textContent=t(en,fr)});const sub=q('#editionText');if(sub)sub.textContent=`${RELEASE.name} · v${RELEASE.version}`;const chip=q('#projectChip');if(chip&&chip.textContent==='GSX3')chip.textContent='UVSI1'}

function savePhysicsFromDOM(){['scenarioFidelity','protectedCaseIndependence','evaluatorUncertainty','canaryCalibration'].forEach(id=>{const e=q(`#physics_${id}`);if(e)overlay.physics[id]=Number(e.value)});derivePhysics();deriveEconomics();receipt('MISSION_PHYSICS',{physics:overlay.physics});save();render();}
function handleAction(action,target){
  if(action==='save-physics')savePhysicsFromDOM();
  if(action==='run-nested')runNested(target.dataset.kind);
  if(action==='run-all-nested')runAllNested();
  if(action==='run-family')runFamily(target.dataset.family);
  if(action==='run-all-families')runAllFamilies();
  if(action==='recalculate-alpha'){deriveEconomics();receipt('ALPHA_UNDERWRITING',{economics:overlay.economics});save();render();}
  if(action==='sign-decision')signDecision(target.dataset.decision);
  if(action==='revoke-decision')revokeDecision(target.dataset.decision);
  if(action==='evaluate-authority')evaluateAuthority();
  if(action==='simulate-action')simulateAction();
  if(action==='constitute-omega')constituteOmega();
  if(action==='revoke-omega')revokeOmega();
  if(action==='run-handover'){const fo=q('#handoverFallbackOwner'),rt=q('#handoverResponse');if(fo)overlay.handover.fallbackOwner=fo.value;if(rt)overlay.handover.responseMinutes=Number(rt.value||15);runHandover()}
  if(action==='rollback-handover')rollbackHandover();
  if(action==='save-ledger')saveLedger();
  if(action==='export-gsso')download(`${safe(core()?.mission?.name)}_GSSO_v4_Constitutional_Record.json`,JSON.stringify(snapshot(),null,2),'application/json');
  if(action==='export-gsso-pack')exportPack();
}
function download(name,data,type='text/plain'){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([data],{type}));a.download=name;document.body.appendChild(a);a.click();setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove()},800)}
const crcTable=(()=>{const t=new Uint32Array(256);for(let n=0;n<256;n++){let c=n;for(let k=0;k<8;k++)c=(c&1)?0xedb88320^(c>>>1):c>>>1;t[n]=c>>>0}return t})();
function crc32(bytes){let c=0xffffffff;for(const b of bytes)c=crcTable[(c^b)&255]^(c>>>8);return(c^0xffffffff)>>>0}
function u16(n){return new Uint8Array([n&255,(n>>>8)&255])}function u32(n){return new Uint8Array([n&255,(n>>>8)&255,(n>>>16)&255,(n>>>24)&255])}
function concat(arrays){const length=arrays.reduce((s,a)=>s+a.length,0),out=new Uint8Array(length);let o=0;arrays.forEach(a=>{out.set(a,o);o+=a.length});return out}
function zip(files){const enc=new TextEncoder(),locals=[],centrals=[];let offset=0;const d=new Date(),time=(d.getHours()<<11)|(d.getMinutes()<<5)|(d.getSeconds()>>1),date=((d.getFullYear()-1980)<<9)|((d.getMonth()+1)<<5)|d.getDate();for(const file of files){const name=enc.encode(file.name),data=file.data instanceof Uint8Array?file.data:enc.encode(String(file.data)),crc=crc32(data),local=concat([u32(0x04034b50),u16(20),u16(0),u16(0),u16(time),u16(date),u32(crc),u32(data.length),u32(data.length),u16(name.length),u16(0),name,data]);locals.push(local);centrals.push(concat([u32(0x02014b50),u16(20),u16(20),u16(0),u16(0),u16(time),u16(date),u32(crc),u32(data.length),u32(data.length),u16(name.length),u16(0),u16(0),u16(0),u16(0),u32(0),u32(offset),name]));offset+=local.length}const central=concat(centrals),end=concat([u32(0x06054b50),u16(0),u16(0),u16(files.length),u16(files.length),u32(central.length),u32(offset),u16(0)]);return new Blob([...locals,central,end],{type:'application/zip'})}
function packFiles(){const s=snapshot(),name=safe(core()?.mission?.name);return [
  {name:'README.txt',data:`GoalOS Ω + SEIZE — Unified Verified Succession Institution · Sovereign Mission Gym × Specialist ASI × Successor Ω ${RELEASE.id}\nMission: ${core()?.mission?.name||'Untitled'}\nState: ${s.successorOmega.status}\n\n${RELEASE.claimBoundary}\n`},
  {name:`${name}_GSSO_Constitutional_Record.json`,data:JSON.stringify(s,null,2)},
  {name:`${name}_Nested_Gyms.json`,data:JSON.stringify(s.nested,null,2)},
  {name:`${name}_Gym_Families.json`,data:JSON.stringify(s.families,null,2)},
  {name:`${name}_Mission_Physics.json`,data:JSON.stringify(s.physics,null,2)},
  {name:`${name}_Alpha_Beta_Spread.json`,data:JSON.stringify(s.economics,null,2)},
  {name:`${name}_Institutional_Decisions.json`,data:JSON.stringify(s.decisions,null,2)},
  {name:`${name}_Authority_Lattice.json`,data:JSON.stringify(s.authorityLattice,null,2)},
  {name:`${name}_Successor_Omega.json`,data:JSON.stringify(s.successorOmega,null,2)},
  {name:`${name}_Handover_Record.json`,data:JSON.stringify(s.handover,null,2)},
  {name:`${name}_Sovereign_Gym_Ledger.json`,data:JSON.stringify(s.ledger,null,2)},
  {name:`${name}_Proof_Receipts.json`,data:JSON.stringify(s.receipts,null,2)},
  {name:`${name}_Core_v4_Project_Snapshot.json`,data:JSON.stringify(core(),null,2)},
  {name:'CLAIM_BOUNDARY.txt',data:RELEASE.claimBoundary}
]}
function exportPack(){const name=safe(core()?.mission?.name);const b=zip(packFiles());const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=`${name}_GoalOS_GSSO_v4_Mission_Pack.zip`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);receipt('GSSO_MISSION_PACK_EXPORTED',{files:packFiles().length});}
function snapshot(){derivePhysics();deriveEconomics();deriveLedger();deriveOmega();return {schema:'GoalOS.GymSpecialistASISuccessorOmega.v4',release:RELEASE.id,generatedAt:now(),projectId:core()?.meta?.projectId,mission:core()?.mission?.name,coreFingerprint:fingerprintCore(),nested:deep(overlay.nested),families:deep(overlay.families),physics:deep(overlay.physics),economics:deep(overlay.economics),decisions:deep(overlay.decisions),authorityLattice:deep(overlay.authorityLattice),successorOmega:deep(overlay.successorOmega),handover:deep(overlay.handover),ledger:deep(overlay.ledger),receipts:deep(overlay.receipts),claimBoundary:RELEASE.claimBoundary}}
function missionPackFiles(prefix='GoalOS'){return packFiles().map(f=>({name:`GSSO_v4/${f.name}`,data:f.data}))}

function installDOM(){
  const nav=q('#nav');if(!nav||q('[data-section="triad"]'))return;
  const buttons={
    triad:`<button class="nav-item" data-section="triad"><span>01</span><b>Triad</b><small>Separate</small></button>`,
    physics:`<button class="nav-item" data-section="physics"><span>03</span><b>Mission Physics</b><small>Reality</small></button>`,
    nested:`<button class="nav-item" data-section="nested"><span>04</span><b>Nested Gyms</b><small>Institution</small></button>`,
    families:`<button class="nav-item" data-section="families"><span>05</span><b>Gym Families</b><small>Custody</small></button>`,
    alpha:`<button class="nav-item" data-section="alpha"><span>08</span><b>Alpha · Beta · Spread</b><small>Underwrite</small></button>`,
    decisions:`<button class="nav-item" data-section="decisions"><span>10</span><b>Five Decisions</b><small>Separate</small></button>`,
    omega:`<button class="nav-item" data-section="omega"><span>16</span><b>Successor Ω</b><small>Constitute</small></button>`,
    handover:`<button class="nav-item" data-section="handover"><span>17</span><b>Handover</b><small>Transfer</small></button>`,
    ledger:`<button class="nav-item" data-section="ledger"><span>18</span><b>Gym Ledger</b><small>Own</small></button>`
  };
  const after=(selector,html)=>q(selector)?.insertAdjacentHTML('afterend',html);
  after('[data-section="welcome"]',buttons.triad);after('[data-section="mission"]',buttons.physics+buttons.nested+buttons.families);after('[data-section="gradient"]',buttons.alpha);after('[data-section="seize"]',buttons.decisions);after('[data-section="authority"]',buttons.omega+buttons.handover+buttons.ledger);
  const numbers={mission:'02',manifold:'06',gradient:'07',seize:'09',jobs:'11',formation:'12',recursive:'13',proof:'14',authority:'15',chronicle:'19',requalify:'20',export:'21'};Object.entries(numbers).forEach(([id,n])=>{const e=q(`[data-section="${id}"] span`);if(e)e.textContent=n});
  const main=q('#main');['triad','physics','nested','families','alpha','decisions','omega','handover','ledger'].forEach(id=>{if(!section(id))main.insertAdjacentHTML('beforeend',`<section id="section-${id}" class="view"></section>`)});
}
function bind(){document.addEventListener('click',e=>{const a=e.target.closest('[data-gss-action]');if(a){e.preventDefault();handleAction(a.dataset.gssAction,a)}});setInterval(updateFromCore,900);document.addEventListener('goalos:gym-access',()=>setTimeout(updateFromCore,100));}
function init(){installDOM();bind();q('#languageButton')?.addEventListener('click',()=>setTimeout(render,0));derivePhysics();deriveEconomics();deriveLedger();deriveOmega();render();}

window.GoalOSGSSO={version:RELEASE.version,releaseId:RELEASE.id,snapshot,missionPackFiles,runAllNested,runAllFamilies,derivePhysics,deriveEconomics,deriveOmega,runHandover,exportPack,reset:()=>{overlay=defaultOverlay();save();render()}};
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
})();
