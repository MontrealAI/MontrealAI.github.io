(() => {
'use strict';

const CFG = window.GOALOS_CONFIG;
const CORE = () => window.GoalOSDemo?.getState?.() || null;
const $ = (s,r=document) => r.querySelector(s);
const $$ = (s,r=document) => [...r.querySelectorAll(s)];
const esc = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const nowISO = () => new Date().toISOString();
const clamp = (v,a=0,b=100) => Math.max(a,Math.min(b,Number(v)||0));
const fmt = value => new Intl.NumberFormat('en-CA',{maximumFractionDigits:1}).format(Number(value)||0);
const money = value => new Intl.NumberFormat('en-CA',{style:'currency',currency:'CAD',maximumFractionDigits:0}).format(Number(value)||0);
const shortAddress = value => value ? `${value.slice(0,6)}…${value.slice(-4)}` : '—';
const canonicalJSON = value => value===null||typeof value!=='object' ? JSON.stringify(value) : Array.isArray(value) ? `[${value.map(canonicalJSON).join(',')}]` : `{${Object.keys(value).sort().map(k=>`${JSON.stringify(k)}:${canonicalJSON(value[k])}`).join(',')}}`;
const downloadText = (name,text,type='text/plain') => { const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([text],{type})); a.download=name; document.body.appendChild(a); a.click(); setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove()},900); };
const toast = message => { const el=$('#toast'); if(!el) return; el.textContent=message; el.classList.remove('hidden'); clearTimeout(toast.timer); toast.timer=setTimeout(()=>el.classList.add('hidden'),3300); };
const modal = (title,html) => { $('#modalTitle').textContent=title; $('#modalBody').innerHTML=html; $('#modal').classList.remove('hidden'); };
const safeJson = value => { try{return JSON.parse(value)}catch{return null} };

const I18N = {
  en:{
    command:'Command Center', navigator:'Navigator Ω', book:'Successor Book', ai:'AI Studio', board:'Board & Metrics', evidence:'Evidence Room',
    accessGranted:'Access granted', offline:'Offline intelligence', secureAI:'Secure AI connected', locked:'Access locked'
  },
  fr:{
    command:'Centre de commandement', navigator:'Navigateur Ω', book:'Livre des successeurs', ai:'Studio IA', board:'Conseil et métriques', evidence:'Salle de preuve',
    accessGranted:'Accès accordé', offline:'Intelligence hors ligne', secureAI:'IA sécurisée connectée', locked:'Accès verrouillé'
  }
};
const lang = () => CORE()?.meta?.lang || 'en';
const tr = key => I18N[lang()]?.[key] || I18N.en[key] || key;

const PLATFORM_KEY = 'goalos_uvsi3_platform_v8_0_0';
const ACCESS_KEY = 'goalos_uvsi3_access_v8_8_0';
function defaultPlatform() {
  return {
    version:CFG.version,
    createdAt:nowISO(),updatedAt:nowISO(),
    navigator:{capabilityHalfLifeDays:120,adaptationLagDays:45,events:[
      {id:'evt-frontier',type:'Capability',title:'Frontier capability changed',impact:78,urgency:70,evidence:'New capability may alter the incumbent mission architecture.',status:'watch',at:nowISO()},
      {id:'evt-provider',type:'Provider',title:'Provider dependency review',impact:64,urgency:52,evidence:'Portability, evidence export and fallback require revalidation.',status:'watch',at:nowISO()},
      {id:'evt-proof',type:'Proof',title:'Fresh-proof window available',impact:86,urgency:82,evidence:'Representative protected work can now be purchased.',status:'underwrite',at:nowISO()}
    ]},
    book:{notes:'Maintain incumbent, immediate, challenger, reserve, hedge and retirement positions.'},
    evidence:[], claims:[],
    ai:{endpoint:CFG.ai.endpoint||'',mode:'offline',consent:false,lastAction:null,lastResult:null,history:[]},
    accessLog:[]
  };
}
function loadPlatform(){ try { return {...defaultPlatform(),...(JSON.parse(localStorage.getItem(PLATFORM_KEY)||'null')||{})}; } catch { return defaultPlatform(); } }
let platform=loadPlatform();
function savePlatform(){ platform.updatedAt=nowISO(); localStorage.setItem(PLATFORM_KEY,JSON.stringify(platform)); renderPlatform(); }

function updateAccessStatus(message,tone='neutral') {
  const el=$('#accessStatus'); if(!el)return;
  el.className=`access-status ${tone}`;
  el.innerHTML=`<span class="access-status-dot"></span><span>${esc(message)}</span>`;
}
function setAccessBusy(busy){ $$('.access-action').forEach(b=>b.disabled=busy); }
const normalizeAddress = value => /^0x[0-9a-fA-F]{40}$/.test(String(value||'')) ? String(value).toLowerCase() : null;
const encodeAddressWord = address => address.replace(/^0x/,'').toLowerCase().padStart(64,'0');
const decodeAddressWord = data => normalizeAddress(`0x${String(data).replace(/^0x/,'').slice(24,64)}`);
const decodeUintWord = (data,index=0) => BigInt(`0x${String(data).replace(/^0x/,'').slice(index*64,(index+1)*64)||'0'}`);
const hexUtf8 = text => '0x'+[...new TextEncoder().encode(text)].map(v=>v.toString(16).padStart(2,'0')).join('');
const randomHex = (bytes=16) => { const out=new Uint8Array(bytes); crypto.getRandomValues(out); return [...out].map(v=>v.toString(16).padStart(2,'0')).join(''); };
async function rpc(method,params=[]){ if(!window.ethereum) throw new Error('No injected Ethereum wallet was found. Open this page in a wallet-enabled browser.'); return window.ethereum.request({method,params}); }
async function ensureMainnet(){ let chain=await rpc('eth_chainId'); if(chain!==CFG.ethereumChainId){ try{ await rpc('wallet_switchEthereumChain',[{chainId:CFG.ethereumChainId}]); } catch(error){ throw new Error('Switch the wallet to Ethereum Mainnet, then try again.'); } chain=await rpc('eth_chainId'); } return chain; }
async function connectWallet(){ const accounts=await rpc('eth_requestAccounts'); const account=normalizeAddress(accounts?.[0]); if(!account) throw new Error('The wallet did not return a valid Ethereum address.'); await ensureMainnet(); return account; }
async function ethCall(to,data){ return rpc('eth_call',[{to,data},'latest']); }
async function tokenBalance(account){ const data='0x70a08231'+encodeAddressWord(account); return BigInt(await ethCall(CFG.token.contract,data)); }
function requiredTokenRaw(){ return BigInt(CFG.token.minimumWhole)*10n**BigInt(CFG.token.decimals); }
function validateClubLabel(label){ const value=String(label||'').trim().toLowerCase(); if(!/^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/.test(value)) throw new Error('Enter one ASCII label using letters, numbers or internal hyphens. Do not enter dots.'); return value; }
async function currentClubOwner(label){
  const fullName=`${validateClubLabel(label)}.${CFG.ens.suffix}`;
  const node=window.GoalOSCrypto.namehash(fullName);
  const ownerData='0x02571be3'+node.slice(2);
  const registryOwner=decodeAddressWord(await ethCall(CFG.ens.registry,ownerData));
  if(!registryOwner || /^0x0{40}$/.test(registryOwner)) return {fullName,node,registryOwner:null,effectiveOwner:null,wrapped:false,expiry:null};
  const wrapper=CFG.ens.nameWrappers.find(w=>w.toLowerCase()===registryOwner);
  if(!wrapper) return {fullName,node,registryOwner,effectiveOwner:registryOwner,wrapped:false,expiry:null};
  const getDataSelector=window.GoalOSCrypto.selector('getData(uint256)');
  let result;
  try { result=await ethCall(wrapper,getDataSelector+node.slice(2)); }
  catch { result=await ethCall(wrapper,'0x6352211e'+node.slice(2)); }
  const clean=String(result).replace(/^0x/,'');
  const effectiveOwner=decodeAddressWord(clean);
  const expiry=clean.length>=192 ? Number(decodeUintWord(clean,2)) : null;
  return {fullName,node,registryOwner,effectiveOwner,wrapped:true,expiry};
}
async function signReceipt(route,account,details){
  const block=await rpc('eth_blockNumber');
  const issuedAt=nowISO();
  const expiresAt=new Date(Date.now()+CFG.access.sessionMinutes*60000).toISOString();
  const origin=location.origin==='null'?'local-file':location.origin;
  const receipt={schema:'GoalOS.AccessReceipt.v1',appId:CFG.appId,version:CFG.version,route,wallet:account,chainId:CFG.ethereumChainId,origin,blockNumber:Number(BigInt(block)),issuedAt,expiresAt,nonce:randomHex(),authorityCreated:'NONE',details};
  const message=[
    'GoalOS UVSI3 Access Receipt',
    `Application: ${CFG.appId}`,
    `Origin: ${origin}`,
    `Wallet: ${String(account).toLowerCase()}`,
    `Route: ${route}`,
    `Issued: ${issuedAt}`,
    `Expires: ${expiresAt}`,
    `Nonce: ${receipt.nonce}`,
    'Authority created: NONE',
    'No token approval, transfer, payment, staking, locking or custody is requested.'
  ].join('\n');
  const signature=await rpc('personal_sign',[hexUtf8(message),account]);
  receipt.message=message; receipt.signature=signature;
  return receipt;
}
function persistAccess(receipt){ sessionStorage.setItem(ACCESS_KEY,JSON.stringify(receipt)); platform.accessLog.unshift({route:receipt.route,wallet:receipt.wallet,issuedAt:receipt.issuedAt,expiresAt:receipt.expiresAt}); platform.accessLog=platform.accessLog.slice(0,50); localStorage.setItem(PLATFORM_KEY,JSON.stringify(platform)); }
function getReceipt(){ return safeJson(sessionStorage.getItem(ACCESS_KEY)||'null'); }
function accessValidByTime(receipt){ return receipt && new Date(receipt.expiresAt).getTime()>Date.now() && receipt.version===CFG.version && receipt.origin===(location.origin==='null'?'local-file':location.origin); }
function unlockApp(receipt){
  document.body.classList.remove('access-locked'); $('#accessGate').classList.add('hidden'); $('#applicationShell').setAttribute('aria-hidden','false'); $('#applicationShell').inert=false;
  const chip=$('#accessChip'); chip.textContent=receipt.route==='local_demo'?'Local QA demo':`${tr('accessGranted')} · ${shortAddress(receipt.wallet)}`; chip.className='status-chip pass';
  if(CORE()?.meta?.currentSection==='command') window.GoalOSDemo?.navTo?.('command');
  renderPlatform();
}
function lockApp(reason='Access requires fresh verification.'){
  sessionStorage.removeItem(ACCESS_KEY); document.body.classList.add('access-locked'); $('#accessGate').classList.remove('hidden'); $('#applicationShell').setAttribute('aria-hidden','true'); $('#applicationShell').inert=true;
  const chip=$('#accessChip'); if(chip){chip.textContent=tr('locked');chip.className='status-chip fail';}
  updateAccessStatus(reason,'fail');
}
async function verifyTokenRoute(){
  setAccessBusy(true); updateAccessStatus('Connecting wallet and reading the current direct AGIALPHA balance…','busy');
  try{
    const account=await connectWallet(); const balance=await tokenBalance(account); const minimum=requiredTokenRaw();
    if(balance<minimum) throw new Error(`This wallet holds ${fmt(Number(balance/10n**18n))} AGIALPHA. At least 1,000,000 direct AGIALPHA is required.`);
    const receipt=await signReceipt('AGIALPHA_DIRECT_BALANCE',account,{contract:CFG.token.contract,balanceRaw:balance.toString(),minimumRaw:minimum.toString(),direct:true});
    persistAccess(receipt); updateAccessStatus('Balance verified. Sign-in receipt created with no spending permission.','pass'); unlockApp(receipt);
  }catch(error){updateAccessStatus(error.message||String(error),'fail')}finally{setAccessBusy(false)}
}
async function verifyClubRoute(){
  setAccessBusy(true); updateAccessStatus('Connecting wallet and verifying current direct AGI Club ownership…','busy');
  try{
    const label=validateClubLabel($('#clubLabel').value); const account=await connectWallet(); const result=await currentClubOwner(label);
    if(!result.effectiveOwner || result.effectiveOwner!==account) throw new Error(`${result.fullName} is not currently held directly by the connected wallet.`);
    if(result.expiry && result.expiry*1000<=Date.now()) throw new Error(`${result.fullName} is wrapped but its recorded expiry is no longer current.`);
    const receipt=await signReceipt('AGI_CLUB_DIRECT_OWNER',account,{name:result.fullName,node:result.node,wrapped:result.wrapped,registryOwner:result.registryOwner,effectiveOwner:result.effectiveOwner,expiry:result.expiry,direct:true});
    persistAccess(receipt); updateAccessStatus('Direct ownership verified. Domain-bound sign-in receipt created.','pass'); unlockApp(receipt);
  }catch(error){updateAccessStatus(error.message||String(error),'fail')}finally{setAccessBusy(false)}
}
async function revalidateAccess(receipt=getReceipt()){
  if(!accessValidByTime(receipt)){lockApp('The access session expired. Verify eligibility again.');return false;}
  if(receipt.route==='local_demo'){unlockApp(receipt);return true;}
  try{
    const accounts=await rpc('eth_accounts'); const account=normalizeAddress(accounts?.[0]); const chain=await rpc('eth_chainId');
    if(account!==receipt.wallet || chain!==CFG.ethereumChainId) throw new Error('Wallet or network changed.');
    if(receipt.route==='AGIALPHA_DIRECT_BALANCE' && await tokenBalance(account)<requiredTokenRaw()) throw new Error('The direct AGIALPHA balance no longer meets the access threshold.');
    if(receipt.route==='AGI_CLUB_DIRECT_OWNER'){
      const result=await currentClubOwner(receipt.details.name.split('.')[0]);
      if(result.effectiveOwner!==account) throw new Error('Current direct AGI Club ownership is no longer verified.');
      if(result.expiry && result.expiry*1000<=Date.now()) throw new Error('The wrapped AGI Club name is expired.');
    }
    unlockApp(receipt); return true;
  } catch(error){ lockApp(`${error.message||error} Access relocked.`); return false; }
}

function coreMetrics(){
  const s=CORE(); if(!s)return {};
  const champion=s.formation.results?.find(r=>r.candidateId===s.formation.championId);
  const proof=s.proof.result;
  const half=platform.navigator.capabilityHalfLifeDays; const lag=platform.navigator.adaptationLagDays;
  return {
    ser:half?lag/half:0,
    candidates:s.candidates?.length||0,
    jobsDone:s.jobs?.filter(j=>j.status==='completed').length||0,
    champion:champion?.candidateName||'Not selected',
    utility:champion?.metrics?.utility||0,
    proofStatus:proof?.status||'Not run',
    proofPass:Boolean(proof?.pass),
    lcb:proof?.paired?.lcb||0,
    generation:s.recursive?.generation||0,
    authority:s.authority?.level||'A0',
    chronicle:s.chronicle?.length||0
  };
}
function pageHead(kicker,title,description,actions=''){return `<div class="page-head"><div><div class="page-kicker">${esc(kicker)}</div><h1>${esc(title)}</h1><p>${esc(description)}</p></div><div class="page-actions">${actions}</div></div>`;}
function renderCommand(){
  const s=CORE(),m=coreMetrics(); if(!s)return;
  $('#section-command').innerHTML=`
    <div class="hero command-hero"><div class="eyebrow">GoalOS v8 · Sovereign Specialist ASI Training Toolkit Ω</div><h1>Constitute the superior successor—under proof.</h1><p>Navigator Ω detects the succession event. SEIZE underwrites the trajectory. AGI Jobs manufacture the complete candidate. The Sovereign Mission Gym trains and tests. Fresh proof may establish Specialist ASI. Accountable admission alone creates Successor Ω.</p><div class="hero-doctrine"><strong>Beta is rented. The Spread is underwritten. Alpha is manufactured. Specialist ASI is proven. Authority never exceeds fresh proof.</strong></div><div class="hero-actions"><button class="button" data-platform-nav="navigator">Detect an event</button><button class="button secondary" data-platform-nav="mission">Constitute a mission</button><button class="button secondary" data-platform-action="guided-cycle">Run the guided synthetic cycle</button><button class="button navy" data-platform-nav="ai">Open AI Studio</button></div><div class="hero-badges"><span>Strict dual access</span><span>Offline-first PWA</span><span>Secure optional AI</span><span>Fresh-proof firewall</span><span>Mission Pack export</span></div></div>
    <div class="stage-ribbon" style="grid-template-columns:repeat(6,1fr)"><div class="stage"><div class="num">01</div><b>Successor Manifold</b><small>Reachable complete architectures</small></div><div class="stage"><div class="num">02</div><b>Mission Advantage Gradient</b><small>Direction of proof-adjusted superiority</small></div><div class="stage"><div class="num">03</div><b>SEIZE Underwriting</b><small>Proof capital and reversible path</small></div><div class="stage"><div class="num">04</div><b>Bounded AGI Jobs</b><small>Work-and-proof contracts</small></div><div class="stage ${m.proofPass?'complete':''}"><div class="num">05</div><b>Specialist ASI</b><small>${m.proofPass?'Synthetic gate passed':'Must be earned'}</small></div><div class="stage ${s.authority.admission?.status==='admitted'?'complete':''}"><div class="num">06</div><b>Successor Ω</b><small>Accountable bounded admission</small></div></div>
    <div class="metrics-grid"><div class="metric-card"><strong>${fmt(m.ser)}</strong><small>Singularity Exposure Ratio</small><div class="trend">${m.ser<1?'Adaptation ahead of half-life':'Adaptation lag requires attention'}</div></div><div class="metric-card"><strong>${m.candidates}</strong><small>Complete candidates</small></div><div class="metric-card"><strong>${m.jobsDone}/21</strong><small>AGI Jobs completed</small></div><div class="metric-card"><strong>G${m.generation}</strong><small>Foundry generation</small></div><div class="metric-card"><strong>${esc(m.authority)}</strong><small>Current authority posture</small></div></div>
    <div class="grid-3"><div class="card"><h2>Current mission</h2><p><strong>${esc(s.mission.name)}</strong></p><p>${esc(s.mission.objective)}</p><button class="button small secondary" data-platform-nav="mission">Review constitution</button></div><div class="card"><h2>Proof state</h2><p><strong>${esc(m.proofStatus)}</strong></p><p>Matched LCB: ${fmt(m.lcb)} · Champion: ${esc(m.champion)}</p><button class="button small secondary" data-platform-nav="proof">Inspect fresh proof</button></div><div class="card"><h2>Institutional memory</h2><p><strong>${m.chronicle} Chronicle records</strong></p><p>Only evidenced, versioned, bounded and challengeable capability should shape later missions.</p><button class="button small secondary" data-platform-nav="chronicle">Open Chronicle</button></div></div>`;
}
function renderNavigator(){
  const n=platform.navigator,ser=n.capabilityHalfLifeDays?n.adaptationLagDays/n.capabilityHalfLifeDays:0;
  $('#section-navigator').innerHTML=`${pageHead('01 · Navigator Ω','Frontier Radar and succession-event detection','Track capability, cost, provider, law, security, infrastructure and market changes that can impair the incumbent.',`<button class="button" data-platform-action="add-event">Record event</button>`)}
  <div class="grid-3"><div class="card highlight"><h2>Capability half-life</h2><label>Expected days before a material assumption changes<input id="capabilityHalfLife" type="number" min="1" value="${n.capabilityHalfLifeDays}"></label></div><div class="card"><h2>Adaptation lag</h2><label>Days to detect, constitute, prove and admit<input id="adaptationLag" type="number" min="1" value="${n.adaptationLagDays}"></label></div><div class="card ${ser<1?'success':'danger'}"><h2>Singularity Exposure Ratio</h2><div class="big-number">${fmt(ser)}</div><p>${ser<1?'The institution is positioned to adapt before material assumptions decay.':'The institution risks operating from stale assumptions.'}</p><button class="button small" data-platform-action="save-radar">Save radar</button></div></div>
  <div class="card" style="margin-top:14px"><h2>Live succession events</h2><div class="event-list">${n.events.map(e=>`<article class="event-card"><div><span class="pill">${esc(e.type)}</span><h3>${esc(e.title)}</h3><p>${esc(e.evidence)}</p></div><div class="event-score"><strong>${e.impact}</strong><small>Impact</small><strong>${e.urgency}</strong><small>Urgency</small></div><div class="button-row"><button class="button small secondary" data-platform-action="underwrite-event" data-event-id="${e.id}">Underwrite</button><button class="button small danger" data-platform-action="remove-event" data-event-id="${e.id}">Remove</button></div></article>`).join('')}</div></div>`;
}
function renderBook(){
  const s=CORE(); if(!s)return; const positions=['incumbent','immediate','challenger','reserve','hedge','retirement'];
  $('#section-book').innerHTML=`${pageHead('02 · Successor Book','Maintain a live portfolio of institutional architectures','The book preserves immediate, challenger, reserve, hedge and retirement options instead of collapsing uncertainty into one recommendation.',`<button class="button" data-platform-action="export-book">Export book</button>`)}<div class="book-grid">${positions.map(position=>{const rows=(s.successorBook||[]).filter(x=>x.position===position);return `<section class="book-column"><h2>${position[0].toUpperCase()+position.slice(1)}</h2><p>${{incumbent:'Current accepted architecture.',immediate:'Fund decisive proof now.',challenger:'Maintain a bounded evidence programme.',reserve:'Preserve option at low carrying cost.',hedge:'Maintain fallback readiness.',retirement:'Contain, migrate, revoke or exit.'}[position]}</p>${rows.length?rows.map(row=>{const c=s.candidates.find(x=>x.id===row.candidateId);return `<article class="book-item"><b>${esc(c?.name||row.candidateId)}</b><small>${esc(row.thesis||'')}</small><select data-platform-book="${esc(row.candidateId)}"><option value="${position}">${position}</option>${positions.filter(p=>p!==position).map(p=>`<option value="${p}">${p}</option>`).join('')}</select></article>`}).join(''):'<div class="book-empty">No position</div>'}</section>`}).join('')}</div><div class="callout dark"><strong>Portfolio law:</strong> Proof capital is the option premium. Full formation and authority are committed only when decisive evidence justifies exercise.</div>`;
}
function proofDimension(name,value,pass){return `<div class="gate ${pass?'pass':'fail'}"><div class="gate-icon">${pass?'✓':'!'}</div><div><b>${esc(name)}</b><small>${fmt(value)} · ${pass?'gate passed':'not yet established'}</small></div></div>`;}
function renderSpecialist(){
  const s=CORE(); if(!s)return; const p=s.proof.result; const candidate=s.candidates.find(c=>c.id===s.proof.frozenCandidateId); const pass=Boolean(p?.pass);
  const vals={Capability:p?.candidate?.metrics?.quality||0,Economics:p?.candidate?.metrics?.value||0,Reliability:p?.candidate?.metrics?.reliability||0,Sovereignty:p?.candidate?.metrics?.sovereignty||0,Governance:p?.candidate?.metrics?.governance||0,Transfer:p?.candidate?.metrics?.transfer||0};
  $('#section-specialist').innerHTML=`${pageHead('11 · Specialist ASI','An earned, mission-bounded comparative proof state','Training creates a candidate. One exact frozen release earns Specialist ASI only by defeating the incumbent and strongest credible alternative on fresh protected work.')}
  <div class="grid-2"><div class="card navy"><div class="proof-seal ${pass?'':'fail'}"><strong>${pass?'ASI':'CANDIDATE'}</strong><small>${pass?'Synthetic fresh-proof gate passed':'No qualifying fresh-proof verdict'}</small></div><h2>${esc(candidate?.name||'No frozen challenger')}</h2><p>${pass?'This local deterministic environment records a simulation pass. It does not establish external real-world Specialist ASI.':'Freeze one exact challenger and run Fresh Proof.'}</p></div><div class="card"><h2>Six superiority dimensions</h2><div class="gates">${Object.entries(vals).map(([k,v])=>proofDimension(k,v,pass&&v>=60)).join('')}</div></div></div>
  <div class="callout ${pass?'green':'red'}"><strong>${pass?'Specialist ASI gate — synthetic pass':'Specialist ASI not earned'}</strong><br>Capability, economics, reliability, sovereignty, governance and transfer must all survive protected comparison with zero hard-gate failure.</div>
  <div class="card"><h2>Separate constitutional transition</h2><div class="flow"><div class="flow-node"><b>Trained candidate</b><small>Formation Gym</small></div><div class="flow-arrow">→</div><div class="flow-node"><b>Frozen challenger</b><small>Exact release</small></div><div class="flow-arrow">→</div><div class="flow-node"><b>Fresh proof</b><small>Independent comparison</small></div><div class="flow-arrow">→</div><div class="flow-node"><b>Specialist ASI</b><small>Mission-dominant proof state</small></div><div class="flow-arrow">→</div><div class="flow-node"><b>Accountable admission</b><small>Human constitutional decision</small></div><div class="flow-arrow">→</div><div class="flow-node"><b>Successor Ω</b><small>Scoped authority</small></div></div></div>`;
}
function aiSchema(){return {"type":"object","additionalProperties":false,"required":["executive_summary","mission_constitution","successor_manifold","mission_advantage_gradient","seize_underwriting","agi_jobs","fresh_proof_plan","authority_recommendation","board_brief","claim_boundary"],"properties":{"executive_summary":{"type":"string"},"mission_constitution":{"type":"object","additionalProperties":false,"required":["mission","objective","beneficiary","incumbent","scope","critical_failures","constraints","authority_ceiling"],"properties":{"mission":{"type":"string"},"objective":{"type":"string"},"beneficiary":{"type":"string"},"incumbent":{"type":"string"},"scope":{"type":"string"},"critical_failures":{"type":"array","items":{"type":"string"}},"constraints":{"type":"array","items":{"type":"string"}},"authority_ceiling":{"type":"string"}}},"successor_manifold":{"type":"array","items":{"type":"object","additionalProperties":false,"required":["name","architecture","thesis","anti_recommendation","position"],"properties":{"name":{"type":"string"},"architecture":{"type":"string"},"thesis":{"type":"string"},"anti_recommendation":{"type":"string"},"position":{"type":"string"}}}},"mission_advantage_gradient":{"type":"object","additionalProperties":false,"required":["direction","next_experiment","proof_adjusted_value_hypothesis"],"properties":{"direction":{"type":"string"},"next_experiment":{"type":"string"},"proof_adjusted_value_hypothesis":{"type":"string"}}},"seize_underwriting":{"type":"object","additionalProperties":false,"required":["decision","proof_capital","next_decisive_evidence","stop_conditions"],"properties":{"decision":{"type":"string"},"proof_capital":{"type":"number"},"next_decisive_evidence":{"type":"string"},"stop_conditions":{"type":"array","items":{"type":"string"}}}},"agi_jobs":{"type":"array","items":{"type":"object","additionalProperties":false,"required":["id","name","objective","owner","evidence_required","verifier","prohibited_actions"],"properties":{"id":{"type":"string"},"name":{"type":"string"},"objective":{"type":"string"},"owner":{"type":"string"},"evidence_required":{"type":"string"},"verifier":{"type":"string"},"prohibited_actions":{"type":"array","items":{"type":"string"}}}}},"fresh_proof_plan":{"type":"object","additionalProperties":false,"required":["challenger","reference_set","cases","metrics","hard_gates","claim_boundary"],"properties":{"challenger":{"type":"string"},"reference_set":{"type":"array","items":{"type":"string"}},"cases":{"type":"string"},"metrics":{"type":"array","items":{"type":"string"}},"hard_gates":{"type":"array","items":{"type":"string"}},"claim_boundary":{"type":"string"}}},"authority_recommendation":{"type":"object","additionalProperties":false,"required":["level","rationale","permitted","prohibited","expiry_trigger"],"properties":{"level":{"type":"string"},"rationale":{"type":"string"},"permitted":{"type":"array","items":{"type":"string"}},"prohibited":{"type":"array","items":{"type":"string"}},"expiry_trigger":{"type":"string"}}},"board_brief":{"type":"object","additionalProperties":false,"required":["decision","rationale","next_action","risks","questions"],"properties":{"decision":{"type":"string"},"rationale":{"type":"string"},"next_action":{"type":"string"},"risks":{"type":"array","items":{"type":"string"}},"questions":{"type":"array","items":{"type":"string"}}}},"claim_boundary":{"type":"string"}}};}
function compactCore(){const s=CORE(); return {meta:s.meta,mission:s.mission,gym:{version:s.gym.version,spec:s.gym.spec,distribution:s.gym.distribution},candidates:s.candidates.map(c=>({id:c.id,name:c.name,kind:c.kind,notes:c.notes,params:c.params,burden:c.burden})),successorBook:s.successorBook,seize:s.seize,jobs:s.jobs,formation:{championId:s.formation.championId,results:s.formation.results},recursive:s.recursive,proof:s.proof,authority:s.authority,chronicle:s.chronicle.slice(0,30),negativeCapability:s.negativeCapability.slice(0,30)};}
function bridgePrompt(action){const evidence=platform.ai.consent?platform.evidence.map(x=>({name:x.name,text:x.text.slice(0,20000)})):[];return `You are the GoalOS UVSI3 institutional intelligence engine. Perform action: ${action}. Preserve the exact distinction between candidacy, fresh comparative proof, Specialist ASI and accountable Mission-Sovereign admission. Never claim real-world proof from simulation. Authority never exceeds fresh evidence.\n\nPROJECT:\n${JSON.stringify(compactCore(),null,2)}\n\nEVIDENCE:\n${JSON.stringify(evidence,null,2)}\n\nReturn one JSON object matching this schema:\n${JSON.stringify(aiSchema(),null,2)}`;}
function offlineAI(action){
  const s=CORE(), mission=s.mission;
  const alternatives=s.candidates.map((c,i)=>({name:c.name,architecture:`${c.kind} complete successor architecture`,thesis:c.notes,position:i===0?'incumbent':c.kind==='sovereign'?'immediate':'challenger',anti_recommendation:`Do not promote ${c.name} until it passes matched fresh work and complete-cost gates.`}));
  return {
    mode:'offline_constitution',action,generatedAt:nowISO(),
    executive_summary:`${mission.name} should proceed through a bounded proof programme, not direct deployment. Freeze the mission, compare complete architectures and purchase the smallest decisive fresh evidence.`,
    mission_constitution:{mission:mission.name,objective:mission.objective,beneficiary:mission.beneficiary,incumbent:mission.incumbent,scope:mission.scope,critical_failures:mission.criticalFailures,constraints:mission.constraints,authority_ceiling:mission.authorityCeiling},
    successor_manifold:alternatives,
    mission_advantage_gradient:{direction:'Increase verifier independence, evidence coverage, portability and failure-aware abstention before expanding raw capability.',next_experiment:'Matched test of the incumbent, strongest general alternative and sovereign hybrid on fresh representative cases.',proof_adjusted_value_hypothesis:'The largest marginal value is expected from stronger evidence, independent verification and lower lifecycle burden rather than model scale alone.'},
    seize_underwriting:{decision:'PROCEED TO BOUNDED PROOF',proof_capital:Number(mission.proofBudget)||0,next_decisive_evidence:s.seize.nextEvidence,stop_conditions:['Any critical miss','Any unauthorized action','Negative complete mission value','Governance regression']},
    agi_jobs:s.jobs.map(j=>({id:String(j.id),name:j.name,objective:j.output,owner:j.owner||'Accountable principal',evidence_required:j.output,verifier:'Independent verifier',prohibited_actions:['Self-certification','Authority expansion','Undeclared external action']})),
    fresh_proof_plan:{challenger:s.proof.frozenCandidateId||s.selectedCandidateId||'Freeze one exact challenger',reference_set:['incumbent','best credible alternative'],cases:'Protected, representative, unseen mission cases with declared interventions.',metrics:['mission quality','complete economics','reliability','sovereignty','governance','transfer'],hard_gates:['positive matched lower bound','zero critical errors','zero unauthorized actions','evidence coverage','rollback readiness'],claim_boundary:'The teaching Gym does not certify; proof cases and scorer internals remain separated from formation.'},
    authority_recommendation:{level:s.authority.level,rationale:'Capability does not automatically create authority; retain the lowest authority posture consistent with current fresh evidence.',permitted:s.authority.permitted||[],prohibited:s.authority.prohibited||[],expiry_trigger:'Any material release, mission, environment, provider, evidence or Authority Envelope change.'},
    board_brief:{decision:s.seize.decision,rationale:'Proceed only through bounded proof and preserve retain, repair, rent, partner and stop alternatives.',next_action:s.seize.nextEvidence,risks:['Gym-to-reality basis risk','Critical-error tail risk','Provider dependency','Proof Debt'],questions:['What changed?','Which incumbent assumption is impaired?','What evidence would change the decision?','What authority is justified now?']},
    claim_boundary:'This offline analysis is deterministic institutional scaffolding. It is not independent fresh proof, professional advice, realized Mission Alpha or real-world Specialist ASI certification.'
  };
}
async function runAI(action){
  platform.ai.lastAction=action; const resultBox=$('#aiResult'); if(resultBox)resultBox.innerHTML='<div class="ai-thinking"><span></span> Compiling the institutional analysis…</div>';
  try{
    let result;
    if(platform.ai.endpoint){
      const response=await fetch(platform.ai.endpoint.replace(/\/$/,'')+'/api/analyze',{method:'POST',headers:{'Content-Type':'application/json','X-GoalOS-App':'goalos-uvsi3-v8-8'},body:JSON.stringify({action,language:lang(),project:compactCore(),evidence:platform.ai.consent?platform.evidence:[],accessReceipt:getReceipt(),schema:aiSchema()})});
      if(!response.ok) throw new Error((await response.text())||`AI backend returned ${response.status}`);
      const payload=await response.json(); result=payload.result||payload;
      platform.ai.mode='secure';
    } else result=offlineAI(action);
    platform.ai.lastResult=result; platform.ai.history.unshift({action,at:nowISO(),mode:platform.ai.mode,result}); platform.ai.history=platform.ai.history.slice(0,20); savePlatform(); window.GoalOSDemo?.record?.('AI_ANALYSIS_COMPLETED',`AI Studio completed ${action}`,result.executive_summary||'',{mode:platform.ai.mode}); toast('AI Studio analysis completed');
  } catch(error){ if(resultBox)resultBox.innerHTML=`<div class="callout red"><strong>AI run failed</strong><br>${esc(error.message||error)}</div>`; toast('AI run failed'); }
}
function renderAIResult(result){if(!result)return '<div class="callout">No AI analysis has been generated yet. The offline engine works immediately; a secure backend adds model-powered analysis.</div>';return `<div class="ai-result"><div class="callout green"><strong>${esc(result.executive_summary||'Analysis complete')}</strong></div><div class="grid-2"><div class="card slim"><h3>Mission Constitution</h3><pre>${esc(JSON.stringify(result.mission_constitution||{},null,2))}</pre></div><div class="card slim"><h3>Mission Advantage Gradient</h3><pre>${esc(JSON.stringify(result.mission_advantage_gradient||{},null,2))}</pre></div><div class="card slim"><h3>SEIZE Underwriting</h3><pre>${esc(JSON.stringify(result.seize_underwriting||{},null,2))}</pre></div><div class="card slim"><h3>Authority Recommendation</h3><pre>${esc(JSON.stringify(result.authority_recommendation||{},null,2))}</pre></div></div><details><summary>Successor Manifold and AGI Jobs</summary><pre>${esc(JSON.stringify({successor_manifold:result.successor_manifold,agi_jobs:result.agi_jobs},null,2))}</pre></details><details><summary>Fresh proof and board brief</summary><pre>${esc(JSON.stringify({fresh_proof_plan:result.fresh_proof_plan,board_brief:result.board_brief,claim_boundary:result.claim_boundary},null,2))}</pre></details><div class="button-row"><button class="button small" data-platform-action="apply-ai">Apply safe fields to project</button><button class="button small secondary" data-platform-action="export-ai">Export JSON</button></div></div>`;}
function renderAI(){
  $('#section-ai').innerHTML=`${pageHead('15 · AI Studio','AI executes the GoalOS methodology—not merely discusses it','Use the deterministic offline compiler, a secure OpenAI Responses API backend, or the ChatGPT Bridge. No API key is ever stored in the browser.')}
  <div class="grid-3"><div class="card ${platform.ai.endpoint?'success':'warning'}"><h2>AI connection</h2><label>Secure backend URL<input id="aiEndpoint" type="url" placeholder="https://your-worker.workers.dev" value="${esc(platform.ai.endpoint)}"></label><div class="button-row"><button class="button small" data-platform-action="save-ai-endpoint">Save endpoint</button><button class="button small secondary" data-platform-action="test-ai">Test</button></div><p class="fine">Blank endpoint = autonomous offline compiler. Never paste an API key here.</p></div><div class="card"><h2>Evidence consent</h2><label class="check-row"><input id="aiEvidenceConsent" type="checkbox" ${platform.ai.consent?'checked':''}><span>Include the locally selected evidence texts in the next secure AI request.</span></label><p class="fine">Evidence remains local unless this box is selected and a secure backend is used.</p></div><div class="card navy"><h2>Current mode</h2><div class="big-number">${platform.ai.endpoint?'SECURE AI':'OFFLINE'}</div><p>${platform.ai.endpoint?'Server-side model execution with store:false.':'Deterministic institutional compiler; no network request.'}</p></div></div>
  <div class="card" style="margin-top:14px"><h2>Execute the methodology</h2><div class="ai-action-grid">${[['mission','Compile Mission Constitution'],['manifold','Generate Successor Manifold'],['seize','Underwrite SEIZE'],['jobs','Compile AGI Jobs'],['proof','Design Fresh Proof'],['board','Prepare Board Brief'],['full_cycle','Run Complete AI Cycle']].map(([id,label])=>`<button class="ai-action" data-platform-ai="${id}"><b>${esc(label)}</b><small>${id==='full_cycle'?'Mission → Manifold → SEIZE → Jobs → Proof → Authority':'One bounded institutional output'}</small></button>`).join('')}</div><div class="button-row" style="margin-top:12px"><button class="button secondary" data-platform-action="bridge-prompt">Copy ChatGPT Bridge prompt</button></div></div>
  <div id="aiResult" style="margin-top:14px">${renderAIResult(platform.ai.lastResult)}</div>`;
}
function renderBoard(){
  const s=CORE(),m=coreMetrics(); if(!s)return; const proof=s.proof.result;
  const recommendation=proof?.pass ? (s.authority.admission?.status==='admitted'?'Operate within current Authority Envelope':'Eligible for accountable admission review') : 'Continue proof, repair or retain incumbent';
  $('#section-board').innerHTML=`${pageHead('16 · Board and Metrics','Govern the succession portfolio','The board sees events, proof, authority, risk, value and renewal—not just model performance.',`<button class="button" data-platform-action="export-board">Export board brief</button>`)}
  <div class="metrics-grid"><div class="metric-card"><strong>${fmt(m.ser)}</strong><small>Singularity Exposure Ratio</small></div><div class="metric-card"><strong>${fmt(m.utility)}</strong><small>Formation champion utility</small></div><div class="metric-card"><strong>${fmt(m.lcb)}</strong><small>Fresh-proof lower bound</small></div><div class="metric-card"><strong>${m.authority}</strong><small>Authority level</small></div><div class="metric-card"><strong>${m.chronicle}</strong><small>Chronicle records</small></div></div>
  <div class="grid-2"><div class="card navy"><h2>Recommended board posture</h2><p class="board-verdict">${esc(recommendation)}</p><p>Next decisive evidence: ${esc(s.seize.nextEvidence)}</p><p>SEIZE state: ${esc(s.seize.decision)}</p></div><div class="card"><h2>Authority-at-Risk review</h2><p>Current ceiling: <strong>${esc(s.mission.authorityCeiling)}</strong></p><p>Current level: <strong>${esc(s.authority.level)}</strong></p><p>Proof: <strong>${esc(proof?.status||'Not run')}</strong></p><p>Unauthorized actions: <strong>${proof?.unauthorizedActions??'—'}</strong></p></div></div>
  <div class="card" style="margin-top:14px"><h2>Board questions</h2><ol class="board-questions"><li>What changed?</li><li>Which incumbent assumption is impaired?</li><li>Which successor positions are in the book?</li><li>What evidence would change the recommendation?</li><li>What proof capital and authority are currently at risk?</li><li>What should happen in 72 hours, 30 days, 90 days and 365 days?</li></ol></div>`;
}
function renderEvidence(){
  $('#section-evidence').innerHTML=`${pageHead('17 · Evidence Room','Ground claims before authority','Attach rights-cleared text, CSV, JSON or Markdown evidence. Files remain browser-local unless explicitly included in a secure AI request.',`<button class="button" data-platform-action="add-evidence">Add files</button>`)}
  <div class="grid-2"><div class="card"><h2>Evidence ledger</h2>${platform.evidence.length?platform.evidence.map((f,i)=>`<article class="evidence-item"><div><b>${esc(f.name)}</b><small>${f.type||'text'} · ${f.text.length.toLocaleString()} characters · ${new Date(f.addedAt).toLocaleString()}</small></div><button class="button small danger" data-platform-action="remove-evidence" data-index="${i}">Remove</button></article>`).join(''):'<div class="callout">No local evidence files attached.</div>'}</div><div class="card"><h2>Claim boundary</h2><p>Browser-local evidence is not independently verified merely because it is attached. Protected fresh proof requires separated custody, protected cases, exact candidate manifests and accountable review.</p><div class="callout dark"><strong>Access ≠ authority. Attachment ≠ proof. AI analysis ≠ acceptance.</strong></div></div></div>
  <div class="card" style="margin-top:14px"><h2>Add a manual evidence claim</h2><div class="form-grid"><label>Claim<input id="claimText" placeholder="What is asserted?"></label><label>Source<input id="claimSource" placeholder="Source, date or provenance"></label><label>Status<select id="claimStatus"><option>unverified</option><option>supported</option><option>contradicted</option><option>expired</option></select></label></div><button class="button small" data-platform-action="add-claim">Add claim</button>${platform.claims.length?`<div class="claim-list">${platform.claims.map((c,i)=>`<div class="claim-row"><span class="pill">${esc(c.status)}</span><b>${esc(c.claim)}</b><small>${esc(c.source)}</small><button class="link-button" data-platform-action="remove-claim" data-index="${i}">remove</button></div>`).join('')}</div>`:''}</div>`;
}
function appendExportPanel(){const section=$('#section-export'); if(!section||section.querySelector('#platformExport'))return; section.insertAdjacentHTML('beforeend',`<div id="platformExport" class="card" style="margin-top:16px"><h2>Complete platform exports</h2><div class="export-grid"><div class="export-card"><div class="export-icon">Ω</div><h3>UVSI3 platform state</h3><p>Navigator, Successor Book, evidence and AI history.</p><button class="button small" data-platform-action="export-platform">Download JSON</button></div><div class="export-card"><div class="export-icon">▣</div><h3>Implementation paper</h3><p>Unified GoalOS Navigator + SEIZE + Gym + Specialist ASI + Successor Ω paper.</p><a class="button small secondary" href="${CFG.links.paperA4}" target="_blank" rel="noopener">Open A4 PDF</a></div><div class="export-card"><div class="export-icon">✓</div><h3>Access receipt</h3><p>Signed eligibility receipt. It creates no authority.</p><button class="button small secondary" data-platform-action="export-access">Download receipt</button></div></div></div>`);}
function renderPlatform(){
  const chip=$('#aiChip'); if(chip){chip.textContent=platform.ai.endpoint?tr('secureAI'):tr('offline');chip.className=`status-chip ${platform.ai.endpoint?'pass':'neutral'}`;}
  renderCommand();renderNavigator();renderBook();renderSpecialist();renderAI();renderBoard();renderEvidence();appendExportPanel();
}
function navigate(id){ window.GoalOSDemo?.navTo?.(id); renderPlatform(); }
async function guidedCycle(){
  navigate('mission'); toast('Guided cycle started: Mission Constitution');
  await new Promise(r=>setTimeout(r,350)); window.GoalOSDemo?.runFormation?.();
  await new Promise(r=>setTimeout(r,350)); await window.GoalOSDemo?.runRecursive?.(5);
  await new Promise(r=>setTimeout(r,350)); window.GoalOSDemo?.freeze?.();
  await new Promise(r=>setTimeout(r,350)); window.GoalOSDemo?.runFreshProof?.();
  navigate('proof'); toast('Guided synthetic cycle completed through Fresh Proof');
}
function addEventModal(){modal('Record a succession event',`<div class="form-grid"><label>Event type<select id="newEventType"><option>Capability</option><option>Cost</option><option>Market</option><option>Law</option><option>Security</option><option>Infrastructure</option><option>Provider</option><option>Evidence</option></select></label><label>Title<input id="newEventTitle" placeholder="What changed?"></label><label>Impact (0-100)<input id="newEventImpact" type="number" min="0" max="100" value="70"></label><label>Urgency (0-100)<input id="newEventUrgency" type="number" min="0" max="100" value="60"></label><label style="grid-column:1/-1">Evidence<textarea id="newEventEvidence" placeholder="Source, provenance and why the change matters"></textarea></label></div><button class="button" data-platform-action="commit-event">Save event</button>`);}
function underwriteEvent(id){const event=platform.navigator.events.find(e=>e.id===id);if(!event)return;window.GoalOSDemo?.patchState?.(draft=>{draft.seize.decision='Event admitted for bounded underwriting';draft.seize.nextEvidence=`Purchase the smallest decisive evidence for: ${event.title}`;draft.chronicle.unshift({id:`event_${Date.now()}`,type:'SUCCESSION_EVENT_DETECTED',title:event.title,detail:event.evidence,data:event,at:nowISO(),generation:draft.meta.institutionGeneration});return draft});navigate('seize');}
function updateBook(candidateId,position){window.GoalOSDemo?.patchState?.(draft=>{const row=draft.successorBook.find(x=>x.candidateId===candidateId);if(row)row.position=position;return draft});}
function applyAI(){const result=platform.ai.lastResult;if(!result)return;window.GoalOSDemo?.patchState?.(draft=>{const m=result.mission_constitution||{};if(m.objective)draft.mission.objective=m.objective;if(m.beneficiary)draft.mission.beneficiary=m.beneficiary;if(Array.isArray(m.critical_failures))draft.mission.criticalFailures=m.critical_failures;if(Array.isArray(m.constraints))draft.mission.constraints=m.constraints;if(result.seize_underwriting?.decision)draft.seize.decision=result.seize_underwriting.decision;if(result.seize_underwriting?.next_decisive_evidence)draft.seize.nextEvidence=result.seize_underwriting.next_decisive_evidence;draft.chronicle.unshift({id:`ai_${Date.now()}`,type:'AI_ANALYSIS_APPLIED',title:'AI Studio analysis applied',detail:result.executive_summary||'',data:{mode:platform.ai.mode},at:nowISO(),generation:draft.meta.institutionGeneration});return draft});toast('Safe AI fields applied; proof and authority were not changed.');}
function boardMarkdown(){const s=CORE(),m=coreMetrics();return `# GoalOS UVSI3 Board Brief\n\n**Project:** ${s.meta.projectId}\n**Mission:** ${s.mission.name}\n**Generated:** ${nowISO()}\n\n## Decision state\n- Singularity Exposure Ratio: ${m.ser.toFixed(2)}\n- SEIZE decision: ${s.seize.decision}\n- Formation champion: ${m.champion}\n- Fresh proof: ${m.proofStatus}; LCB ${m.lcb.toFixed(2)}\n- Authority: ${s.authority.level}\n- Chronicle records: ${m.chronicle}\n\n## Next decisive evidence\n${s.seize.nextEvidence}\n\n## Claim boundary\nThis board brief is generated from the local GoalOS project. It is not independent fresh proof, professional advice or production authority.\n`;}
function accessDetails(){modal('How access verification works',`<div class="callout dark"><strong>Access = current direct AGI Club owner OR current direct holder of at least 1,000,000 official AGIALPHA on Ethereum Mainnet.</strong></div><h3>AGI Club route</h3><p>The app computes the ENS namehash of the exact single-label name <code>label.club.agi.eth</code>, reads the current owner from the ENS Registry, and—when wrapped—reads the effective owner and expiry from the NameWrapper. The connected wallet must be the current direct owner.</p><h3>AGIALPHA route</h3><p>The app calls <code>balanceOf(connectedWallet)</code> on <code>${CFG.token.contract}</code>. Only the connected wallet’s current direct Mainnet balance counts.</p><h3>Session</h3><p>After verification, the wallet signs a domain-bound receipt with an expiry and the explicit statement <code>authorityCreated = NONE</code>. Eligibility is rechecked on focus, account changes, network changes and periodically. Any uncertainty relocks the app.</p><h3>What is never requested</h3><p>No token approval, transfer, payment, staking, locking, burning, deposit, custody or transaction authority.</p>`);}
function localDemo(){const receipt={schema:'GoalOS.AccessReceipt.v1',appId:CFG.appId,version:CFG.version,route:'local_demo',wallet:'0x0000000000000000000000000000000000000000',origin:location.origin==='null'?'local-file':location.origin,issuedAt:nowISO(),expiresAt:new Date(Date.now()+60*60000).toISOString(),authorityCreated:'NONE',details:{localOnly:true}};persistAccess(receipt);unlockApp(receipt);}
async function testAIEndpoint(){if(!platform.ai.endpoint){toast('Enter a secure backend URL first');return;}try{const r=await fetch(platform.ai.endpoint.replace(/\/$/,'')+'/api/health');if(!r.ok)throw new Error(await r.text());const data=await r.json();toast(`Secure AI backend ready · ${data.model||'model configured'}`)}catch(e){toast(`Backend test failed: ${e.message}`)}}
function handlePlatformClick(event){
  const nav=event.target.closest('[data-platform-nav]'); if(nav){navigate(nav.dataset.platformNav);return;}
  const ai=event.target.closest('[data-platform-ai]'); if(ai){runAI(ai.dataset.platformAi);return;}
  const action=event.target.closest('[data-platform-action]'); if(!action)return;
  const id=action.dataset.platformAction;
  if(id==='guided-cycle')guidedCycle();
  if(id==='add-event')addEventModal();
  if(id==='commit-event'){const title=$('#newEventTitle')?.value.trim();if(!title){toast('Event title is required');return;}platform.navigator.events.unshift({id:`evt_${Date.now()}`,type:$('#newEventType').value,title,impact:clamp($('#newEventImpact').value),urgency:clamp($('#newEventUrgency').value),evidence:$('#newEventEvidence').value.trim(),status:'watch',at:nowISO()});savePlatform();$('#modal').classList.add('hidden');}
  if(id==='save-radar'){platform.navigator.capabilityHalfLifeDays=Math.max(1,Number($('#capabilityHalfLife').value)||1);platform.navigator.adaptationLagDays=Math.max(1,Number($('#adaptationLag').value)||1);savePlatform();toast('Navigator metrics saved');}
  if(id==='underwrite-event')underwriteEvent(action.dataset.eventId);
  if(id==='remove-event'){platform.navigator.events=platform.navigator.events.filter(e=>e.id!==action.dataset.eventId);savePlatform();}
  if(id==='export-book')downloadText('GoalOS_Successor_Book.json',JSON.stringify(CORE()?.successorBook||[],null,2),'application/json');
  if(id==='save-ai-endpoint'){platform.ai.endpoint=$('#aiEndpoint').value.trim().replace(/\/$/,'');savePlatform();toast(platform.ai.endpoint?'Secure backend URL saved':'Offline mode enabled');}
  if(id==='test-ai')testAIEndpoint();
  if(id==='bridge-prompt'){navigator.clipboard.writeText(bridgePrompt(platform.ai.lastAction||CFG.ai.defaultAction)).then(()=>toast('ChatGPT Bridge prompt copied')).catch(()=>modal('ChatGPT Bridge prompt',`<textarea style="min-height:420px">${esc(bridgePrompt(platform.ai.lastAction||CFG.ai.defaultAction))}</textarea>`));}
  if(id==='apply-ai')applyAI();
  if(id==='export-ai')downloadText('GoalOS_AI_Studio_Result.json',JSON.stringify(platform.ai.lastResult,null,2),'application/json');
  if(id==='export-board')downloadText('GoalOS_UVSI3_Board_Brief.md',boardMarkdown(),'text/markdown');
  if(id==='add-evidence')$('#evidenceFileInput').click();
  if(id==='remove-evidence'){platform.evidence.splice(Number(action.dataset.index),1);savePlatform();}
  if(id==='add-claim'){const claim=$('#claimText').value.trim();if(!claim)return toast('Claim text is required');platform.claims.unshift({claim,source:$('#claimSource').value.trim(),status:$('#claimStatus').value,at:nowISO()});savePlatform();}
  if(id==='remove-claim'){platform.claims.splice(Number(action.dataset.index),1);savePlatform();}
  if(id==='export-platform')downloadText('GoalOS_UVSI3_Platform_State.json',JSON.stringify(platform,null,2),'application/json');
  if(id==='export-access'){const r=getReceipt();if(!r)return toast('No current access receipt');downloadText('GoalOS_Access_Receipt.json',JSON.stringify(r,null,2),'application/json');}
}
async function readEvidenceFiles(files){for(const file of files){if(file.size>1024*1024){toast(`${file.name} exceeds the 1 MB local evidence limit`);continue;}const text=await file.text();platform.evidence.push({name:file.name,type:file.type||'text/plain',size:file.size,text:text.slice(0,CFG.ai.maxEvidenceChars),addedAt:nowISO()});}savePlatform();}
function init(){
  $('#verifyTokenButton').addEventListener('click',verifyTokenRoute); $('#verifyClubButton').addEventListener('click',verifyClubRoute); $('#accessDetailsButton').addEventListener('click',accessDetails); $('#localDemoButton').addEventListener('click',localDemo);
  document.addEventListener('click',handlePlatformClick); document.addEventListener('goalos:state',()=>setTimeout(renderPlatform,0));
  document.addEventListener('change',event=>{if(event.target.matches('[data-platform-book]'))updateBook(event.target.dataset.platformBook,event.target.value);if(event.target.id==='aiEvidenceConsent'){platform.ai.consent=event.target.checked;savePlatform();}});
  $('#evidenceFileInput').addEventListener('change',event=>{readEvidenceFiles([...event.target.files]);event.target.value='';});
  let lastActivity=Date.now(); ['click','keydown','touchstart','pointerdown'].forEach(name=>document.addEventListener(name,()=>lastActivity=Date.now(),{passive:true}));
  setInterval(()=>{const receipt=getReceipt();if(receipt&&Date.now()-lastActivity>CFG.access.inactivityMinutes*60000)lockApp('The session was locked after inactivity.');},60000);
  setInterval(()=>{if(getReceipt())revalidateAccess();},CFG.access.revalidateMinutes*60000);
  window.addEventListener('focus',()=>{if(getReceipt())revalidateAccess();});
  if(window.ethereum?.on){window.ethereum.on('accountsChanged',()=>{if(getReceipt())revalidateAccess()});window.ethereum.on('chainChanged',()=>{if(getReceipt())revalidateAccess()});}
  const localEligible=CFG.access.allowLocalDemo && ['localhost','127.0.0.1',''].includes(location.hostname); if(localEligible)$('#localDemoButton').classList.remove('hidden');
  const receipt=getReceipt(); if(accessValidByTime(receipt))revalidateAccess(receipt); else lockApp('Connect an Ethereum wallet to begin.');
  renderPlatform();
  if('serviceWorker' in navigator && location.protocol.startsWith('http')) navigator.serviceWorker.register('sw.js').catch(console.warn);
}
document.addEventListener('DOMContentLoaded',()=>setTimeout(init,0));
})();
