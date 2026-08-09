(() => {
  'use strict';
  if (window.__GOALOS_DUAL_ACCESS_BI2__) return;
  window.__GOALOS_DUAL_ACCESS_BI2__ = true;
  const C = Object.freeze({
    release: 'v3.3.0-SN5-BI2',
    contract: '0xA61a3B3a130a9c20768EEBF97E21515A6046a1fA',
    contractLower: '0xa61a3b3a130a9c20768eebf97e21515a6046a1fa',
    threshold: 1000000000000000000000000n,
    thresholdText: '1,000,000',
    chainId: '0x1',
    chainIdNumber: 1,
    canonical: 'https://montrealai.github.io/goalos-singularity-navigator-omega-agi-club-owner-access/',
    legalDigest: '0x286db7afa2ad5e9042ef31aa6331a301dd89407832d56cced000c7c3a06dbf3a',
    sessionKey: 'goalos.sn5.bi2.agialpha.access.receipt',
    lockReasonKey: 'goalos.sn5.bi2.agialpha.lock.reason',
    absoluteMs: 30 * 60 * 1000,
    idleMs: 10 * 60 * 1000,
    revalidateMs: 5 * 60 * 1000
  });
  const T = {
    en: {
      kicker:'Second qualifying route · Read-only', title:'AGIALPHA balance-qualified access',
      rule:'Access = Current AGI Club direct owner OR Current direct holder of at least 1,000,000 official AGIALPHA on Ethereum Mainnet',
      copy:'This route unlocks the same browser-local GoalOS operating surfaces. It is a separate conditional-access class and does not confer AGI Club membership, identity, governance, professional authority, commercial rights, financial rights, or permanent access.',
      safety:'Read-only verification. No token approval, transfer, deposit, burn, stake, lock, custody, private key, seed phrase, or blockchain transaction is requested.',
      accept:'I accept the Dual-Access Constitution, Terms, Privacy Notice, Security Notice, Rights & Reuse terms, and No-Offer Notice.',
      boundary:'I understand that balance eligibility creates no AGI Club status, institutional authority, financial entitlement, redemption value, or representation about AGIALPHA’s market value.',
      button:'Connect, sign and verify balance', working:'Verifying Ethereum Mainnet, contract and balance…', signing:'Balance qualifies. Review and sign the temporary access receipt…', granted:'Qualified. Opening the institution…',
      noWallet:'No compatible Ethereum wallet was detected. Install or open a wallet that injects an EIP-1193 provider, then retry.', wrongNetwork:'Please switch the wallet to Ethereum Mainnet.', denied:'The connected wallet does not currently hold the required 1,000,000 official AGIALPHA.', rejected:'The signature request was cancelled. No access was granted.', failed:'Verification could not be completed. No access was granted.',
      currentBalance:'Current direct balance', contract:'Qualifying contract', legal:'Read the access constitution', launcher:'1M AGIALPHA access', close:'Close',
      status:'AGIALPHA-qualified access', expires:'session expires', lock:'Lock', receipt:'Receipt', locked:'The prior AGIALPHA session was closed because eligibility or session conditions changed.', exportNote:'Local receipt only; it creates no external authority.'
    },
    fr: {
      kicker:'Deuxième voie admissible · Lecture seule', title:'Accès conditionnel selon le solde AGIALPHA',
      rule:'Accès = Propriétaire direct actuel de l’AGI Club OU Détenteur direct actuel d’au moins 1 000 000 AGIALPHA officiels sur le réseau principal Ethereum',
      copy:'Cette voie déverrouille les mêmes surfaces opérationnelles locales de GoalOS. Il s’agit d’une catégorie distincte d’accès conditionnel qui ne confère aucune adhésion, identité ou gouvernance de l’AGI Club, aucune autorité professionnelle, aucun droit commercial ou financier et aucun accès permanent.',
      safety:'Vérification en lecture seule. Aucune approbation, transfert, dépôt, destruction, mise en jeu, verrouillage ou garde des jetons, aucune clé privée, phrase de récupération ou transaction blockchain ne sont demandés.',
      accept:'J’accepte la Constitution d’accès double, les Conditions, l’Avis de confidentialité, l’Avis de sécurité, les conditions relatives aux droits et à la réutilisation ainsi que l’Avis d’absence d’offre.',
      boundary:'Je comprends que l’admissibilité selon le solde ne crée aucun statut de l’AGI Club, aucune autorité institutionnelle, aucun droit financier, aucune valeur de rachat et aucune déclaration concernant la valeur marchande d’AGIALPHA.',
      button:'Connecter, signer et vérifier le solde', working:'Vérification du réseau principal Ethereum, du contrat et du solde…', signing:'Le solde est admissible. Vérifiez et signez le reçu d’accès temporaire…', granted:'Admissibilité confirmée. Ouverture de l’institution…',
      noWallet:'Aucun portefeuille Ethereum compatible n’a été détecté. Installez ou ouvrez un portefeuille fournissant un service EIP-1193, puis réessayez.', wrongNetwork:'Veuillez faire passer le portefeuille au réseau principal Ethereum.', denied:'Le portefeuille connecté ne détient pas actuellement les 1 000 000 AGIALPHA officiels requis.', rejected:'La demande de signature a été annulée. Aucun accès n’a été accordé.', failed:'La vérification n’a pas pu être complétée. Aucun accès n’a été accordé.',
      currentBalance:'Solde direct actuel', contract:'Contrat admissible', legal:'Lire la constitution d’accès', launcher:'Accès 1 M AGIALPHA', close:'Fermer',
      status:'Accès admissible AGIALPHA', expires:'fin de session', lock:'Verrouiller', receipt:'Reçu', locked:'La session AGIALPHA précédente a été fermée parce que les conditions d’admissibilité ou de session ont changé.', exportNote:'Reçu local uniquement; il ne crée aucune autorité externe.'
    }
  };
  const S = { receipt:null, gate:null, gateRestore:null, timer:null, idleTimer:null, lastActivity:Date.now(), modal:null, embedded:false, busy:false, providerHandlers:false, verificationSerial:0 };
  const q = (s,r=document) => r.querySelector(s), qa=(s,r=document)=>[...r.querySelectorAll(s)];
  const safeText = el => (el && (el.innerText || el.textContent) || '').replace(/\s+/g,' ').trim();
  function lang(){
    const l=(document.documentElement.lang||'').toLowerCase(); if(l.startsWith('fr')) return 'fr';
    const body=safeText(document.body).slice(0,1200).toLowerCase(); return /français|accès|propriétaire/.test(body) && !/english/.test(body.slice(0,300)) ? 'fr':'en';
  }
  function tx(){ return T[lang()]; }
  function escapeHtml(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
  function isVisible(el){ if(!el||!el.isConnected) return false; const s=getComputedStyle(el),r=el.getBoundingClientRect(); return s.display!=='none'&&s.visibility!=='hidden'&&s.opacity!=='0'&&r.width>2&&r.height>2; }
  function coverage(el){const r=el.getBoundingClientRect();return Math.max(0,Math.min(innerWidth,r.right)-Math.max(0,r.left))*Math.max(0,Math.min(innerHeight,r.bottom)-Math.max(0,r.top))/(Math.max(1,innerWidth*innerHeight));}
  function findGate(){
    if(S.gate && S.gate.isConnected) return S.gate;
    const selectors=['#accessGate','#access-gate','#access-gateway','#gate','#lockScreen','#lock-screen','#accessOverlay','#access-overlay','.access-gate','.access-gateway','.lock-screen','.access-overlay','[data-access-gate]','[data-goalos-gate]','dialog[open]'];
    for(const sel of selectors){for(const el of qa(sel)){if(isVisible(el)&&/AGI Club|club\.agi\.eth|owner access|accès/i.test(safeText(el))){S.gate=el;return el;}}}
    const tagged=qa('body *').filter(el=>isVisible(el)&&/club\.agi\.eth|AGI Club direct owner|propriétaire direct.*AGI Club/i.test(safeText(el)));
    for(const seed of tagged){
      let el=seed; while(el&&el!==document.body){const s=getComputedStyle(el);if((s.position==='fixed'||s.position==='sticky'||el.getAttribute('role')==='dialog')&&coverage(el)>.18){S.gate=el;return el;}el=el.parentElement;}
    }
    const stack=document.elementsFromPoint(Math.floor(innerWidth/2),Math.floor(innerHeight/2));
    for(const el of stack){if(el!==document.body&&isVisible(el)&&coverage(el)>.35&&/AGI Club|club\.agi\.eth/i.test(safeText(el))){S.gate=el;return el;}}
    if(tagged[0]){let el=tagged[0];while(el.parentElement&&el.parentElement!==document.body&&coverage(el.parentElement)<.86)el=el.parentElement;S.gate=el;return el;}
    return null;
  }
  function legalLinks(){ return `<a class="gda-link" href="governance/DUAL_ACCESS_CONSTITUTION.html" target="_blank" rel="noopener">${escapeHtml(tx().legal)}</a>`; }
  function cardHtml(){const x=tx(); return `<div class="gda-kicker">${escapeHtml(x.kicker)}</div><h2 class="gda-title">${escapeHtml(x.title)}</h2><p class="gda-rule">${escapeHtml(x.rule)}</p><p class="gda-copy">${escapeHtml(x.copy)}</p><span class="gda-contract"><b>${escapeHtml(x.contract)}:</b> ${C.contract}</span><div class="gda-safety">${escapeHtml(x.safety)}</div><label class="gda-check"><input id="gda-accept" type="checkbox"><span>${escapeHtml(x.accept)}</span></label><label class="gda-check"><input id="gda-boundary" type="checkbox"><span>${escapeHtml(x.boundary)}</span></label><div class="gda-actions"><button class="gda-button" id="gda-verify" type="button">${escapeHtml(x.button)}</button>${legalLinks()}</div><div class="gda-status" id="gda-status" role="status" aria-live="polite"></div><div class="gda-mini">${escapeHtml(x.exportNote)}</div>`;}
  function bindCard(root){
    const b=q('#gda-verify',root); if(b&&!b.dataset.bound){b.dataset.bound='1';b.addEventListener('click',verifyAndGrant);}
  }
  function renderCard(){
    const existing=q('#goalos-dual-access-card'); if(existing){existing.innerHTML=cardHtml();bindCard(existing);return existing;}
    const card=document.createElement('section');card.id='goalos-dual-access-card';card.setAttribute('aria-label',tx().title);card.innerHTML=cardHtml();
    const gate=findGate(); let anchor=null;
    if(gate){
      const seeds=qa('button,input[type="button"],input[type="submit"],a',gate).filter(el=>/connect|verify|wallet|vérifier|portefeuille/i.test(safeText(el)));
      anchor=seeds[0]?.closest('form,.card,article,section,div')||q('form,.card,article,section',gate)||gate;
      const parent=anchor.parentElement||gate;
      parent.appendChild(card);S.embedded=true;
    } else {
      ensureModal();q('.gda-modal-panel',S.modal).appendChild(card);S.embedded=false;
    }
    bindCard(card); ensureLauncher(); return card;
  }
  function ensureLauncher(){
    let b=q('#goalos-dual-access-launcher'); if(!b){b=document.createElement('button');b.id='goalos-dual-access-launcher';b.type='button';document.body.appendChild(b);b.addEventListener('click',()=>{ensureModal();S.modal.hidden=false;q('#gda-verify',S.modal)?.focus();});}
    b.textContent=tx().launcher;b.hidden=S.embedded||!!S.receipt;
  }
  function ensureModal(){
    if(S.modal&&S.modal.isConnected)return S.modal;const m=document.createElement('div');m.id='goalos-dual-access-modal';m.hidden=true;m.setAttribute('role','dialog');m.setAttribute('aria-modal','true');m.innerHTML=`<div class="gda-modal-panel"><button class="gda-modal-close" type="button" aria-label="${escapeHtml(tx().close)}">×</button></div>`;document.body.appendChild(m);m.addEventListener('click',e=>{if(e.target===m)m.hidden=true;});q('.gda-modal-close',m).addEventListener('click',()=>m.hidden=true);document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!m.hidden)m.hidden=true;});S.modal=m;return m;
  }
  function setStatus(message,kind=''){const el=q('#gda-status');if(el){el.textContent=message;el.dataset.kind=kind;}const b=q('#gda-verify');if(b)b.disabled=kind==='busy';}
  function provider(){return window.ethereum&&typeof window.ethereum.request==='function'?window.ethereum:null;}
  function normalizeAddress(v){const a=String(v||'');if(!/^0x[0-9a-fA-F]{40}$/.test(a))throw new Error('INVALID_ADDRESS');return a.toLowerCase();}
  async function ensureMainnet(p){let c=(await p.request({method:'eth_chainId'})).toLowerCase();if(c!==C.chainId){try{await p.request({method:'wallet_switchEthereumChain',params:[{chainId:C.chainId}]});}catch(e){throw Object.assign(new Error('WRONG_NETWORK'),{cause:e});}c=(await p.request({method:'eth_chainId'})).toLowerCase();if(c!==C.chainId)throw new Error('WRONG_NETWORK');}}
  function balanceData(address){return '0x70a08231'+address.slice(2).padStart(64,'0');}
  async function readEligibility(p,address){
    await ensureMainnet(p);const code=await p.request({method:'eth_getCode',params:[C.contract,'latest']});if(!code||code==='0x'||code==='0x0')throw new Error('TOKEN_CONTRACT_UNAVAILABLE');
    const [rawHex,blockHex]=await Promise.all([p.request({method:'eth_call',params:[{to:C.contract,data:balanceData(address)},'latest']}),p.request({method:'eth_blockNumber'})]);
    const raw=BigInt(rawHex||'0x0'),block=BigInt(blockHex||'0x0');return {raw,block,qualifies:raw>=C.threshold};
  }
  function formatUnits(raw){const whole=raw/1000000000000000000n,frac=(raw%1000000000000000000n).toString().padStart(18,'0').slice(0,4).replace(/0+$/,'');return whole.toLocaleString('en-US')+(frac?'.'+frac:'');}
  function random32(){const a=new Uint8Array(32);crypto.getRandomValues(a);return '0x'+[...a].map(x=>x.toString(16).padStart(2,'0')).join('');}
  function toHexUtf8(s){return '0x'+[...new TextEncoder().encode(s)].map(x=>x.toString(16).padStart(2,'0')).join('');}
  async function signReceipt(p,address,e){
    const now=Math.floor(Date.now()/1000),exp=now+Math.floor(C.absoluteMs/1000),nonce=random32(),origin=location.origin+location.pathname;
    const message={accessClass:'AGIALPHA_BALANCE_QUALIFIED',holder:address,token:C.contract,minimumBalance:C.threshold.toString(),observedBalance:e.raw.toString(),verificationBlock:e.block.toString(),issuedAt:String(now),expiresAt:String(exp),nonce,origin,legalDigest:C.legalDigest,authority:'NONE'};
    const typed={domain:{name:'GoalOS Singularity Navigator Omega',version:'3.3.0-SN5-BI2',chainId:1,verifyingContract:C.contract},primaryType:'GoalOSAccess',types:{EIP712Domain:[{name:'name',type:'string'},{name:'version',type:'string'},{name:'chainId',type:'uint256'},{name:'verifyingContract',type:'address'}],GoalOSAccess:[{name:'accessClass',type:'string'},{name:'holder',type:'address'},{name:'token',type:'address'},{name:'minimumBalance',type:'uint256'},{name:'observedBalance',type:'uint256'},{name:'verificationBlock',type:'uint256'},{name:'issuedAt',type:'uint256'},{name:'expiresAt',type:'uint256'},{name:'nonce',type:'bytes32'},{name:'origin',type:'string'},{name:'legalDigest',type:'bytes32'},{name:'authority',type:'string'}]},message};
    try{return {...message,signature:await p.request({method:'eth_signTypedData_v4',params:[address,JSON.stringify(typed)]}),signatureMethod:'EIP-712',release:C.release,chainId:1,qualifyingContract:C.contract,minimumDisplay:C.thresholdText,observedBalanceFormatted:formatUnits(e.raw),createdAt:new Date(now*1000).toISOString(),expiresAtISO:new Date(exp*1000).toISOString(),localReceiptOnly:true,authorityCreated:'None'};}
    catch(err){if(err&&Number(err.code)===4001)throw err;const personal=`GoalOS Singularity Navigator Ω temporary access receipt\nRelease: ${C.release}\nAccess class: AGIALPHA balance-qualified\nHolder: ${address}\nNetwork: Ethereum Mainnet (1)\nToken: ${C.contract}\nMinimum raw balance: ${C.threshold}\nObserved raw balance: ${e.raw}\nVerification block: ${e.block}\nIssued at: ${now}\nExpires at: ${exp}\nNonce: ${nonce}\nOrigin: ${origin}\nLegal digest: ${C.legalDigest}\nAuthority created: NONE\nNo token approval, transfer, deposit, burn, staking, locking, custody or transaction is authorized.`;let sig;try{sig=await p.request({method:'personal_sign',params:[toHexUtf8(personal),address]});}catch(e2){if(e2&&Number(e2.code)===4001)throw e2;sig=await p.request({method:'personal_sign',params:[address,toHexUtf8(personal)]});}return {...message,signature:sig,signatureMethod:'personal_sign',signedMessage:personal,release:C.release,chainId:1,qualifyingContract:C.contract,minimumDisplay:C.thresholdText,observedBalanceFormatted:formatUnits(e.raw),createdAt:new Date(now*1000).toISOString(),expiresAtISO:new Date(exp*1000).toISOString(),localReceiptOnly:true,authorityCreated:'None'};}
  }
  async function verifyAndGrant(){
    if(S.busy)return;const x=tx(),accept=q('#gda-accept'),boundary=q('#gda-boundary');if(!accept?.checked||!boundary?.checked){setStatus(lang()==='fr'?'Veuillez accepter les deux confirmations avant de continuer.':'Accept both confirmations before continuing.','error');return;}
    const p=provider();if(!p){setStatus(x.noWallet,'error');return;}S.busy=true;setStatus(x.working,'busy');const serial=++S.verificationSerial;
    try{const accounts=await p.request({method:'eth_requestAccounts'});const address=normalizeAddress(accounts&&accounts[0]);const e=await readEligibility(p,address);if(serial!==S.verificationSerial)return;if(!e.qualifies){setStatus(`${x.denied} ${x.currentBalance}: ${formatUnits(e.raw)} AGIALPHA.`,'error');return;}setStatus(x.signing,'busy');const receipt=await signReceipt(p,address,e);const accounts2=await p.request({method:'eth_accounts'});if(normalizeAddress(accounts2&&accounts2[0])!==address)throw new Error('ACCOUNT_CHANGED');const e2=await readEligibility(p,address);if(!e2.qualifies)throw new Error('BALANCE_CHANGED');receipt.observedBalance=e2.raw.toString();receipt.observedBalanceFormatted=formatUnits(e2.raw);receipt.verificationBlock=e2.block.toString();sessionStorage.setItem(C.sessionKey,JSON.stringify(receipt));S.receipt=receipt;setStatus(x.granted,'ok');setTimeout(()=>grant(receipt),220);}
    catch(err){const code=Number(err&&err.code);if(code===4001)setStatus(x.rejected,'error');else if(String(err&&err.message).includes('WRONG_NETWORK'))setStatus(x.wrongNetwork,'error');else setStatus(`${x.failed}${err&&err.message?' ('+err.message+')':''}`,'error');}
    finally{S.busy=false;const b=q('#gda-verify');if(b)b.disabled=false;}
  }
  function hideGate(){
    const gate=findGate();if(gate){S.gate=gate;S.gateRestore={display:gate.style.display,visibility:gate.style.visibility,opacity:gate.style.opacity,pointerEvents:gate.style.pointerEvents,hidden:gate.hidden,ariaHidden:gate.getAttribute('aria-hidden')};gate.classList.add('goalos-dual-access-original-gate');gate.style.setProperty('display','none','important');gate.style.setProperty('visibility','hidden','important');gate.style.setProperty('pointer-events','none','important');gate.hidden=true;gate.setAttribute('aria-hidden','true');}
    document.documentElement.classList.add('goalos-agialpha-unlocked');document.body.classList.add('goalos-agialpha-unlocked','unlocked','access-granted','is-unlocked','sn5-unlocked');document.body.style.removeProperty('overflow');document.documentElement.dataset.goalosAccess='agialpha-balance-qualified';
    const views=qa('section.view,.view[id^="view-"],[data-view]');if(views.length){let parent=views[0].parentElement;while(parent&&parent!==document.body){const s=getComputedStyle(parent);if(s.display==='none')parent.style.setProperty('display','block','important');if(s.visibility==='hidden')parent.style.setProperty('visibility','visible','important');if(parent.hasAttribute('hidden'))parent.hidden=false;if(parent.hasAttribute('inert'))parent.removeAttribute('inert');parent=parent.parentElement;}let active=views.find(v=>v.classList.contains('active')&&isVisible(v));if(!active){const preferred=q('[data-route="command"],[data-route="opportunity"],[data-route="mission"]');if(preferred)preferred.click();else{active=views.find(v=>/command|opportunity|mission|home/i.test(v.id))||views[0];active.classList.add('active');active.hidden=false;}}}
    for(const el of qa('[inert]')){if(el.querySelector?.('section.view,.view[id^="view-"]'))el.removeAttribute('inert');}
    for(const name of ['goalos:access-granted','goalos-access-granted','access:granted','sn5:unlocked'])window.dispatchEvent(new CustomEvent(name,{detail:S.receipt}));
  }
  function statusPill(){let p=q('#goalos-dual-access-status');if(!p){p=document.createElement('div');p.id='goalos-dual-access-status';document.body.appendChild(p);}const x=tx(),exp=new Date(Number(S.receipt.expiresAt)*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});p.innerHTML=`<span class="gda-dot" aria-hidden="true"></span><span>${escapeHtml(x.status)} · <span class="gda-balance">${escapeHtml(S.receipt.observedBalanceFormatted)} AGIALPHA</span> · ${escapeHtml(x.expires)} ${escapeHtml(exp)}</span><button type="button" data-gda="receipt">${escapeHtml(x.receipt)}</button><button type="button" data-gda="lock">${escapeHtml(x.lock)}</button>`;q('[data-gda="receipt"]',p).addEventListener('click',downloadReceipt);q('[data-gda="lock"]',p).addEventListener('click',()=>lock('manual',true));}
  function grant(receipt){S.receipt=receipt;hideGate();if(S.modal)S.modal.hidden=true;q('#goalos-dual-access-launcher')?.setAttribute('hidden','');statusPill();S.lastActivity=Date.now();installMonitors();insertLegalNotice();}
  function downloadReceipt(){if(!S.receipt)return;const data=JSON.stringify({...S.receipt,notice:'Local access receipt only. It creates no AGI Club membership, institutional authority, financial right, professional opinion or external entitlement.'},null,2),a=document.createElement('a');a.href=URL.createObjectURL(new Blob([data],{type:'application/json'}));a.download=`GoalOS_AGIALPHA_Access_Receipt_${S.receipt.holder.slice(0,8)}_${C.release}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
  function restoreGate(){const g=S.gate;if(g&&g.isConnected&&S.gateRestore){g.classList.remove('goalos-dual-access-original-gate');g.style.display=S.gateRestore.display;g.style.visibility=S.gateRestore.visibility;g.style.opacity=S.gateRestore.opacity;g.style.pointerEvents=S.gateRestore.pointerEvents;g.hidden=S.gateRestore.hidden;if(S.gateRestore.ariaHidden===null)g.removeAttribute('aria-hidden');else g.setAttribute('aria-hidden',S.gateRestore.ariaHidden);}document.documentElement.classList.remove('goalos-agialpha-unlocked');document.body.classList.remove('goalos-agialpha-unlocked','unlocked','access-granted','is-unlocked','sn5-unlocked');delete document.documentElement.dataset.goalosAccess;q('#goalos-dual-access-status')?.remove();S.receipt=null;renderCard();ensureLauncher();}
  function lock(reason,reload=false){sessionStorage.removeItem(C.sessionKey);sessionStorage.setItem(C.lockReasonKey,reason||'eligibility');clearInterval(S.timer);clearInterval(S.idleTimer);S.timer=S.idleTimer=null;restoreGate();if(reload)setTimeout(()=>location.reload(),60);}
  async function revalidate(source='timer'){
    if(!S.receipt)return false;const now=Math.floor(Date.now()/1000);if(now>=Number(S.receipt.expiresAt)){lock('expired',source!=='test');return false;}const p=provider();if(!p){lock('provider',source!=='test');return false;}try{const chain=(await p.request({method:'eth_chainId'})).toLowerCase(),accounts=await p.request({method:'eth_accounts'});if(chain!==C.chainId||normalizeAddress(accounts&&accounts[0])!==normalizeAddress(S.receipt.holder))throw new Error('ACCOUNT_OR_CHAIN');const e=await readEligibility(p,S.receipt.holder);if(!e.qualifies)throw new Error('BALANCE');S.receipt.observedBalance=e.raw.toString();S.receipt.observedBalanceFormatted=formatUnits(e.raw);S.receipt.lastRevalidatedAt=new Date().toISOString();S.receipt.lastVerificationBlock=e.block.toString();sessionStorage.setItem(C.sessionKey,JSON.stringify(S.receipt));statusPill();return true;}catch(err){lock('revalidation:'+String(err&&err.message||err),source!=='test');return false;}}
  function installMonitors(){if(S.timer)clearInterval(S.timer);if(S.idleTimer)clearInterval(S.idleTimer);S.timer=setInterval(()=>revalidate('timer'),C.revalidateMs);S.idleTimer=setInterval(()=>{if(S.receipt&&Date.now()-S.lastActivity>=C.idleMs)lock('inactivity',true);},15000);for(const ev of ['pointerdown','keydown','touchstart','scroll'])window.addEventListener(ev,()=>S.lastActivity=Date.now(),{passive:true});window.addEventListener('focus',()=>S.receipt&&revalidate('focus'));document.addEventListener('visibilitychange',()=>{if(!document.hidden&&S.receipt)revalidate('visibility');});const p=provider();if(p&&typeof p.on==='function'&&!S.providerHandlers){p.on('accountsChanged',()=>S.receipt&&lock('account-changed',true));p.on('chainChanged',()=>S.receipt&&lock('network-changed',true));p.on('disconnect',()=>S.receipt&&lock('wallet-disconnected',true));S.providerHandlers=true;}document.addEventListener('click',async e=>{if(!S.receipt)return;const el=e.target.closest?.('a,button');if(!el)return;const s=(safeText(el)+' '+(el.getAttribute('download')||'')+' '+(el.getAttribute('href')||'')).toLowerCase();if(/export|evidence bundle|proof bundle|receipt pack|exporter|télécharger.*preuve/.test(s)){e.preventDefault();e.stopImmediatePropagation();if(await revalidate('protected-export'))el.click();}},true);}
  async function restore(){const raw=sessionStorage.getItem(C.sessionKey);if(!raw)return false;try{const r=JSON.parse(raw);if(r.release!==C.release||String(r.qualifyingContract||r.token).toLowerCase()!==C.contractLower||r.accessClass!=='AGIALPHA_BALANCE_QUALIFIED'||Date.now()/1000>=Number(r.expiresAt))throw new Error('STALE');S.receipt=r;const ok=await revalidate('restore');if(ok)grant(r);return ok;}catch(e){sessionStorage.removeItem(C.sessionKey);return false;}}
  function insertLegalNotice(){
    if(q('#goalos-dual-legal-notice'))return;const host=q('#view-legal .card,#view-legal,section[id*="legal" i],.legal-pane')||null;if(!host)return;const n=document.createElement('aside');n.id='goalos-dual-legal-notice';n.innerHTML=lang()==='fr'?`<h3>Constitution d’accès double · ${C.release}</h3><p><b>Accès :</b> propriétaire direct actuel de l’AGI Club <b>OU</b> détenteur direct actuel d’au moins 1 000 000 AGIALPHA officiels sur le réseau principal Ethereum.</p><p>La voie AGIALPHA est une vérification de solde en lecture seule. Elle ne crée aucune adhésion à l’AGI Club, aucune autorité, aucun droit financier ou commercial et aucun accès permanent.</p>${legalLinks()}`:`<h3>Dual-Access Constitution · ${C.release}</h3><p><b>Access:</b> current AGI Club direct owner <b>OR</b> current direct holder of at least 1,000,000 official AGIALPHA on Ethereum Mainnet.</p><p>The AGIALPHA route is a read-only balance qualification. It creates no AGI Club membership, authority, financial or commercial rights, or permanent access.</p>${legalLinks()}`;host.prepend(n);
  }
  function updateLanguage(){const c=q('#goalos-dual-access-card');if(c){c.innerHTML=cardHtml();bindCard(c);}const l=q('#goalos-dual-access-launcher');if(l)l.textContent=tx().launcher;if(S.receipt)statusPill();}
  async function boot(){ensureModal();renderCard();insertLegalNotice();const reason=sessionStorage.getItem(C.lockReasonKey);if(reason){sessionStorage.removeItem(C.lockReasonKey);setStatus(tx().locked,'error');}await restore();new MutationObserver(()=>{if(!S.receipt){if(!q('#goalos-dual-access-card'))renderCard();}updateLanguage();insertLegalNotice();}).observe(document.documentElement,{attributes:true,attributeFilter:['lang'],childList:true,subtree:false});setTimeout(()=>{if(!S.receipt&&!q('#goalos-dual-access-card'))renderCard();},700);setTimeout(()=>{if(!S.receipt&&!q('#goalos-dual-access-card'))renderCard();},1800);}
  window.GoalOSDualAccess=Object.freeze({config:C,verify:verifyAndGrant,revalidate:()=>revalidate('test'),lock:()=>lock('manual',false),getReceipt:()=>S.receipt,render:renderCard});
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot,{once:true});else boot();
})();
