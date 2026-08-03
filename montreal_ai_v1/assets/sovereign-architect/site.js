(() => {
  const $ = (s, r=document) => r.querySelector(s);
  const $$ = (s, r=document) => [...r.querySelectorAll(s)];
  const year = new Date().getFullYear();
  const isFr = document.documentElement.lang.toLowerCase().startsWith('fr');
  $$('[data-year]').forEach(el => el.textContent = year);

  // Mobile navigation
  const menuButton = $('[data-menu-button]');
  const mobilePanel = $('[data-mobile-panel]');
  const pageRegions = () => $$('main,footer');
  let menuInvoker = null;
  const setMenuBackgroundInert = value => pageRegions().forEach(el => { el.inert = value; });
  const closeMenu = (restoreFocus=true) => {
    if (!mobilePanel || !menuButton) return;
    const wasOpen = mobilePanel.classList.contains('open');
    mobilePanel.classList.remove('open');
    menuButton.setAttribute('aria-expanded','false');
    document.body.classList.remove('menu-open');
    setMenuBackgroundInert(false);
    if (wasOpen && restoreFocus) (menuInvoker || menuButton).focus();
  };
  menuButton?.addEventListener('click', e => {
    const opening = !mobilePanel.classList.contains('open');
    if (!opening) { closeMenu(); return; }
    menuInvoker = e.currentTarget;
    mobilePanel.classList.add('open');
    menuButton.setAttribute('aria-expanded','true');
    document.body.classList.add('menu-open');
    setMenuBackgroundInert(true);
    requestAnimationFrame(() => mobilePanel.querySelector('a')?.focus());
  });
  mobilePanel?.addEventListener('click', e => { if (e.target.closest('a')) closeMenu(false); });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && mobilePanel?.classList.contains('open')) { e.preventDefault(); closeMenu(); }
    if (e.key === 'Tab' && mobilePanel?.classList.contains('open')) {
      const focusable = [menuButton, ...$$('a,button,[tabindex]:not([tabindex="-1"])', mobilePanel)].filter(Boolean).filter(el => !el.disabled);
      if (!focusable.length) return;
      const first=focusable[0], last=focusable[focusable.length-1];
      if (e.shiftKey && document.activeElement===first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement===last) { e.preventDefault(); first.focus(); }
    }
  });

  // Command navigation
  const command = $('[data-command]');
  const commandOpen = $('[data-command-open]');
  const commandClose = $('[data-command-close]');
  const commandInput = $('[data-command-input]');
  const commandResults = $('[data-command-results]');
  let commandInvoker = null;
  const entries = isFr ? [
    ['Accueil','index.html','Institution permanente et Maison phare'],
    ['Institution','institution.html','Doctrine, propriété et continuité de MONTREAL.AI'],
    ['GoalOS','goalos.html','Institution phare gouvernée par la preuve'],
    ['Maisons','maisons.html','Fédération et Foundry'],
    ['Salle de preuve','proof.html','Preuves, dossiers et limites'],
    ['Recherche','research.html','Série canonique GoalOS et publications'],
    ['Fondateur','founder.html','Vincent Boucher — Fondateur et architecte principal'],
    ['Sprint Autorité IA & Preuve','sprint.html','Dix jours pour une architecture de décision gouvernée par la preuve'],
    ['Commission fondatrice','commission.html','Sovereign Genesis 001'],
    ['Juridique et réglementaire','legal.html','Forteresse réglementaire'],
    ['Patrimoine','heritage.html','Dossier public depuis 2003'],
    ['Chronicle','chronicle.html','Publications et dossiers institutionnels'],
    ['Statut','status.html','Registre public du cycle de vie'],
    ['Vérificateur','verify.html','Vérification locale des dossiers']
  ] : [
    ['Home','index.html','Permanent parent and flagship institution'],
    ['Institution','institution.html','MONTREAL.AI doctrine, ownership and continuity'],
    ['GoalOS','goalos.html','Flagship proof-governed intelligence institution'],
    ['Maisons','maisons.html','Federation and Foundry'],
    ['Proof Room','proof.html','Evidence, records and limitations'],
    ['Research','research.html','Canonical GoalOS series and papers'],
    ['Founder','founder.html','Vincent Boucher — Founder and Principal Architect'],
    ['AI Authority & Proof Sprint','sprint.html','Ten business days to a proof-governed deployment decision'],
    ['Founding Commission','commission.html','Sovereign Genesis 001'],
    ['Legal & Regulatory','legal.html','Regulatory Fortress'],
    ['Heritage','heritage.html','Public record since 2003'],
    ['Chronicle','chronicle.html','Institutional releases and records'],
    ['Status','status.html','Public lifecycle ledger'],
    ['Verifier','verify.html','Browser-local record verification']
  ];
  const renderCommand = (q='') => {
    if (!commandResults) return;
    const query = q.trim().toLowerCase();
    const rows = entries.filter(x => !query || `${x[0]} ${x[2]}`.toLowerCase().includes(query));
    commandResults.innerHTML = rows.map(x => `<a href="${x[1]}"><span><b>${x[0]}</b><small>${x[2]}</small></span><span>→</span></a>`).join('') || `<p>${isFr ? 'Aucune surface institutionnelle correspondante.' : 'No matching institutional surface.'}</p>`;
  };
  const commandBackground = () => $$('header,main,footer');
  const setCommandBackgroundInert = value => commandBackground().forEach(el => { el.inert = value; });
  const openCommand = invoker => {
    if (!command) return;
    closeMenu(false);
    commandInvoker = invoker || document.activeElement;
    command.classList.add('open');
    command.setAttribute('aria-hidden','false');
    document.body.classList.add('menu-open');
    setCommandBackgroundInert(true);
    renderCommand('');
    requestAnimationFrame(() => commandInput?.focus());
  };
  const closeCommand = () => {
    if (!command) return;
    const wasOpen = command.classList.contains('open');
    command.classList.remove('open');
    command.setAttribute('aria-hidden','true');
    document.body.classList.remove('menu-open');
    setCommandBackgroundInert(false);
    if (wasOpen) commandInvoker?.focus?.();
  };
  commandOpen?.addEventListener('click', e => openCommand(e.currentTarget));
  commandClose?.addEventListener('click', closeCommand);
  commandInput?.addEventListener('input', e => renderCommand(e.target.value));
  command?.addEventListener('click', e => { if (e.target === command) closeCommand(); });
  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); command?.classList.contains('open') ? closeCommand() : openCommand(commandOpen); }
    if (e.key === 'Escape' && command?.classList.contains('open')) closeCommand();
    if (e.key === 'Tab' && command?.classList.contains('open')) {
      const focusable = $$('a,button,input,[tabindex]:not([tabindex="-1"])', command).filter(el => !el.disabled);
      if (!focusable.length) return;
      const first = focusable[0], last = focusable[focusable.length-1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  });

  // Interactive GoalOS engine
  const engineData = isFr ? {
    direction:{title:'Direction',law:'Observer le monde avant que la demande n’existe.',body:'Navigator Ω surveille les capacités, les coûts, la réglementation, la stratégie et la dérive institutionnelle, puis formule le prochain objectif conséquent.',failure:'Hypothèses périmées, stratégie réactive et objectifs arrivant trop tard.'},
    constitution:{title:'Constitution',law:'Aucune autonomie sans une Enveloppe d’autorité explicite.',body:'GoalOS fige l’objectif, la portée, les rôles, le fardeau de preuve, l’autorité d’acceptation, les droits d’arrêt et les actions interdites.',failure:'Un agent qui élargit silencieusement son propre mandat.'},
    formation:{title:'Formation',law:'L’institution doit être composée avant d’agir.',body:'GJFFI Ω et Global Launch Ω déterminent la juridiction, le capital, le talent, l’infrastructure, l’approvisionnement et les portes transactionnelles requises.',failure:'Un plan techniquement plausible sans voie légale ou opérationnelle.'},
    execution:{title:'Exécution',law:'Les nœuds exécutent; les arêtes portent un sens déclaré.',body:'Chat, agents, outils, opérateurs humains et AGI Jobs exécutent des tâches bornées dans un graphe typé, avec dépendances explicites et isolation avant fusion.',failure:'Mille agents qui se trompent ensemble.'},
    proof:{title:'Preuve',law:'Exécuter n’est pas valider; valider n’est pas accepter.',body:'Les ProofBundles entrent dans un Evidence Docket, des réviseurs indépendants contestent les assertions matérielles et l’institution responsable décide ACCEPTER, RÉPARER ou REJETER.',failure:'Une sortie éloquente confondue avec l’achèvement.'},
    memory:{title:'Mémoire',law:'Seule une capacité acceptée, attribuable et libérée de droits peut survivre.',body:'Office Ω surveille la validité continue. Chronicle admet, versionne, expire ou révoque la capacité. Le Validated Skill Graph gouverne la réutilisation protégée et une nouvelle mission successeure.',failure:'Échec, hallucination ou contexte expiré contaminant la mémoire institutionnelle.'}
  } : {
    direction:{title:'Direction',law:'Observe the world before the request exists.',body:'Navigator Ω monitors capability, cost, regulation, strategy and institutional drift, then frames the next consequential objective.',failure:'Stale assumptions, reactive strategy and objectives that arrive too late.'},
    constitution:{title:'Constitution',law:'No autonomy without an explicit Authority Envelope.',body:'GoalOS freezes the objective, scope, roles, evidence burden, acceptance authority, stop rights and prohibited actions.',failure:'An agent quietly expanding its own mandate.'},
    formation:{title:'Formation',law:'The institution must be composed before it acts.',body:'GJFFI Ω and Global Launch Ω determine jurisdiction, capital, talent, infrastructure, procurement and the transaction gates required to make the mission real.',failure:'A technically plausible plan with no lawful or operational path to execution.'},
    execution:{title:'Execution',law:'Nodes perform work; edges carry declared meaning.',body:'Chat, agents, tools, human operators and AGI Jobs execute bounded tasks inside a typed graph with explicit dependencies and isolation before merge.',failure:'A thousand agents being wrong together.'},
    proof:{title:'Proof',law:'Execution is not validation; validation is not acceptance.',body:'ProofBundles enter an Evidence Docket, independent reviewers challenge material claims and the responsible institution decides ACCEPT, REPAIR or REJECT.',failure:'Fluent output being mistaken for completion.'},
    memory:{title:'Memory',law:'Only accepted, attributable and rights-cleared capability may survive.',body:'Office Ω monitors continuing validity. Chronicle admits, versions, expires or revokes capability. The Validated Skill Graph governs protected reuse and a fresh successor mission.',failure:'Failure, hallucination or expired context contaminating institutional memory.'}
  };
  const engineNodes = $$('[data-engine-node]');
  const enginePanel = $('.engine-detail');
  if (enginePanel) enginePanel.id = enginePanel.id || 'goalos-engine-panel';
  engineNodes.forEach((node,idx) => {
    node.id = node.id || `goalos-engine-tab-${idx+1}`;
    node.setAttribute('aria-controls', enginePanel?.id || 'goalos-engine-panel');
  });
  const updateEngine = key => {
    const d = engineData[key]; if (!d) return;
    engineNodes.forEach(n => {
      const active=n.dataset.engineNode===key;
      n.setAttribute('aria-selected', String(active));
      n.tabIndex=active?0:-1;
      if (active && enginePanel) enginePanel.setAttribute('aria-labelledby',n.id);
    });
    $('[data-engine-title]')?.replaceChildren(document.createTextNode(d.title));
    $('[data-engine-law]')?.replaceChildren(document.createTextNode(d.law));
    $('[data-engine-body]')?.replaceChildren(document.createTextNode(d.body));
    $('[data-engine-failure]')?.replaceChildren(document.createTextNode(d.failure));
  };
  engineNodes.forEach((node, idx) => {
    node.addEventListener('click', () => updateEngine(node.dataset.engineNode));
    node.addEventListener('keydown', e => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) return;
      e.preventDefault();
      let next = idx;
      if (e.key === 'ArrowRight') next = (idx+1)%engineNodes.length;
      if (e.key === 'ArrowLeft') next = (idx-1+engineNodes.length)%engineNodes.length;
      if (e.key === 'Home') next = 0;
      if (e.key === 'End') next = engineNodes.length-1;
      engineNodes[next].focus(); updateEngine(engineNodes[next].dataset.engineNode);
    });
  });
  if (engineNodes.length) updateEngine(engineNodes[0].dataset.engineNode);

  // Proof Gradient
  const gradientData = isFr ? {
    output:{title:'Sortie candidate',body:'Un modèle ou un opérateur produit une assertion, un artefact, un plan, du code ou une proposition d’action. Il ne possède aucune autorité automatique.'},
    claim:{title:'Assertion matérielle',body:'La mission identifie ce qui doit être vrai pour que la sortie compte et inscrit les éléments non étayés comme dette de preuve.'},
    evidence:{title:'Preuve assemblée',body:'Des AGI Jobs sur mesure rapportent sources, tests, traces, preuve négative, provenance et historique de réparation dans des ProofBundles.'},
    challenge:{title:'Contestation indépendante',body:'Des réviseurs sans conflit rejouent, attaquent, comparent et révèlent les contradictions sans remplacer l’acceptation responsable.'},
    acceptance:{title:'Acceptation responsable',body:'Le client ou l’autorité autorisée décide ACCEPTER, RÉPARER ou REJETER et conserve la responsabilité de l’usage conséquent.'},
    capability:{title:'Capacité admise par Chronicle',body:'Seule une capacité acceptée, délimitée, attribuable et libérée de droits peut entrer dans Chronicle, être versionnée et subir un test successeur.'}
  } : {
    output:{title:'Candidate output',body:'A model or operator produces a claim, artifact, plan, codebase or action proposal. It has no automatic authority.'},
    claim:{title:'Material claim',body:'The mission identifies what must be true for the output to matter and records unsupported portions as Proof Debt.'},
    evidence:{title:'Evidence assembled',body:'Custom AGI Jobs return sources, tests, traces, negative evidence, provenance and repair history in ProofBundles.'},
    challenge:{title:'Independent challenge',body:'Conflict-cleared reviewers replay, attack, compare and surface contradictions without replacing responsible acceptance.'},
    acceptance:{title:'Accountable acceptance',body:'The authorized customer or authority decides ACCEPT, REPAIR or REJECT and retains responsibility for consequential use.'},
    capability:{title:'Chronicle-admitted capability',body:'Only accepted, scoped, attributable and rights-cleared capability may enter Chronicle, be versioned and face a fresh successor test.'}
  };
  const gradientSteps = $$('[data-gradient-step]');
  const gradientPanel = $('.gradient-detail');
  if (gradientPanel) gradientPanel.id = gradientPanel.id || 'proof-gradient-panel';
  gradientSteps.forEach((node,idx) => {
    node.setAttribute('role','tab');
    node.id=node.id||`proof-gradient-tab-${idx+1}`;
    node.setAttribute('aria-controls',gradientPanel?.id||'proof-gradient-panel');
  });
  const updateGradient = key => {
    const d = gradientData[key]; if (!d) return;
    gradientSteps.forEach(n => {
      const active=n.dataset.gradientStep===key;
      n.setAttribute('aria-selected',String(active));
      n.tabIndex=active?0:-1;
      if (active&&gradientPanel) gradientPanel.setAttribute('aria-labelledby',n.id);
    });
    $('[data-gradient-title]')?.replaceChildren(document.createTextNode(d.title));
    $('[data-gradient-body]')?.replaceChildren(document.createTextNode(d.body));
  };
  gradientSteps.forEach((n,idx) => {
    n.addEventListener('click', () => updateGradient(n.dataset.gradientStep));
    n.addEventListener('keydown',e => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) return;
      e.preventDefault();let next=idx;
      if(e.key==='ArrowRight')next=(idx+1)%gradientSteps.length;
      if(e.key==='ArrowLeft')next=(idx-1+gradientSteps.length)%gradientSteps.length;
      if(e.key==='Home')next=0;if(e.key==='End')next=gradientSteps.length-1;
      gradientSteps[next].focus();updateGradient(gradientSteps[next].dataset.gradientStep);
    });
  });
  if (gradientSteps.length) updateGradient(gradientSteps[0].dataset.gradientStep);

  // Local mission / commission receipt
  const form = $('[data-receipt-form]');
  const receiptBox = $('[data-receipt]');
  const receiptPre = $('[data-receipt-json]');
  const receiptDigest = $('[data-receipt-digest]');
  const downloadBtn = $('[data-receipt-download]');
  let currentReceipt = null;
  const canonical = value => {
    if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
    if (value && typeof value === 'object') return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`;
    return JSON.stringify(value);
  };
  const sha256Fallback = text => {
    const utf8 = unescape(encodeURIComponent(text));
    const words=[];const bitLen=utf8.length*8;
    for(let i=0;i<utf8.length;i++) words[i>>2]|=utf8.charCodeAt(i)<<(24-(i%4)*8);
    words[bitLen>>5]|=0x80<<(24-bitLen%32);words[((bitLen+64>>9)<<4)+15]=bitLen;
    const k=[0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    let h=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    const rotr=(n,x)=>(x>>>n)|(x<<(32-n));
    for(let i=0;i<words.length;i+=16){const w=new Array(64);for(let t=0;t<16;t++)w[t]=words[i+t]|0;for(let t=16;t<64;t++){const s0=rotr(7,w[t-15])^rotr(18,w[t-15])^(w[t-15]>>>3);const s1=rotr(17,w[t-2])^rotr(19,w[t-2])^(w[t-2]>>>10);w[t]=(w[t-16]+s0+w[t-7]+s1)|0;}let[a,b,c,d,e,f,g,hh]=h;for(let t=0;t<64;t++){const S1=rotr(6,e)^rotr(11,e)^rotr(25,e);const ch=(e&f)^((~e)&g);const temp1=(hh+S1+ch+k[t]+w[t])|0;const S0=rotr(2,a)^rotr(13,a)^rotr(22,a);const maj=(a&b)^(a&c)^(b&c);const temp2=(S0+maj)|0;hh=g;g=f;f=e;e=(d+temp1)|0;d=c;c=b;b=a;a=(temp1+temp2)|0;}h=[(h[0]+a)|0,(h[1]+b)|0,(h[2]+c)|0,(h[3]+d)|0,(h[4]+e)|0,(h[5]+f)|0,(h[6]+g)|0,(h[7]+hh)|0];}
    return h.map(x=>(x>>>0).toString(16).padStart(8,'0')).join('');
  };
  const sha256 = async text => {
    const bytes = new TextEncoder().encode(text);
    if (globalThis.crypto?.subtle) {
      try { const hash=await crypto.subtle.digest('SHA-256',bytes);return [...new Uint8Array(hash)].map(b=>b.toString(16).padStart(2,'0')).join(''); } catch {}
    }
    return sha256Fallback(text);
  };
  form?.addEventListener('submit', async e => {
    e.preventDefault();
    const fd = new FormData(form);
    const objective = String(fd.get('objective')||'').trim();
    const authority = String(fd.get('authority')||'').trim();
    const exposure = String(fd.get('exposure')||'').trim();
    const timeline = String(fd.get('timeline')||'').trim();
    if (!objective || !authority) return;
    const payload = {
      schema:'https://montrealai.github.io/schemas/public-safe-commission-brief-v1',
      kind:'PUBLIC_SAFE_COMMISSION_BRIEF',
      status:'DRAFT_NO_AUTHORITY',
      created_at:new Date().toISOString(),
      publisher:'MONTREAL.AI',
      objective,acceptance_authority:authority,consequence_exposure:exposure||'Not specified',desired_timeline:timeline||'Not specified',
      authority_envelope:isFr?'Cette ébauche locale n’accorde aucune autorité de production, juridique, financière, réglementaire, fiduciaire ou opérationnelle.':'No production, legal, financial, regulatory, fiduciary or operational authority is granted by this browser-local draft.',
      evidence_standard:isFr?'Un engagement écrit distinct doit figer les assertions matérielles, le Evidence Docket, la contestation indépendante et le processus d’acceptation responsable.':'A separate written engagement must freeze the material claims, Evidence Docket, independent challenge and accountable acceptance process.',
      privacy_boundary:isFr?'Contient uniquement des renseignements de qualification publics et sûrs. Aucun contenu confidentiel, privilégié, personnel, classifié, identifiant ou tiers non autorisé.':'Contains public-safe qualification information only. No confidential, privileged, personal, classified, credential or unauthorized third-party material.',
      next_step:isFr?'Demander la voie d’admission protégée approuvée et, au besoin, conclure une entente distincte.':'Request the approved protected intake path and, where appropriate, execute a separate agreement.'
    };
    const serialized = canonical(payload);
    const digest = await sha256(serialized);
    currentReceipt = {...payload,sha256:digest,receipt_id:`MAI-${digest.slice(0,12).toUpperCase()}`};
    if (receiptPre) receiptPre.textContent = JSON.stringify(currentReceipt,null,2);
    if (receiptDigest) receiptDigest.textContent = currentReceipt.receipt_id;
    if (receiptBox) {
      receiptBox.classList.add('active');
      receiptBox.setAttribute('aria-live','polite');
      receiptBox.tabIndex=-1;
      receiptBox.scrollIntoView({behavior:'smooth',block:'nearest'});
      receiptBox.focus({preventScroll:true});
    }
  });
  downloadBtn?.addEventListener('click', () => {
    if (!currentReceipt) return;
    const blob = new Blob([JSON.stringify(currentReceipt,null,2)],{type:'application/json'});
    const url = URL.createObjectURL(blob); const a=document.createElement('a');
    a.href=url;a.download=`${currentReceipt.receipt_id}.json`;a.click();URL.revokeObjectURL(url);
  });

  // Reveal animation
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches && 'IntersectionObserver' in window) {
    const obs = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) { entry.target.dataset.revealed='true'; obs.unobserve(entry.target); }
    }), {threshold:.08});
    $$('[data-reveal]').forEach(el => obs.observe(el));
  } else $$('[data-reveal]').forEach(el => el.dataset.revealed='true');
})();
