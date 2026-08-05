(() => {
  const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
  const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
  const isFr=document.documentElement.lang.toLowerCase().startsWith('fr');

  // Scroll progress
  const progress=$('[data-crown-progress]');
  if(progress){
    const update=()=>{const max=document.documentElement.scrollHeight-innerHeight;const p=max>0?scrollY/max:0;progress.style.width=`${Math.min(100,Math.max(0,p*100))}%`;};
    addEventListener('scroll',update,{passive:true});addEventListener('resize',update);update();
  }

  // Reveal
  if(!reduced&&'IntersectionObserver' in window){
    const obs=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){e.target.dataset.crownRevealed='true';obs.unobserve(e.target);}}),{threshold:.1});
    $$('[data-crown-reveal]').forEach(el=>obs.observe(el));
  }else $$('[data-crown-reveal]').forEach(el=>el.dataset.crownRevealed='true');

  // Sovereign dial
  const dialData=isFr?{
    observe:{number:'01 · Observer',title:'Voir la réalité avant qu’elle devienne évidente.',body:'Suivre les capacités, les coûts, la réglementation, la preuve, la demande et la dérive institutionnelle. L’intelligence peut chercher largement; l’action commence seulement avec un objectif conséquent et une autorité responsable.'},
    govern:{number:'02 · Gouverner',title:'Constituer l’autorité avant l’exécution.',body:'Figer la mission, les actions permises, le fardeau de preuve, l’autorité d’acceptation, les droits d’arrêt et les actes interdits. Aucun système n’élargit silencieusement son propre mandat.'},
    prove:{number:'03 · Prouver',title:'Faire mériter la croyance aux assertions conséquentes.',body:'Transformer les assertions non étayées en dette de preuve, AGI Jobs bornés, ProofBundles, Evidence Dockets et contestation indépendante. L’exécution n’est jamais son propre verdict.'},
    compound:{number:'04 · Composer',title:'Préserver uniquement ce qui a survécu.',body:'Chronicle admet la capacité acceptée, attribuable et libérée de droits. Une nouvelle mission successeure doit démontrer l’amélioration avant que l’institution puisse affirmer qu’elle a évolué.'}
  }:{
    observe:{number:'01 · Observe',title:'See reality before it becomes obvious.',body:'Track capability, cost, regulation, evidence, demand and institutional drift. Intelligence searches broadly; institutional action begins only when a consequential objective and responsible authority exist.'},
    govern:{number:'02 · Govern',title:'Constitute authority before execution.',body:'Freeze the mission, permitted actions, evidence burden, acceptance authority, stop rights and prohibited acts. No system silently expands its own mandate.'},
    prove:{number:'03 · Prove',title:'Make consequential claims earn belief.',body:'Convert unsupported assertions into Proof Debt, bounded AGI Jobs, ProofBundles, Evidence Dockets and independent challenge. Execution is never its own verdict.'},
    compound:{number:'04 · Compound',title:'Preserve only what survived.',body:'Chronicle admits accepted, attributable and rights-cleared capability. A fresh successor mission must demonstrate improvement before the institution may claim it evolved.'}
  };  const dialButtons=$$('[data-dial]');
  const dialNumber=$('[data-dial-number]'),dialTitle=$('[data-dial-title]'),dialBody=$('[data-dial-body]');
  const setDial=key=>{const d=dialData[key];if(!d)return;dialButtons.forEach(b=>b.setAttribute('aria-selected',String(b.dataset.dial===key)));if(dialNumber)dialNumber.textContent=d.number;if(dialTitle)dialTitle.textContent=d.title;if(dialBody)dialBody.textContent=d.body;};
  dialButtons.forEach((b,i)=>{b.setAttribute('role','tab');b.tabIndex=i===0?0:-1;b.addEventListener('click',()=>{dialButtons.forEach(x=>x.tabIndex=x===b?0:-1);setDial(b.dataset.dial)});b.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(e.key))return;e.preventDefault();let n=i;if(['ArrowRight','ArrowDown'].includes(e.key))n=(i+1)%dialButtons.length;if(['ArrowLeft','ArrowUp'].includes(e.key))n=(i-1+dialButtons.length)%dialButtons.length;if(e.key==='Home')n=0;if(e.key==='End')n=dialButtons.length-1;dialButtons.forEach((x,j)=>x.tabIndex=j===n?0:-1);dialButtons[n].focus();setDial(dialButtons[n].dataset.dial);});});
  if(dialButtons.length)setDial(dialButtons[0].dataset.dial);

  // Hero living graph canvas
  const heroCanvas=$('[data-crown-canvas]');
  const animateGraph=(canvas,opts={})=>{
    if(!canvas)return;
    const ctx=canvas.getContext('2d');let w=0,h=0,dpr=1,raf=0,active=true;
    const count=opts.count||48;let nodes=[];let pointer={x:-9999,y:-9999};
    const resize=()=>{const r=canvas.getBoundingClientRect();dpr=Math.min(devicePixelRatio||1,2);w=Math.max(1,r.width);h=Math.max(1,r.height);canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);nodes=Array.from({length:Math.max(16,Math.round(count*Math.min(1,w/1200)))},(_,i)=>({x:Math.random()*w,y:Math.random()*h,vx:(Math.random()-.5)*.13,vy:(Math.random()-.5)*.13,r:i%11===0?2.4:1.25,p:i%7===0?1:0}));};
    const draw=()=>{ctx.clearRect(0,0,w,h);const link=opts.link||170;for(const n of nodes){n.x+=n.vx;n.y+=n.vy;if(n.x<0||n.x>w)n.vx*=-1;if(n.y<0||n.y>h)n.vy*=-1;const dx=n.x-pointer.x,dy=n.y-pointer.y,dist=Math.hypot(dx,dy);if(dist<170&&dist>1){n.x+=dx/dist*.12;n.y+=dy/dist*.12;}}
      ctx.lineWidth=.65;for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){const a=nodes[i],b=nodes[j],d=Math.hypot(a.x-b.x,a.y-b.y);if(d<link){ctx.strokeStyle=`rgba(165,105,18,${(1-d/link)*.16})`;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}}
      for(const n of nodes){ctx.fillStyle=n.p?'rgba(177,111,15,.72)':'rgba(12,47,78,.52)';ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,Math.PI*2);ctx.fill();if(n.p){ctx.strokeStyle='rgba(231,194,111,.24)';ctx.beginPath();ctx.arc(n.x,n.y,n.r+7,0,Math.PI*2);ctx.stroke();}}
      if(active&&!reduced)raf=requestAnimationFrame(draw);
    };
    const move=e=>{const r=canvas.getBoundingClientRect();pointer={x:e.clientX-r.left,y:e.clientY-r.top};};
    canvas.parentElement?.addEventListener('pointermove',move,{passive:true});canvas.parentElement?.addEventListener('pointerleave',()=>pointer={x:-9999,y:-9999});
    const ro=new ResizeObserver(resize);ro.observe(canvas);resize();draw();
    document.addEventListener('visibilitychange',()=>{active=!document.hidden;if(active&&!reduced)draw();else cancelAnimationFrame(raf);});
  };
  animateGraph(heroCanvas,{count:70,link:185});

  // GoalOS constitutional spine
  const spineData=isFr?{
    direction:{index:'01 · Direction',title:'Découvrir ce qui compte avant que la demande existe.',body:'GoalOS Attracts et Navigator Ω détectent où les décisions conséquentes dépassent leur preuve, puis identifient le plus petit test capable de modifier l’action.',law:'Chercher n’est pas obtenir la permission.',failure:'Échec évité : hypothèses périmées et demande synthétique.'},
    constitution:{index:'02 · Constitution',title:'Figer la mission et son autorité.',body:'La Mission Constitution établit l’objectif, la portée, les rôles, les limites de risque, le fardeau de preuve, l’acceptation responsable, les droits d’arrêt et les actions interdites.',law:'Aucune autonomie sans Enveloppe d’autorité.',failure:'Échec évité : extension du mandat et responsabilité ambiguë.'},
    formation:{index:'03 · Formation',title:'Composer l’institution avant qu’elle agisse.',body:'GJFFI Ω et Global Launch Ω cartographient la juridiction, le capital, le talent, l’infrastructure, l’approvisionnement et les portes transactionnelles nécessaires.',law:'Un plan n’est pas une institution.',failure:'Échec évité : stratégie plausible mais inexécutable.'},
    execution:{index:'04 · Exécution',title:'Construire un graphe typé de travail borné.',body:'Modèles, agents, outils et opérateurs humains exécutent des AGI Jobs sur des arêtes déclarées de données, autorité, preuve, ressources et gouvernance—avec isolation avant fusion.',law:'Les nœuds travaillent; les arêtes portent le sens.',failure:'Échec évité : mille agents qui se trompent ensemble.'},
    proof:{index:'05 · Preuve',title:'Faire survivre les assertions matérielles à la contestation.',body:'Les assertions non étayées deviennent dette de preuve. ProofBundles, preuve négative, rejeu et revue indépendante entrent dans un Evidence Docket avant toute confiance conséquente.',law:'Exécuter ≠ Valider ≠ Accepter.',failure:'Échec évité : sortie éloquente confondue avec l’achèvement.'},
    memory:{index:'06 · Mémoire',title:'Laisser Chronicle décider ce qui peut survivre.',body:'Seule une capacité acceptée, attribuable, libérée de droits, délimitée, versionnée et révocable peut entrer dans la mémoire institutionnelle et le Validated Skill Graph.',law:'La mémoire est une décision d’admission.',failure:'Échec évité : hallucination et contexte expiré devenant précédent.'},
    successor:{index:'07 · Successeur',title:'Faire mériter son nom à la génération suivante.',body:'Une mission adjacente nouvelle, sous contraintes égales ou plus strictes, mesure si la capacité retenue améliore réellement le coût, le temps, la qualité, l’intervention et la discipline de preuve.',law:'Pas de Mission 2, pas de composition.',failure:'Échec évité : auto-amélioration par affirmation.'}
  }:{
    direction:{index:'01 · Direction',title:'Discover what matters before the request exists.',body:'GoalOS Attracts and Navigator Ω detect where consequential decisions are outrunning their evidence, then identify the smallest proof capable of changing action.',law:'Search is not permission.',failure:'Failure prevented: stale assumptions and synthetic demand.'},
    constitution:{index:'02 · Constitution',title:'Freeze the mission and its authority.',body:'The Mission Constitution establishes objective, scope, roles, risk limits, evidence burden, responsible acceptance, stop rights and prohibited actions.',law:'No autonomy without an Authority Envelope.',failure:'Failure prevented: mandate expansion and ambiguous accountability.'},
    formation:{index:'03 · Formation',title:'Compose the institution before it acts.',body:'GJFFI Ω and Global Launch Ω map jurisdiction, capital, talent, infrastructure, procurement and the transaction gates required to make the objective executable.',law:'A plan is not an institution.',failure:'Failure prevented: technically plausible but inoperable strategy.'},
    execution:{index:'04 · Execution',title:'Build a typed graph of bounded work.',body:'Models, agents, tools and human operators execute custom AGI Jobs through declared data, authority, proof, resource and governance edges—with isolation before merge.',law:'Nodes perform work; edges carry meaning.',failure:'Failure prevented: a thousand agents being wrong together.'},
    proof:{index:'05 · Proof',title:'Make material claims survive challenge.',body:'Unsupported claims become Proof Debt. ProofBundles, negative evidence, replay and independent review enter an Evidence Docket before any consequential reliance.',law:'Execute ≠ Validate ≠ Accept.',failure:'Failure prevented: fluent output mistaken for completion.'},
    memory:{index:'06 · Memory',title:'Let Chronicle decide what may survive.',body:'Only accepted, attributable, rights-cleared, scoped, versioned and revocable capability may enter institutional memory and the Validated Skill Graph.',law:'Memory is an admission decision.',failure:'Failure prevented: hallucination and expired context becoming precedent.'},
    successor:{index:'07 · Successor',title:'Make the next generation earn its name.',body:'A fresh adjacent mission under equal or tighter constraints measures whether retained capability genuinely improves cost, time, quality, intervention and evidence discipline.',law:'No Mission 2, no compounding.',failure:'Failure prevented: self-improvement by assertion.'}
  };  const spineTabs=$$('[data-spine-stage]');
  const spineIndex=$('[data-spine-index]'),spineTitle=$('[data-spine-title]'),spineBody=$('[data-spine-body]'),spineLaw=$('[data-spine-law]'),spineFailure=$('[data-spine-failure]');
  const setSpine=key=>{const d=spineData[key];if(!d)return;spineTabs.forEach(t=>t.setAttribute('aria-selected',String(t.dataset.spineStage===key)));if(spineIndex)spineIndex.textContent=d.index;if(spineTitle)spineTitle.textContent=d.title;if(spineBody)spineBody.textContent=d.body;if(spineLaw)spineLaw.textContent=d.law;if(spineFailure)spineFailure.textContent=d.failure;};
  spineTabs.forEach((t,i)=>{t.setAttribute('role','tab');t.tabIndex=i===0?0:-1;t.addEventListener('click',()=>{spineTabs.forEach(x=>x.tabIndex=x===t?0:-1);setSpine(t.dataset.spineStage)});t.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();let n=i;if(e.key==='ArrowRight')n=(i+1)%spineTabs.length;if(e.key==='ArrowLeft')n=(i-1+spineTabs.length)%spineTabs.length;if(e.key==='Home')n=0;if(e.key==='End')n=spineTabs.length-1;spineTabs.forEach((x,j)=>x.tabIndex=j===n?0:-1);spineTabs[n].focus();setSpine(spineTabs[n].dataset.spineStage);});});
  if(spineTabs.length)setSpine(spineTabs[0].dataset.spineStage);
  animateGraph($('[data-spine-canvas]'),{count:46,link:145});

  // Proof gradient
  const gradientData=isFr?{
    output:{title:'Sortie candidate',body:'Un modèle, un agent ou un opérateur produit une assertion, un plan, une base de code, une analyse ou une action proposée. Elle ne possède aucune autorité automatique.'},
    claim:{title:'Assertion matérielle',body:'La mission enregistre ce qui doit être vrai pour que la sortie compte et convertit les parties non étayées en dette de preuve explicite.'},
    evidence:{title:'Preuve assemblée',body:'Des AGI Jobs sur mesure rapportent sources, tests, traces, preuve négative, provenance et historique de réparation dans des ProofBundles.'},
    challenge:{title:'Contestation indépendante',body:'Des réviseurs sans conflit rejouent, attaquent, comparent et exposent les contradictions sans remplacer le client responsable.'},
    acceptance:{title:'Acceptation responsable',body:'Le client ou l’autorité autorisée décide ACCEPTER, RÉPARER ou REJETER et conserve la responsabilité de l’usage conséquent.'},
    capability:{title:'Capacité admise par Chronicle',body:'Seule une capacité acceptée, délimitée, attribuable et libérée de droits peut entrer dans Chronicle, être versionnée et subir un test successeur.'}
  }:{
    output:{title:'Candidate output',body:'A model, agent or operator produces a claim, plan, codebase, analysis or proposed action. It has no automatic authority.'},
    claim:{title:'Material claim',body:'The mission records what must be true for the output to matter and converts unsupported portions into explicit Proof Debt.'},
    evidence:{title:'Evidence assembled',body:'Custom AGI Jobs return sources, tests, traces, negative evidence, provenance and repair history inside ProofBundles.'},
    challenge:{title:'Independent challenge',body:'Conflict-cleared reviewers replay, attack, compare and surface contradictions without replacing the responsible customer.'},
    acceptance:{title:'Accountable acceptance',body:'The authorized customer or authority decides ACCEPT, REPAIR or REJECT and retains responsibility for consequential use.'},
    capability:{title:'Chronicle-admitted capability',body:'Only accepted, scoped, attributable and rights-cleared capability may enter Chronicle, be versioned and face a fresh successor test.'}
  };  const gradientButtons=$$('[data-crown-gradient]');const gradientTitle=$('[data-crown-gradient-title]'),gradientBody=$('[data-crown-gradient-body]');
  const setGradient=key=>{const d=gradientData[key];if(!d)return;gradientButtons.forEach(b=>b.setAttribute('aria-selected',String(b.dataset.crownGradient===key)));if(gradientTitle)gradientTitle.textContent=d.title;if(gradientBody)gradientBody.textContent=d.body;};
  gradientButtons.forEach((b,i)=>{b.setAttribute('role','tab');b.tabIndex=i===0?0:-1;b.addEventListener('click',()=>{gradientButtons.forEach(x=>x.tabIndex=x===b?0:-1);setGradient(b.dataset.crownGradient)});b.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();let n=i;if(e.key==='ArrowRight')n=(i+1)%gradientButtons.length;if(e.key==='ArrowLeft')n=(i-1+gradientButtons.length)%gradientButtons.length;if(e.key==='Home')n=0;if(e.key==='End')n=gradientButtons.length-1;gradientButtons.forEach((x,j)=>x.tabIndex=j===n?0:-1);gradientButtons[n].focus();setGradient(gradientButtons[n].dataset.crownGradient);});});
  if(gradientButtons.length)setGradient(gradientButtons[0].dataset.crownGradient);
})();
