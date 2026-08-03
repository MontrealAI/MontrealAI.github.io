(() => {
  'use strict';
  const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
  const fr=(document.documentElement.lang||'').toLowerCase().startsWith('fr');

  const graph=$('[data-v120-graph]');
  if(graph){
    const toolbar=$('[data-v120-graph-toolbar]',graph);
    const detailTitle=$('[data-v120-detail-title]',graph);
    const detailCopy=$('[data-v120-detail-copy]',graph);
    toolbar?.addEventListener('click',e=>{
      const b=e.target.closest('button[data-filter]'); if(!b)return;
      graph.dataset.filter=b.dataset.filter;
      $$('button[data-filter]',toolbar).forEach(x=>x.classList.toggle('active',x===b));
    });
    const copy=fr?{
      objective:['Objectif','La mission conséquente, le principal responsable et le test d’acceptation sont figés avant l’exécution.'],
      dependency:['Graphe de dépendances','Les dépendances réelles, les chemins critiques et les travaux véritablement indépendants sont déclarés.'],
      intelligence:['Intelligence spécialisée','Chaque nœud reçoit le contexte, les outils et la compétence nécessaires — rien de plus.'],
      isolation:['Exécution isolée','Les nœuds à risque sont séparés avant toute fusion; l’autorité reste bornée et révocable.'],
      validation:['Validation indépendante','Une nouvelle perspective tente de falsifier les affirmations et de découvrir les lacunes.'],
      docket:['Evidence Docket','Les affirmations, la provenance, les essais, les échecs, les coûts, les droits et l’incertitude sont réunis.'],
      chronicle:['Chronicle','L’institution responsable décide ce qui peut être mémorisé, limité, révoqué ou oublié.'],
      skill:['Graphe de capacités validées','Seule une capacité délimitée, prouvée, versionnée et assortie de droits peut être réutilisée.'],
      reuse:['Invocation protégée','La capacité privée peut être invoquée sous une autorité vérifiable sans être livrée ni copiée.'],
      capital:['Valeur et capital','La valeur acceptée finance des évaluateurs, des outils, de l’infrastructure et une mission plus difficile.'],
      parent:['MONTREAL.AI · GoalOS','La constitution demeure au centre; le graphe ne peut pas se déclarer lui-même réussi ni étendre sa propre autorité.']
    }:{
      objective:['Objective','The consequential mission, accountable principal and acceptance test are frozen before execution.'],
      dependency:['Dependency graph','Real dependencies, critical paths and genuinely independent work are declared.'],
      intelligence:['Specialized intelligence','Each node receives the context, tools and capability it needs—nothing more.'],
      isolation:['Isolated execution','Risk-bearing nodes are separated before merge; authority remains bounded and revocable.'],
      validation:['Independent validation','A fresh perspective attempts to falsify claims and expose missing evidence.'],
      docket:['Evidence Docket','Claims, provenance, tests, failures, costs, rights and unresolved uncertainty are assembled.'],
      chronicle:['Chronicle','The responsible institution decides what may be remembered, limited, revoked or forgotten.'],
      skill:['Validated Skill Graph','Only scoped, proven, versioned and rights-cleared capability may be reused.'],
      reuse:['Protected invocation','Private capability may be invoked under verifiable authority without being delivered or copied.'],
      capital:['Value and capital','Accepted value finances better evaluators, tools, infrastructure and a harder mission.'],
      parent:['MONTREAL.AI · GoalOS','The constitution remains central; the graph cannot certify itself or expand its own authority.']
    };
    $$('[data-v120-node]',graph).forEach(n=>{
      const activate=()=>{ $$('[data-v120-node]',graph).forEach(x=>x.classList.toggle('active',x===n)); const v=copy[n.dataset.v120Node]; if(v){detailTitle.textContent=v[0];detailCopy.textContent=v[1];} };
      n.addEventListener('click',activate); n.addEventListener('focus',activate);
    });
  }

  const tabs=$$('[data-v120-tablist]');
  tabs.forEach(list=>{
    const buttons=$$('[role="tab"]',list); const panelSelector=list.dataset.panelTarget; const panel=$(panelSelector);
    const data=fr?{
      chat:['Conversation → mission','Répondre avec preuve lorsque la réponse suffit; constituer un AGI Job lorsque l’action est requise.'],
      attract:['Changement → objectif autorisé','Détecter l’incertitude coûteuse, calculer la valeur de la preuve et n’engager qu’un principal légitime.'],
      graph:['Objectif → graphe gouverné','Concevoir un graphe de travail, d’autorité, de preuve, de mémoire et de capital dont chaque arête a un sens déclaré.'],
      commercial:['Offre → revenu vérifié','Qualifier, conclure, livrer, prouver, renouveler et réinvestir uniquement la valeur acceptée par la réalité.'],
      skill:['Preuve → capacité','Chronicle n’admet que ce qui a réussi; une nouvelle génération doit encore le prouver sur une mission fraîche.'],
      value:['Décision → valeur mesurable','Rendre explicites, modifiables et falsifiables les hypothèses économiques avant qu’elles méritent l’autorité.']
    }:{
      chat:['Conversation → mission','Answer with evidence when an answer is sufficient; constitute an AGI Job when action is required.'],
      attract:['World change → authorized objective','Detect expensive uncertainty, calculate the value of proof and engage only a legitimate principal.'],
      graph:['Objective → governed graph','Design a graph of work, authority, proof, memory and capital in which every edge carries declared meaning.'],
      commercial:['Offer → verified revenue','Qualify, contract, deliver, prove, renew and reinvest only value that reality accepted.'],
      skill:['Proof → capability','Chronicle admits only what passed; a new generation must still earn superiority on a fresh mission.'],
      value:['Decision → measurable value','Make economic assumptions explicit, editable and falsifiable before they deserve authority.']
    };
    const activate=(btn)=>{
      buttons.forEach((b,i)=>{const on=b===btn;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1;});
      if(panel){const v=data[btn.dataset.v120Tab];panel.querySelector('b').textContent=v?.[0]||'';panel.querySelector('p').textContent=v?.[1]||'';}
    };
    buttons.forEach((b,i)=>{
      b.addEventListener('click',()=>activate(b));
      b.addEventListener('keydown',e=>{let j=i;if(e.key==='ArrowRight'||e.key==='ArrowDown')j=(i+1)%buttons.length;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')j=(i-1+buttons.length)%buttons.length;else if(e.key==='Home')j=0;else if(e.key==='End')j=buttons.length-1;else return;e.preventDefault();activate(buttons[j]);buttons[j].focus();});
    });
  });
})();

// Institutional navigation and modal accessibility refinements.
(() => {
  'use strict';
  const menuBtn=document.querySelector('.menu-btn');
  const mobile=document.querySelector('.mobile-panel');
  const closeMobile=()=>{if(!mobile?.classList.contains('open'))return;mobile.classList.remove('open');document.body.classList.remove('menu-open');menuBtn?.setAttribute('aria-expanded','false');menuBtn?.focus();};
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&mobile?.classList.contains('open'))closeMobile();});

  const dialog=document.querySelector('.command');
  if(!dialog)return;
  let invoker=null;
  const focusables=()=>[...dialog.querySelectorAll('input,button,a[href],[tabindex]:not([tabindex="-1"])')].filter(x=>!x.disabled&&x.offsetParent!==null);
  const sync=()=>{
    const open=dialog.classList.contains('open');
    dialog.setAttribute('aria-hidden',String(!open));
    document.body.classList.toggle('command-open',open);
    if(open){invoker=invoker||document.activeElement;const input=dialog.querySelector('input');setTimeout(()=>input?.focus(),0);}
    else if(invoker&&document.contains(invoker)){invoker.focus();invoker=null;}
  };
  new MutationObserver(sync).observe(dialog,{attributes:true,attributeFilter:['class']});
  document.querySelectorAll('[data-command]').forEach(b=>b.addEventListener('click',()=>{invoker=b;},{capture:true}));
  dialog.addEventListener('keydown',e=>{
    if(e.key!=='Tab'||!dialog.classList.contains('open'))return;
    const list=focusables();if(!list.length)return;
    const first=list[0],last=list[list.length-1];
    if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus();}
    else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus();}
  });
  sync();
})();
