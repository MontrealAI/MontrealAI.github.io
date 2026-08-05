'use strict';
(()=>{
  const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
  const FR=document.documentElement.lang.toLowerCase().startsWith('fr');
  const continuum=FR?{
    core:{title:'GoalOS Ω — Intelligence souveraine',text:'Le noyau constitutionnel transforme un objectif en mission, borne l’autorité, exige une preuve inspectable, sépare validation et acceptation, puis gouverne ce qui peut être réutilisé.',href:'sovereign-intelligence.html',cta:'Ouvrir l’intelligence souveraine'},
    navigator:{title:'Navigateur de la Singularité GoalOS Ω',text:'Le Navigateur détecte les changements de frontière, génère des architectures concurrentes et détermine où l’institution doit aller ensuite.',href:'navigator.html',cta:'Entrer dans le Navigateur'},
    office:{title:'Bureau de la Singularité GoalOS Ω',text:'Le Bureau observe, gouverne, prouve, reconstitue et cumule afin que l’institution continue de mériter la direction choisie.',href:'office.html',cta:'Entrer dans le Bureau'},
    launch:{title:'Lancement mondial GoalOS Ω',text:'Lancement mondial compose juridictions, capital, talents, capacités et infrastructure autour d’un objectif — puis gouverne la preuve du résultat.',href:'global-launch.html',cta:'Constituer un lancement mondial'}
  }:{
    core:{title:'GoalOS Ω — Sovereign Intelligence',text:'The constitutional core turns an objective into a mission, bounds authority, requires inspectable evidence, separates validation from acceptance and governs what may be reused.',href:'sovereign-intelligence.html',cta:'Open Sovereign Intelligence'},
    navigator:{title:'GoalOS Singularity Navigator Ω',text:'Navigator detects frontier shifts, generates competing architectures and determines where the institution should go next.',href:'navigator.html',cta:'Enter the Navigator'},
    office:{title:'GoalOS Singularity Office Ω',text:'Office observes, governs, proves, reconstitutes and compounds so the institution keeps deserving the direction it selected.',href:'office.html',cta:'Enter the Office'},
    launch:{title:'GoalOS Global Launch Ω',text:'Global Launch composes jurisdictions, capital, talent, capability and infrastructure around an objective—then governs proof of the result.',href:'global-launch.html',cta:'Constitute a Global Launch'}
  };
  const detail=$('[data-continuum-detail]');
  $$('[data-continuum]').forEach(btn=>btn.addEventListener('click',()=>{
    const v=continuum[btn.dataset.continuum]; if(!v||!detail)return;
    $$('[data-continuum]').forEach(x=>x.classList.toggle('active',x===btn));
    detail.querySelector('h3').textContent=v.title; detail.querySelector('p').textContent=v.text;
    const a=detail.querySelector('a'); a.href=v.href; a.textContent=v.cta+' →';
  }));

  const loop=FR?{
    objective:['Objectif','Définir ce qui doit être préservé, bâti ou devenir — et pourquoi cela compte maintenant.'],
    authority:['Autorité','Figer le principal, les actions permises, les limites, les conditions d’arrêt et l’autorité d’acceptation.'],
    execution:['Exécution','Exécuter uniquement dans la mission constituée, les systèmes approuvés et le budget autorisé.'],
    evidence:['Preuve','Préserver la provenance, les essais, les échecs, les interventions, les coûts et les droits.'],
    validation:['Validation','Inviter une contestation indépendante, la relecture, la dissidence et un verdict borné.'],
    acceptance:['Acceptation','L’institution responsable consigne ACCEPTER, RÉPARER ou REJETER.'],
    chronicle:['Chronicle','N’admettre que des capacités bornées, versionnées, révocables et assorties de droits clairs.'],
    successor:['Successeur','Tester une Mission 2 sur une tâche nouvelle sous des contraintes égales ou plus strictes.'],
    capital:['Capital','Réinvestir uniquement la valeur vérifiée dans la distribution, la capacité et une mission plus exigeante.'],
    compound:['Cumul','Le résultat accepté renforce le parent, la prochaine mission et la prochaine Maison.']
  }:{
    objective:['Objective','Define what must be preserved, built or become—and why it matters now.'],
    authority:['Authority','Freeze the principal, permitted actions, limits, stop conditions and acceptance authority.'],
    execution:['Execution','Act only inside the constituted mission, approved systems and authorized budget.'],
    evidence:['Evidence','Preserve provenance, tests, failures, interventions, costs and rights.'],
    validation:['Validation','Invite independent challenge, replay, dissent and a bounded verdict.'],
    acceptance:['Acceptance','The responsible institution records ACCEPT, REPAIR or REJECT.'],
    chronicle:['Chronicle','Admit only scoped, versioned, revocable and rights-cleared capability.'],
    successor:['Successor','Test a fresh Mission 2 under equal or stricter constraints.'],
    capital:['Capital','Reinvest only verified value into distribution, capability and a harder mission.'],
    compound:['Compound','The accepted result strengthens the parent, the next mission and the next Maison.']
  };
  const loopDetail=$('[data-v70-loop-detail]');
  $$('[data-v70-loop]').forEach(btn=>btn.addEventListener('click',()=>{
    const v=loop[btn.dataset.v70Loop]; if(!v||!loopDetail)return;
    $$('[data-v70-loop]').forEach(x=>x.classList.toggle('active',x===btn));
    loopDetail.querySelector('h3').textContent=v[0]; loopDetail.querySelector('p').textContent=v[1];
  }));

  const lightbox=$('.v70-lightbox'), lightImg=lightbox?.querySelector('img');
  $$('[data-lightbox]').forEach(btn=>btn.addEventListener('click',()=>{
    if(!lightbox||!lightImg)return; lightImg.src=btn.dataset.lightbox; lightImg.alt=btn.dataset.alt||'';
    lightbox.classList.add('open'); lightbox.querySelector('button')?.focus(); document.body.style.overflow='hidden';
  }));
  const closeLight=()=>{if(!lightbox)return;lightbox.classList.remove('open');lightImg.removeAttribute('src');document.body.style.overflow=''};
  lightbox?.querySelector('button')?.addEventListener('click',closeLight);
  lightbox?.addEventListener('click',e=>{if(e.target===lightbox)closeLight()});
  addEventListener('keydown',e=>{if(e.key==='Escape')closeLight()});

  $$('[data-year]').forEach(x=>x.textContent=new Date().getFullYear());

  // Minimal, privacy-preserving hero parallax. Disabled for reduced motion.
  if(!matchMedia('(prefers-reduced-motion: reduce)').matches){
    const bg=$('.v70-hero-bg');
    addEventListener('pointermove',e=>{
      if(!bg||innerWidth<900)return;
      const x=(e.clientX/innerWidth-.5)*8, y=(e.clientY/innerHeight-.5)*5;
      bg.style.transform=`scale(1.025) translate(${x}px,${y}px)`;
    },{passive:true});
  }
})();
