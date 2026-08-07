(() => {
  'use strict';
  const isFr=(document.documentElement.lang||'').toLowerCase().startsWith('fr');
  const goalosCopy = isFr ? {
    objective:['CONSTITUER CE QUI COMPTE','Objectif','Figer l’objectif conséquent, le responsable, le test d’acceptation et les conditions d’arrêt avant le début du travail.'],
    authority:['BORNER LES CONSÉQUENCES','Autorité','Nommer chaque acteur, outil, catégorie de données, budget, interdiction, route d’escalade et condition de retour arrière.'],
    execution:['AGIR UNIQUEMENT DANS LA MISSION','Travail borné','Exécuter au moyen d’un graphe d’action explicite tout en préservant l’autorité, la provenance, les coûts et les interventions.'],
    evidence:['RENDRE LE RÉSULTAT INSPECTABLE','Preuve','Assembler les affirmations, la provenance, les tests, les échecs, les interventions, les coûts, les droits et l’incertitude non résolue.'],
    acceptance:['SÉPARER LE VERDICT DE LA RESPONSABILITÉ','Acceptation','Les réviseurs indépendants peuvent contester la preuve; seule l’institution responsable consigne ACCEPTER, RÉPARER ou REJETER.'],
    chronicle:['COMPOSER UNIQUEMENT CE QUI RÉUSSIT','Chronicle','Admettre séparément une capacité délimitée et aux droits réglés, puis prouver son transfert sur une mission successeure nouvelle.']
  } : {
    objective:['CONSTITUTE WHAT MATTERS','Objective','Freeze the consequential objective, accountable principal, acceptance test and stop conditions before work begins.'],
    authority:['BOUND CONSEQUENCE','Authority','Name every actor, tool, data class, budget, prohibition, escalation route and rollback condition.'],
    execution:['ACT ONLY INSIDE THE MISSION','Bounded work','Execute through an explicit action graph while preserving authority, provenance, cost and intervention records.'],
    evidence:['MAKE THE RESULT INSPECTABLE','Evidence','Assemble claims, provenance, tests, failures, interventions, costs, rights and unresolved uncertainty.'],
    acceptance:['SEPARATE VERDICT FROM RESPONSIBILITY','Acceptance','Independent reviewers may challenge the evidence; only the accountable institution records ACCEPT, REPAIR or REJECT.'],
    chronicle:['COMPOUND ONLY WHAT PASSES','Chronicle','Admit scoped, rights-cleared capability separately and prove transfer on a fresh successor mission.']
  };
  const stage = document.querySelector('[data-apex-stage]');
  if(stage){
    const nodes=[...stage.querySelectorAll('[data-apex-node]')];
    const kicker=stage.querySelector('[data-stage-kicker]');
    const title=stage.querySelector('[data-stage-title]');
    const copy=stage.querySelector('[data-stage-copy]');
    const activate=(node)=>{
      nodes.forEach(n=>n.classList.toggle('active',n===node));
      const d=goalosCopy[node.dataset.apexNode]||[];
      if(kicker)kicker.textContent=d[0]||'';
      if(title)title.textContent=d[1]||'';
      if(copy)copy.textContent=d[2]||'';
    };
    nodes.forEach(n=>n.addEventListener('click',()=>activate(n)));
  }
  const compassCopy = isFr ? {
    detect:'Observer les capacités, coûts, contraintes, lois, goulots et discontinuités avant que le plan courant devienne obsolète.',
    branch:'Générer des architectures concurrentes et des futurs plausibles plutôt que d’accepter une seule réponse persuasive.',
    bound:'Spécifier l’autorité, la preuve, le risque, la réversibilité et les conditions d’arrêt de chaque trajectoire plausible.',
    select:'Faire avancer uniquement l’architecture qui survit le mieux à la gouvernance, à la preuve et au standard de décision responsable.'
  } : {
    detect:'Observe capabilities, costs, constraints, laws, bottlenecks and discontinuities before the current plan becomes obsolete.',
    branch:'Generate competing architectures and plausible futures rather than accepting one persuasive answer.',
    bound:'Specify authority, evidence, risk, reversibility and stop conditions for every plausible path.',
    select:'Advance only the architecture that best survives governance, evidence and the accountable decision standard.'
  };
  const compass=document.querySelector('[data-compass-stage]');
  if(compass){
    const detail=compass.querySelector('[data-compass-detail]');
    compass.querySelectorAll('[data-compass]').forEach(c=>{
      c.setAttribute('tabindex','0');
      const show=()=>{if(detail)detail.textContent=compassCopy[c.dataset.compass]||''};
      c.addEventListener('mouseenter',show);c.addEventListener('focus',show);c.addEventListener('click',show);
    });
  }
  const expandable=[...document.querySelectorAll('.media-frame img,.architecture-plate img,.stack-visual img')];
  if(expandable.length){
    let viewer=document.querySelector('.apex-media-viewer');
    if(!viewer){
      viewer=document.createElement('div');viewer.className='apex-media-viewer';viewer.setAttribute('role','dialog');viewer.setAttribute('aria-modal','true');viewer.setAttribute('aria-label',isFr?'Visuel agrandi':'Expanded visual');
      viewer.innerHTML='<button type="button" aria-label="'+(isFr?'Fermer le visuel agrandi':'Close expanded visual')+'">×</button><img alt=""><div class="viewer-caption"></div>';
      document.body.appendChild(viewer);
    }
    const vimg=viewer.querySelector('img'),cap=viewer.querySelector('.viewer-caption'),close=()=>{viewer.classList.remove('open');document.body.style.overflow='';};
    viewer.querySelector('button').addEventListener('click',close);viewer.addEventListener('click',e=>{if(e.target===viewer)close()});document.addEventListener('keydown',e=>{if(e.key==='Escape')close()});
    expandable.forEach(img=>{
      const frame=img.closest('.media-frame,.architecture-plate,.stack-visual');if(frame)frame.dataset.expandable='';
      img.setAttribute('tabindex','0');img.setAttribute('role','button');img.setAttribute('aria-label',(img.alt|| (isFr?'Visuel':'Visual'))+(isFr?' — ouvrir en plein écran':' — open full screen'));
      const open=()=>{vimg.src=img.currentSrc||img.src;vimg.alt=img.alt||'';cap.textContent=img.alt||'';viewer.classList.add('open');document.body.style.overflow='hidden';viewer.querySelector('button').focus();};
      img.addEventListener('click',open);img.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open();}});
    });
  }
})();
