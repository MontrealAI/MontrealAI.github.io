(()=>{
  const fr=document.documentElement.lang.toLowerCase().startsWith('fr');
  const proof=fr?{
    objective:['Objectif constitué','Définir le résultat, le propriétaire de mission, l’autorité et les critères d’acceptation avant le travail.','Aucun travail conséquent ne commence sans objectif et autorité explicites.'],
    contract:['Mission Contract','Figer le périmètre, les entrées, les sorties, les contraintes, les interdictions, le budget et les conditions d’arrêt.','Le contrat de mission transforme une intention en travail gouvernable.'],
    debt:['Proof Debt','Identifier chaque assertion matérielle non encore soutenue par une preuve suffisante.','Ce qui n’est pas prouvé devient une dette visible, jamais une certitude silencieuse.'],
    jobs:['AGI Jobs','Décomposer la dette de preuve en travail borné avec propriétaires, outils, dépendances et tests explicites.','Chaque tâche doit produire un résultat inspectable ou un échec utile.'],
    bundles:['ProofBundles','Assembler artefacts, traces, sources, tests, contre-tests, échecs, réparations et provenance.','La preuve voyage avec le résultat.'],
    docket:['Evidence Docket','Constituer le dossier complet qui permet de rejouer, contester et comprendre le résultat.','Le dénominateur complet demeure visible.'],
    review:['Révision indépendante','Faire rejouer et attaquer les assertions matérielles hors du contexte qui les a produites.','Un millier d’agents peuvent encore se tromper ensemble.'],
    acceptance:['Acceptation responsable','L’autorité compétente décide ACCEPTER, RÉPARER, CONDITIONNER, REJETER, PIVOTER ou ARRÊTER.','Exécuter, accepter et admettre sont des verdicts distincts.'],
    chronicle:['Chronicle','N’admettre que la capacité acceptée, attribuable, libérée de droits, versionnée, révocable et encore actuelle.','La mémoire institutionnelle est un privilège gouverné.'],
    skill:['Validated Skill Graph','Relier missions, preuves, politiques, évaluateurs, droits, limites et capacité réutilisable.','La capacité ne compose que dans son périmètre autorisé.'],
    reuse:['Réutilisation protégée','Invoquer la capacité sans exposer l’intelligence privée qui l’a créée.','Preuve publique. Capacité privée. Autorité gouvernée.'],
    compound:['Composition','Tester sur une mission nouvelle et réinvestir uniquement la valeur que la réalité a acceptée.','La génération suivante doit mériter le droit d’être dite meilleure.']
  }:{
    objective:['Constituted objective','Define the outcome, mission owner, authority and acceptance criteria before work begins.','No consequential work begins without an explicit objective and authority.'],
    contract:['Mission Contract','Freeze scope, inputs, outputs, constraints, prohibitions, budget and stop conditions.','The Mission Contract turns intent into governable work.'],
    debt:['Proof Debt','Identify every material claim not yet supported by sufficient evidence.','What is not proved becomes visible debt—never silent certainty.'],
    jobs:['AGI Jobs','Decompose Proof Debt into bounded work with explicit owners, tools, dependencies and tests.','Every task must produce inspectable evidence or a useful failure.'],
    bundles:['ProofBundles','Assemble artifacts, traces, sources, tests, counter-tests, failures, repairs and provenance.','Proof travels with the result.'],
    docket:['Evidence Docket','Constitute the complete record required to replay, challenge and understand the result.','The full denominator remains visible.'],
    review:['Independent review','Replay and attack material claims outside the context that produced them.','A thousand agents can still be wrong together.'],
    acceptance:['Accountable acceptance','The responsible authority decides ACCEPT, REPAIR, CONDITION, REJECT, PIVOT or STOP.','Execute, Accept and Admit are different verdicts.'],
    chronicle:['Chronicle','Admit only accepted, attributable, rights-cleared, versioned, revocable and current capability.','Institutional memory is a governed privilege.'],
    skill:['Validated Skill Graph','Connect missions, evidence, policy, evaluators, rights, limits and reusable capability.','Capability compounds only within its authorized scope.'],
    reuse:['Protected reuse','Invoke capability without exposing the private intelligence that created it.','Public proof. Private capability. Governed authority.'],
    compound:['Compounding','Test on a fresh mission and reinvest only value that reality accepted.','The next generation must earn the right to be called better.']
  };
  const buttons=[...document.querySelectorAll('[data-apex-proof]')],title=document.querySelector('[data-apex-proof-title]'),body=document.querySelector('[data-apex-proof-body]'),law=document.querySelector('[data-apex-proof-law]'),orb=document.querySelector('[data-apex-proof-orb]');
  const setProof=(key,focus=false)=>{const data=proof[key];if(!data)return;buttons.forEach((b,i)=>{const active=b.dataset.apexProof===key;b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;if(active&&focus)b.focus()});if(title)title.textContent=data[0];if(body)body.textContent=data[1];if(law)law.textContent=data[2];if(orb)orb.textContent=String(buttons.findIndex(b=>b.dataset.apexProof===key)+1).padStart(2,'0')};
  buttons.forEach((b,i)=>{b.setAttribute('role','tab');b.addEventListener('click',()=>setProof(b.dataset.apexProof));b.addEventListener('keydown',e=>{if(!['ArrowRight','ArrowLeft','Home','End'].includes(e.key))return;e.preventDefault();let n=i;if(e.key==='ArrowRight')n=(i+1)%buttons.length;if(e.key==='ArrowLeft')n=(i-1+buttons.length)%buttons.length;if(e.key==='Home')n=0;if(e.key==='End')n=buttons.length-1;setProof(buttons[n].dataset.apexProof,true)})});if(buttons.length)setProof(buttons[0].dataset.apexProof);
  const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
  if(!reduced){const io=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){if(e.target.animate)e.target.animate([{opacity:.05,transform:'translateY(22px)'},{opacity:1,transform:'translateY(0)'}],{duration:680,easing:'cubic-bezier(.2,.75,.2,1)',fill:'none'});io.unobserve(e.target)}}),{threshold:.12});document.querySelectorAll('.apex-reveal').forEach(el=>io.observe(el));const scene=document.querySelector('[data-apex-parallax]');if(scene&&matchMedia('(pointer:fine)').matches){scene.addEventListener('pointermove',e=>{const r=scene.getBoundingClientRect(),x=(e.clientX-r.left)/r.width-.5,y=(e.clientY-r.top)/r.height-.5;scene.style.transform=`rotateX(${(-y*2.3).toFixed(2)}deg) rotateY(${(x*2.3).toFixed(2)}deg)`});scene.addEventListener('pointerleave',()=>scene.style.transform='') }}
})();
