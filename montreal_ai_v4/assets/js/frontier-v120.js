
(() => {
  'use strict';
  const fr=(document.documentElement.lang||'').toLowerCase().startsWith('fr');
  const copy=fr?{
    attract:['ARCHITECTURE DE RÉFÉRENCE','GoalOS Attracts Ω','Découvre les objectifs conséquents, établit le consentement, qualifie l’autorité et ne constitue une mission que lorsque la valeur de la preuve est réelle.','Aucune permission, aucun engagement.'],
    chat:['PROGRAMME DE RECHERCHE','GoalOS Chat Ω','Une porte d’entrée IA vérifiée : répondre avec preuve lorsque la réponse suffit; lancer des AGI Jobs bornés lorsque l’action est requise.','La conversation ne devient action qu’après constitution.'],
    graph:['PROGRAMME DE RECHERCHE','Graph Engineering Ω','Remplace la chaîne opaque par un graphe typé de travail, preuve, autorité, mémoire et capital. Les nœuds agissent; les arêtes portent une signification déclarée.','Des arêtes réelles seulement. La preuve gouverne le graphe.'],
    machine:['INSTITUTION DE RÉFÉRENCE','Machine commerciale autonome Ω','Attire, qualifie, vend, livre, prouve, renouvelle et réinvestit sous des offres fermées, des limites de risque et une acceptation client séparée.','Aucun profit vérifié, aucun réinvestissement automatique.'],
    skill:['RECHERCHE PUBLIQUE · CAPACITÉ PRIVÉE','Graphe des compétences validées','Préserve les missions, preuves, méthodes, droits, échecs, révocations et capacités admises; seule une capacité validée peut influencer la suite.','Garder l’intelligence privée. Rendre la preuve publique.'],
    value:['SCÉNARIOS DE GESTION ILLUSTRATIFS','Dossier de réalisation de valeur Ω','Rend la thèse de valeur explicite, éditable et falsifiable pour les décisions de déploiement, d’allocation de capital et de preuve commerciale.','Les chiffres modélisés ne sont ni des résultats réalisés ni des garanties.'],
    masterclass:['PROGRAMME DE FORMATION','MasterClasses GoalOS','La MasterClass Frontière explique l’institution gouvernée par la preuve; la MasterClass Exécutive traduit les sorties IA en décisions inspectables.','La formation n’accorde ni autorité ni certification opérationnelle.'],
    reserve:['PUBLICATION ET SURFACE PUBLIQUE','Réserve souveraine Ω · P4.0','Autorité révocable, mémoire gouvernée, règlement et amélioration sous preuve pour des agents de plus en plus capables.','L’intelligence propose. La preuve décide. L’institution autorise.']
  }:{
    attract:['REFERENCE ARCHITECTURE','GoalOS Attracts Ω','Discovers consequential objectives, establishes consent, qualifies authority and constitutes a mission only when the value of proof is real.','No permission, no engagement.'],
    chat:['RESEARCH PROGRAMME','GoalOS Chat Ω','A verified AI front door: answer with evidence when an answer is enough; launch bounded AGI Jobs when action is required.','Conversation becomes action only after constitution.'],
    graph:['RESEARCH PROGRAMME','Graph Engineering Ω','Replaces the opaque chain with a typed graph of work, proof, authority, memory and capital. Nodes act; edges carry declared meaning.','Real edges only. Proof governs the graph.'],
    machine:['REFERENCE INSTITUTION','Autonomous Commercial Machine Ω','Attracts, qualifies, sells, delivers, proves, renews and reinvests under closed offers, risk limits and separate customer acceptance.','No verified profit, no automatic reinvestment.'],
    skill:['PUBLIC RESEARCH · PRIVATE CAPABILITY','Validated Skill Graph','Preserves missions, evidence, methods, rights, failures, revocations and admitted capabilities; only validated capability may influence what comes next.','Keep intelligence private. Make proof public.'],
    value:['ILLUSTRATIVE MANAGEMENT SCENARIOS','Value Realization Dossier Ω','Makes the value thesis explicit, editable and falsifiable for deployment, capital-allocation and commercial-proof decisions.','Modeled figures are not realized results or guarantees.'],
    masterclass:['FORMATION PROGRAMME','GoalOS MasterClasses','The Frontier MasterClass explains the proof-gated institution; the Executive MasterClass turns AI output into inspectable decisions.','Education grants no operational authority or certification.'],
    reserve:['PUBLICATION & PUBLIC SURFACE','Sovereign Reserve Ω · P4.0','Revocable authority, governed memory, settlement and proof-gated improvement for increasingly capable agents.','Intelligence proposes. Evidence decides. Institutions authorize.']
  };
  const engine=document.querySelector('[data-frontier-engine]');
  if(engine){
    const buttons=[...engine.querySelectorAll('[data-frontier-step]')];
    const status=engine.querySelector('[data-frontier-status]'), title=engine.querySelector('[data-frontier-title]'), text=engine.querySelector('[data-frontier-text]'), law=engine.querySelector('[data-frontier-law]');
    const activate=(button)=>{buttons.forEach(b=>{const on=b===button;b.classList.toggle('active',on);b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1;});const d=copy[button.dataset.frontierStep];if(!d)return;status.textContent=d[0];title.textContent=d[1];text.textContent=d[2];law.textContent=d[3];};
    buttons.forEach((b,i)=>{b.setAttribute('role','tab');b.setAttribute('aria-selected',i===0?'true':'false');b.tabIndex=i===0?0:-1;b.addEventListener('click',()=>activate(b));b.addEventListener('keydown',e=>{let n=i;if(e.key==='ArrowRight'||e.key==='ArrowDown')n=(i+1)%buttons.length;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')n=(i-1+buttons.length)%buttons.length;else if(e.key==='Home')n=0;else if(e.key==='End')n=buttons.length-1;else return;e.preventDefault();buttons[n].focus();activate(buttons[n]);});});
  }
})();
