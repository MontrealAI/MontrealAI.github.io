(() => {
  const isFr=document.documentElement.lang.toLowerCase().startsWith('fr');
  const data=isFr?{
    mission:{n:'01',title:'Mission',body:'MONTREAL.AI transforme un problème important en objectif borné, mesurable et économiquement conséquent.',law:'La capacité ne choisit pas sa propre destination. La mission précède l’exécution.'},
    authority:{n:'02',title:'Autorité',body:'GoalOS fixe les permissions, les interdictions, les seuils d’escalade, le budget et le retour arrière.',law:'Aucune autonomie sans une enveloppe d’autorité explicite.'},
    orchestration:{n:'03',title:'Orchestration',body:'Modèles, agents, chercheurs, startups et partenaires sont composés selon la mission et demeurent remplaçables.',law:'Distribuer l’exécution sans distribuer le contrôle constitutionnel.'},
    proof:{n:'04',title:'Preuve',body:'Proof Gradient relie assertions, sources, tests, limites, validation indépendante et décision client.',law:'Une sortie n’est pas une preuve; une preuve n’est pas encore une acceptation.'},
    distribution:{n:'05',title:'Distribution',body:'MONTREAL.AI conserve la voie de confiance par laquelle les organisations découvrent, achètent et étendent la capacité.',law:'La distribution directe transforme l’intelligence en relation institutionnelle durable.'},
    memory:{n:'06',title:'Mémoire',body:'Les données confidentielles restent protégées; l’apprentissage licite et généralisé renforce l’Evidence Graph.',law:'Chaque mission doit rendre la suivante plus forte sans prendre ce qui ne nous appartient pas.'},
    ownership:{n:'07',title:'Propriété',body:'Le parent conserve la marque, l’architecture, la mémoire, l’allocation du capital et le droit de constituer les Maisons.',law:'La valeur composée demeure dans l’institution contrôlée par le fondateur.'}
  }:{
    mission:{n:'01',title:'Mission',body:'MONTREAL.AI turns an important problem into a bounded, measurable and economically consequential objective.',law:'Capability does not choose its own destination. Mission precedes execution.'},
    authority:{n:'02',title:'Authority',body:'GoalOS fixes permissions, prohibited actions, escalation thresholds, budget and rollback.',law:'No autonomy without an explicit Authority Envelope.'},
    orchestration:{n:'03',title:'Orchestration',body:'Models, agents, researchers, startups and partners are composed around the mission and remain replaceable.',law:'Distribute execution without distributing constitutional control.'},
    proof:{n:'04',title:'Proof',body:'Proof Gradient connects claims, sources, tests, limitations, independent validation and the customer decision.',law:'Output is not proof; proof is not yet acceptance.'},
    distribution:{n:'05',title:'Distribution',body:'MONTREAL.AI retains the trusted route through which organizations discover, purchase and expand capability.',law:'Direct distribution turns intelligence into a durable institutional relationship.'},
    memory:{n:'06',title:'Memory',body:'Confidential customer material remains protected while lawful generalized learning strengthens the Evidence Graph.',law:'Every mission should make the next stronger without taking what is not ours.'},
    ownership:{n:'07',title:'Ownership',body:'The parent retains the brand, architecture, memory, capital allocation and right to constitute the Maisons.',law:'Compounded value remains inside the founder-controlled institution.'}
  };
  const tabs=[...document.querySelectorAll('[data-gravity-tab]')];
  const panel=document.querySelector('#gravity-panel');
  const update=key=>{const d=data[key];if(!d)return;tabs.forEach(t=>{const active=t.dataset.gravityTab===key;t.setAttribute('aria-selected',String(active));t.tabIndex=active?0:-1;if(active&&panel)panel.setAttribute('aria-labelledby',t.id)});document.querySelector('[data-gravity-number]')?.replaceChildren(document.createTextNode(d.n));document.querySelector('[data-gravity-title]')?.replaceChildren(document.createTextNode(d.title));document.querySelector('[data-gravity-body]')?.replaceChildren(document.createTextNode(d.body));document.querySelector('[data-gravity-law]')?.replaceChildren(document.createTextNode(d.law));};
  tabs.forEach((t,i)=>{t.addEventListener('click',()=>update(t.dataset.gravityTab));t.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(e.key))return;e.preventDefault();let n=i;if(['ArrowRight','ArrowDown'].includes(e.key))n=(i+1)%tabs.length;if(['ArrowLeft','ArrowUp'].includes(e.key))n=(i-1+tabs.length)%tabs.length;if(e.key==='Home')n=0;if(e.key==='End')n=tabs.length-1;tabs[n].focus();update(tabs[n].dataset.gravityTab)})});
  if(tabs.length)update(tabs[0].dataset.gravityTab);
})();