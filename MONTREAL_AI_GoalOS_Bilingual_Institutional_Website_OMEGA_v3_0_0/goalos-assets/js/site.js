(()=>{
  'use strict';
  const q=(s,r=document)=>r.querySelector(s), qa=(s,r=document)=>[...r.querySelectorAll(s)];
  const money=(n,lang='en')=>new Intl.NumberFormat(lang==='fr'?'fr-CA':'en-CA',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(Number(n)||0);

  const menu=q('.menu-btn'), nav=q('.nav');
  if(menu&&nav){
    menu.addEventListener('click',()=>{const open=nav.classList.toggle('open');menu.setAttribute('aria-expanded',String(open));});
    qa('a',nav).forEach(a=>a.addEventListener('click',()=>{nav.classList.remove('open');menu.setAttribute('aria-expanded','false');}));
    document.addEventListener('keydown',e=>{if(e.key==='Escape'){nav.classList.remove('open');menu.setAttribute('aria-expanded','false');}});
  }
  qa('[data-year]').forEach(y=>y.textContent=new Date().getFullYear());

  async function copyText(text,button){
    try{
      if(navigator.clipboard&&window.isSecureContext) await navigator.clipboard.writeText(text);
      else{const ta=document.createElement('textarea');ta.value=text;ta.style.position='fixed';ta.style.opacity='0';document.body.appendChild(ta);ta.select();document.execCommand('copy');ta.remove();}
      if(button){const old=button.textContent;button.textContent=button.dataset.copied||((document.documentElement.lang||'').startsWith('fr')?'Copié':'Copied');setTimeout(()=>button.textContent=old,1400);}
    }catch(_e){ if(button) button.textContent=(document.documentElement.lang||'').startsWith('fr')?'Copie impossible':'Copy unavailable'; }
  }
  qa('[data-copy]').forEach(b=>b.addEventListener('click',()=>copyText(b.dataset.copy||'',b)));

  qa('[data-roi-calculator]').forEach(calc=>{
    const fee=q('[data-fee]',calc), value=q('[data-value]',calc), avoided=q('[data-avoided]',calc);
    const multiple=q('[data-multiple]',calc), roi=q('[data-roi]',calc), benefit=q('[data-benefit]',calc);
    const lang=(document.documentElement.lang||'en').startsWith('fr')?'fr':'en';
    const update=()=>{
      const f=Math.max(0,Number(fee?.value)||0), v=Math.max(0,Number(value?.value)||0), a=Math.max(0,Number(avoided?.value)||0);
      const total=v+a, net=total-f;
      multiple.textContent=f?`${(total/f).toFixed(1)}×`:'—';
      roi.textContent=f?`${((net/f)*100).toFixed(0)}%`:'—';
      benefit.textContent=money(net,lang);
    };
    [fee,value,avoided].forEach(x=>x&&x.addEventListener('input',update)); update();
  });

  const missionDefinitions={
    fr:{
      deployment:{name:'GoalOS AI Deployment Proof Mission',why:'Décider si un système IA doit être déployé, acheté, élargi, réparé, restreint ou arrêté.',base:[25000,75000],first:'Inventaire du système, matrice de revendications et plan d’évaluation gelé',proof:'Dossier de preuves, tests de qualité, sécurité, économie et réversibilité',office:'Bureau de preuve de déploiement IA'},
      compliance:{name:'GoalOS AI Evidence Readiness Mission',why:'Établir un état de preuve continu reliant obligations, contrôles, systèmes, exceptions et remédiations.',base:[30000,100000],first:'Inventaire des systèmes et cartographie obligations → contrôles → preuves',proof:'Preuves actuelles, traçables, attribuables et révisables par les professionnels requis',office:'Bureau de conformité et de preuves d’audit'},
      rfp:{name:'GoalOS RFP Proof Mission',why:'Répondre plus vite à une occasion importante sans propager de revendications commerciales non supportées.',base:[10000,30000],first:'Décision bid/no-bid, graphe des exigences et base de revendications approuvées',proof:'Chaque réponse matérielle liée à une preuve, une politique, un propriétaire et une date',office:'Bureau RFP-to-Revenue'},
      diligence:{name:'GoalOS Strategic Diligence Proof Room',why:'Transformer une thèse d’investissement, d’acquisition, de partenariat ou d’achat en décision inspectable.',base:[50000,150000],first:'Thèse, matrice de revendications, accès à la salle de données après contrat',proof:'Contradictions, scénarios, limites, révision spécialiste et conditions de poursuite',office:'Bureau de preuve de portefeuille'},
      venture:{name:'GoalOS Venture Proof Mission',why:'Déterminer si une nouvelle entreprise mérite d’être construite, réparée, pivotée, élargie ou arrêtée.',base:[20000,75000],first:'Registre d’hypothèses, segments, offre, prix et plan d’expériences consenties',proof:'Réponse réelle du marché, coûts complets, échecs, décisions et critères d’arrêt',office:'Bureau autonome d’exploitation de venture'}
    },
    en:{
      deployment:{name:'GoalOS AI Deployment Proof Mission',why:'Decide whether an AI system should be deployed, bought, scaled, repaired, restricted, or stopped.',base:[25000,75000],first:'System inventory, claim matrix, and frozen evaluation plan',proof:'Evidence Docket with quality, security, economic, and reversibility tests',office:'AI Deployment Proof Office'},
      compliance:{name:'GoalOS AI Evidence Readiness Mission',why:'Create a continuous evidence state linking obligations, controls, systems, exceptions, and remediation.',base:[30000,100000],first:'System inventory and obligation → control → evidence map',proof:'Current, attributable, traceable evidence routed to the required qualified reviewers',office:'AI Compliance & Audit Evidence Office'},
      rfp:{name:'GoalOS RFP Proof Mission',why:'Respond to an important opportunity faster without propagating unsupported commercial claims.',base:[10000,30000],first:'Bid/no-bid decision, requirement graph, and approved claim base',proof:'Every material answer linked to evidence, policy, owner, and date',office:'RFP-to-Revenue Proof Office'},
      diligence:{name:'GoalOS Strategic Diligence Proof Room',why:'Turn an investment, acquisition, partnership, or procurement thesis into an inspectable decision.',base:[50000,150000],first:'Thesis, claim matrix, and contracted data-room access',proof:'Contradictions, scenarios, limits, specialist review, and continuation conditions',office:'Portfolio Proof Office'},
      venture:{name:'GoalOS Venture Proof Mission',why:'Determine whether a new venture deserves to be built, repaired, pivoted, scaled, or stopped.',base:[20000,75000],first:'Assumption ledger, segments, offer, price, and consented experiment plan',proof:'Real market response, full costs, failures, decisions, and stop conditions',office:'Autonomous Venture Operating Office'}
    }
  };

  qa('[data-mission-lab]').forEach(lab=>{
    const lang=lab.dataset.lang==='fr'?'fr':'en', defs=missionDefinitions[lang];
    const decision=q('[data-decision]',lab), stake=q('[data-stake]',lab), urgency=q('[data-urgency]',lab), proof=q('[data-proof]',lab), recurring=q('[data-recurring]',lab), ack=q('[data-ack]',lab);
    const name=q('[data-result-name]',lab), why=q('[data-result-why]',lab), price=q('[data-result-price]',lab), first=q('[data-result-first]',lab), proofOut=q('[data-result-proof]',lab), expand=q('[data-result-expand]',lab), build=q('[data-build-brief]',lab), safe=q('.safe-brief',lab), pre=q('[data-brief]',lab), copy=q('[data-copy-brief]',lab), email=q('[data-email-brief]',lab);
    const label=(select)=>select?.options[select.selectedIndex]?.textContent?.trim()||'';
    const recommendation=()=>{
      const d=defs[decision.value]; let low=d.base[0], high=d.base[1];
      const stakeValue=Number(stake.value)||250000;
      const proofFactor={supportive:1,robust:1.35,assurance:1.8}[proof.value]||1;
      const urgencyFactor={standard:1,fast:1.15,critical:1.35}[urgency.value]||1;
      const scaleFactor=stakeValue>=25000000?1.6:stakeValue>=5000000?1.35:stakeValue>=1000000?1.15:1;
      low=Math.round(low*proofFactor*urgencyFactor*scaleFactor/5000)*5000;
      high=Math.round(high*proofFactor*urgencyFactor*scaleFactor/5000)*5000;
      const recurringText=recurring.value==='yes'?d.office:(lang==='fr'?'Mission suivante seulement si la preuve le justifie':'A next mission only if the evidence justifies it');
      name.textContent=d.name; why.textContent=d.why; price.textContent=`${money(low,lang)}–${money(high,lang)}`; first.textContent=d.first; proofOut.textContent=d.proof; expand.textContent=recurringText;
      build.disabled=!ack.checked;
      return {d,low,high,recurringText};
    };
    [decision,stake,urgency,proof,recurring,ack].forEach(x=>x&&x.addEventListener('change',()=>{recommendation();safe.hidden=true;safe.classList.remove('active');}));
    const compose=()=>{
      const r=recommendation();
      if(lang==='fr') return `Objet : Qualification non confidentielle d’une mission GoalOS\n\nBonjour MONTREAL.AI,\n\nJe souhaite qualifier une mission potentielle sans transmettre de renseignement confidentiel.\n\nCatégorie de décision : ${label(decision)}\nValeur approximative en jeu : ${label(stake)}\nÉchéance : ${label(urgency)}\nCharge de preuve recherchée : ${label(proof)}\nBesoin récurrent après la décision : ${label(recurring)}\nMission recommandée par le laboratoire local : ${r.d.name}\nFourchette indicative : ${money(r.low,'fr')}–${money(r.high,'fr')}\n\nÀ ce stade, je ne joins aucun secret commercial, renseignement personnel, donnée réglementée, code, identifiant, vulnérabilité, document privilégié ou information de tiers. Je comprends qu’une entente écrite et un canal approuvé doivent précéder tout échange sensible.\n\nJe souhaite discuter de l’ajustement, du commanditaire, du budget, du délai et de la prochaine étape.\n\nCordialement,\n[Nom]\n[Organisation]\n[Rôle]`;
      return `Subject: Non-confidential qualification of a GoalOS mission\n\nHello MONTREAL.AI,\n\nI would like to qualify a potential mission without transmitting confidential information.\n\nDecision category: ${label(decision)}\nApproximate value at stake: ${label(stake)}\nDeadline: ${label(urgency)}\nRequired proof burden: ${label(proof)}\nRecurring need after the decision: ${label(recurring)}\nMission recommended by the local lab: ${r.d.name}\nIndicative range: ${money(r.low,'en')}–${money(r.high,'en')}\n\nAt this stage I am not attaching trade secrets, personal information, regulated data, source code, credentials, vulnerabilities, privileged material, or third-party information. I understand that a written agreement and approved channel must precede any sensitive exchange.\n\nI would like to discuss fit, sponsor, budget, timing, and the appropriate next step.\n\nRegards,\n[Name]\n[Organization]\n[Role]`;
    };
    build?.addEventListener('click',()=>{if(!ack.checked)return;const text=compose();pre.textContent=text;safe.hidden=false;safe.classList.add('active');const subject=lang==='fr'?'Qualification non confidentielle d’une mission GoalOS':'Non-confidential GoalOS mission qualification';email.href=`mailto:secretariat@montreal.ai?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(text)}`;safe.scrollIntoView({behavior:'smooth',block:'nearest'});});
    copy?.addEventListener('click',()=>copyText(pre.textContent||'',copy));
    recommendation();
  });
})();
