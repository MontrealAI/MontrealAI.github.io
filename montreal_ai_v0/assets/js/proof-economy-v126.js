(() => {
 const data={
  output:{k:'CANDIDATE WORK',t:'Output is not authority.',p:'Models and agents may draft, reason, propose and execute. Their result remains a candidate until evidence, challenge and responsible acceptance establish what may matter.',e:'Candidate output → Proof Debt'},
  debt:{k:'UNSUPPORTED CLAIMS',t:'Proof Debt becomes work.',p:'Unsupported material claims are converted into explicit proof obligations. The mission creates only the AGI Jobs needed to retire its own proof debt.',e:'Proof Debt → Custom AGI Jobs'},
  jobs:{k:'BOUNDED EXECUTION',t:'AGI Jobs produce evidence.',p:'Agents, people, tools and systems act inside a frozen Authority Envelope. Every job returns evidence, failure, cost, provenance and rights information.',e:'AGI Jobs → ProofBundles'},
  docket:{k:'INSPECTABLE RECORD',t:'Evidence earns justified belief.',p:'ProofBundles enter an Evidence Docket. Independent validators may replay, attack, condition, reject or require repair. Validation informs acceptance; it does not replace it.',e:'ProofBundles → Evidence Docket → Challenge'},
  chronicle:{k:'GOVERNED MEMORY',t:'Chronicle decides what may survive.',p:'Only accepted, attributable, rights-cleared, scoped, versioned and revocable capability may become institutional memory.',e:'Acceptance → Chronicle Gate'},
  skill:{k:'REUSABLE CAPABILITY',t:'Validated skill may be invoked—not surrendered.',p:'Private operational intelligence remains protected. Authorized agents receive a scoped, revocable right to invoke Chronicle-admitted capability.',e:'Chronicle → Validated Skill Graph'},
  root:{k:'PUBLIC COMMITMENT',t:'Proof can be public while intelligence remains private.',p:'Merkle roots, commitments, attestations and policy proofs can make lineage inspectable without placing private capability on-chain as plaintext.',e:'Validated Skill → On-chain Graph Root'},
  future:{k:'FRESH SUCCESSOR',t:'A new generation must earn the right to be called better.',p:'Only superior performance on fresh work under equal or tighter constraints supports a claim of improvement. A newer version is not proof.',e:'Root-verifiable capability → Harder Future Mission ↺'}
 };
 const buttons=[...document.querySelectorAll('[data-pe-node]')],k=document.querySelector('[data-pe-kicker]'),t=document.querySelector('[data-pe-title]'),p=document.querySelector('[data-pe-copy]'),e=document.querySelector('[data-pe-equation]');
 function select(btn){const v=data[btn.dataset.peNode];if(!v)return;buttons.forEach(b=>{const on=b===btn;b.setAttribute('aria-selected',on?'true':'false');b.tabIndex=on?0:-1});if(k)k.textContent=v.k;if(t)t.textContent=v.t;if(p)p.textContent=v.p;if(e)e.textContent=v.e}
 buttons.forEach((b,i)=>{b.addEventListener('click',()=>select(b));b.addEventListener('keydown',ev=>{let n=i;if(ev.key==='ArrowRight'||ev.key==='ArrowDown')n=(i+1)%buttons.length;else if(ev.key==='ArrowLeft'||ev.key==='ArrowUp')n=(i-1+buttons.length)%buttons.length;else if(ev.key==='Home')n=0;else if(ev.key==='End')n=buttons.length-1;else return;ev.preventDefault();buttons[n].focus();select(buttons[n])})});
 if(buttons[0])select(buttons[0]);
})();
