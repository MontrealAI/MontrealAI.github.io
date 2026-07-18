(function(){
  'use strict';
  const M=window.GoalOSMembership;
  const $=s=>document.querySelector(s);
  const state={account:'',provider:null,signer:null,runtime:null,parsed:null,ownership:null,tier:null};
  function esc(s){return String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
  function toast(msg){const t=$('#toast');if(!t)return;t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2600);}
  function setStatus(msg,kind=''){const el=$('#gateStatus');if(!el)return;el.className='status '+kind;el.innerHTML=msg;}
  function setStep(n,status,detail){const el=document.querySelector(`[data-step="${n}"]`);if(!el)return;el.classList.remove('active','done','error');if(status)el.classList.add(status);const st=el.querySelector('.state');if(st)st.textContent=detail||({active:'Active',done:'Done',error:'Action required'}[status]||'Pending');}
  function short(a){return a?`${a.slice(0,6)}…${a.slice(-4)}`:'—';}
  function deepLink(){const target=location.host+location.pathname+location.search+location.hash;return `https://metamask.app.link/dapp/${target}`;}
  async function detect(){
    try{state.runtime=await M.loadRuntime();}
    catch(err){setStep(1,'error','Configuration unavailable');setStatus(esc(err.message||err),'bad');$('#connectBtn').disabled=true;$('#verifyBtn').disabled=true;return;}
    const label=M.lastLabel();if(label)$('#clubLabel').value=label;
    $('#parentName').textContent=state.runtime.parentName;
    $('#termsVersion').textContent=state.runtime.termsVersion;
    $('#privacyVersion').textContent=state.runtime.privacyVersion;
    $('#mobileMetaMaskLink').href=deepLink();
    if(!window.ethereum){
      setStatus('MetaMask was not detected in this browser. Install MetaMask on desktop, or open this page inside the MetaMask mobile browser.','warn');
      $('#connectBtn').textContent='Install / open MetaMask';
      return;
    }
    try{
      const c=await M.connectMetaMask(false);state.provider=c.provider;state.account=c.address;
      setStep(1,'done',short(c.address));$('#walletValue').textContent=short(c.address);$('#verifyBtn').disabled=false;
      const primary=await c.provider.lookupAddress(c.address).catch(()=>null);
      if(primary){try{const p=M.parseDirectName(primary,state.runtime.parentName);$('#clubLabel').value=p.label;}catch{}}
      setStatus('MetaMask is already connected. Enter your direct AGI Club label and verify ownership.','ok');
    }catch{}
    const saved=M.readSession();
    if(saved){
      $('#resumeBtn').classList.remove('hidden');
      $('#resumeHint').textContent=`Resume ${saved.name}`;
    }
  }
  async function connect(){
    if(!window.ethereum){location.href='https://metamask.io/download/';return;}
    try{
      setStep(1,'active','Opening MetaMask');setStatus('Opening MetaMask…','warn');
      const c=await M.connectMetaMask(true);state.provider=c.provider;state.signer=c.signer;state.account=c.address;
      setStep(1,'done',short(c.address));$('#walletValue').textContent=short(c.address);$('#verifyBtn').disabled=false;
      const primary=await c.provider.lookupAddress(c.address).catch(()=>null);
      if(primary){try{const p=M.parseDirectName(primary,state.runtime.parentName);if(!$('#clubLabel').value)$('#clubLabel').value=p.label;}catch{}}
      setStep(2,'active','Enter your label');setStatus('MetaMask connected on Ethereum Mainnet. Enter your direct club.agi.eth label.','ok');
    }catch(err){setStep(1,'error','Not connected');setStatus(esc(err.message||err),'bad');}
  }
  async function verifyAndEnter(){
    try{
      if(!$('#acceptLegal').checked)throw new Error('Review and accept the Terms, Privacy Notice, and Web3 Risk Disclosure first.');
      if(!state.account||!state.provider){const c=await M.connectMetaMask(true);state.provider=c.provider;state.signer=c.signer;state.account=c.address;}
      if(!state.signer)state.signer=await state.provider.getSigner();
      setStep(2,'active','Checking name');
      state.parsed=M.parseDirectName($('#clubLabel').value,state.runtime.parentName);
      $('#clubLabel').value=state.parsed.label;
      setStep(2,'done',state.parsed.name);
      setStep(3,'active','Reading Ethereum Mainnet');setStatus(`Verifying current ownership of <b>${esc(state.parsed.name)}</b>…`,'warn');
      state.ownership=await M.verifyOwnership(state.parsed,state.account,state.provider);
      setStep(3,'done',state.ownership.ownershipPath);
      state.tier=await M.resolveTier(state.parsed.name,state.parsed.node);
      setStep(4,'active','Signature required');setStatus('Ownership verified. MetaMask will ask for one readable, gasless activation signature.','ok');
      const session=await M.createSession({parsed:state.parsed,ownership:state.ownership,address:state.account,signer:state.signer,tierInfo:state.tier});
      setStep(4,'done','Activated');
      $('#successName').textContent=session.name;$('#successTier').textContent=session.tier;$('#successWallet').textContent=short(session.address);$('#successPath').textContent=session.ownershipPath;
      $('#successPanel').classList.remove('hidden');$('#enterBtn').href='./member/index.html';
      setStatus(`<b>Membership activated.</b> ${esc(session.name)} is controlled by the connected MetaMask wallet.`,'ok');
      $('#successPanel').scrollIntoView({behavior:'smooth',block:'center'});
    }catch(err){
      const msg=String(err.message||err);setStatus(esc(msg),'bad');
      if(/name|direct|registered|owner|expired/i.test(msg))setStep(3,'error','Not verified');
      else if(/Terms|Privacy|Risk/i.test(msg))setStep(2,'error','Legal acceptance required');
      else setStep(4,'error','Activation stopped');
    }
  }
  async function resume(){
    try{setStatus('Revalidating the saved session against Ethereum Mainnet…','warn');const session=await M.validateSession({interactive:true});location.href='./member/index.html';}
    catch(err){M.clearSession();$('#resumeBtn').classList.add('hidden');setStatus(esc(err.message||err),'bad');}
  }
  function preview(){
    const value=$('#previewInput').value.trim();if(value.length<10){toast('Describe the outcome in one sentence.');return;}
    const low=value.toLowerCase();let offer='Proof Mission Sprint',buyer='Executives with an expensive decision',outcome='An evidence-backed proceed, repair, narrow, or stop decision';
    if(/vendor|buy|purchase/.test(low)){offer='AI Buyer Proof Room';buyer='CIO, procurement, security, risk, and board stakeholders';outcome='A defensible buy, pilot, negotiate, or reject decision';}
    else if(/deploy|production|workflow/.test(low)){offer='AI Deployment Decision Sprint';buyer='CIO, CTO, risk, compliance, and business-unit owners';outcome='A deployment decision with conditions, monitoring, and rollback';}
    else if(/sell|customer|revenue/.test(low)){offer='Three-Customer Proof Run';buyer='A precisely defined ideal customer';outcome='Three paid comparable missions and one repeat purchase';}
    const rows={Offer:offer,Buyer:buyer,Outcome:outcome,'First proof burden':'Claims, evidence, acceptance tests, independent review, and measurable customer outcome'};
    $('#previewResult').innerHTML=Object.entries(rows).map(([k,v])=>`<div class="result-row"><label>${esc(k)}</label><b>${esc(v)}</b></div>`).join('');
  }
  function bind(){
    $('#connectBtn').addEventListener('click',connect);$('#verifyBtn').addEventListener('click',verifyAndEnter);$('#resumeBtn').addEventListener('click',resume);$('#previewBtn').addEventListener('click',preview);
    $('#clubLabel').addEventListener('keydown',e=>{if(e.key==='Enter')verifyAndEnter();});
    if(window.ethereum){window.ethereum.on?.('accountsChanged',()=>{M.clearSession();location.reload();});window.ethereum.on?.('chainChanged',()=>{M.clearSession();location.reload();});}
    document.querySelectorAll('[data-preset]').forEach(b=>b.addEventListener('click',()=>{$('#previewInput').value=b.dataset.preset;preview();}));
  }
  document.addEventListener('DOMContentLoaded',()=>{bind();detect();});
})();
