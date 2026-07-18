(function(){
  'use strict';
  const M=window.GoalOSMembership,$=s=>document.querySelector(s),$$=s=>Array.from(document.querySelectorAll(s));
  let session=null;
  const short=a=>a?`${a.slice(0,6)}…${a.slice(-4)}`:'—';
  const esc=s=>String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  function showTab(id){$$('.member-section').forEach(s=>s.classList.toggle('active',s.dataset.memberSection===id));$$('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===id));if(id==='machine'&&!$('#machineFrame').src)$('#machineFrame').src='./machine.html';}
  function auditRows(s){
    const rows=[['Direct ENS membership',s.name],['Current wallet',s.address],['ENS node',s.node],['Ownership path',s.ownershipPath],['Wrapped name',String(!!s.wrapped)],['Wrapped expiry',s.wrappedExpiry?new Date(s.wrappedExpiry*1000).toLocaleString():'Not applicable / not set'],['Profile',s.tier],['Tier source',s.tierSource],['Operator manifest verified',String(!!s.manifestVerified)],['Terms version',s.termsVersion],['Privacy version',s.privacyVersion],['Session issued',new Date(s.issuedAt).toLocaleString()],['Session expires',new Date(s.expiresAt).toLocaleString()],['Release',s.releaseVersion]];
    $('#auditTable').innerHTML=rows.map(([k,v])=>`<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join('');
    $('#auditConsole').textContent=JSON.stringify({membership:{name:s.name,node:s.node,wallet:s.address,ownershipPath:s.ownershipPath,tier:s.tier,capabilities:s.capabilities},legal:{termsVersion:s.termsVersion,privacyVersion:s.privacyVersion},session:{issuedAt:s.issuedAt,expiresAt:s.expiresAt,lastValidatedAt:s.lastValidatedAt||null},boundary:'Static Web3 access gating controls normal use and entitlement. It does not make publicly hosted JavaScript confidential or impossible to copy after legitimate delivery.'},null,2);
  }
  function resources(s){
    const cap=new Set(s.capabilities||[]),all=[
      ['Complete Money Machine','The complete browser-local commercial institution.','machine','complete-machine'],
      ['Five-Year Financial Model','Editable revenue, ARR, cash, margin, hiring, and scenario workbook.','./resources/GoalOS_Financial_Projections_v1.xlsx','financial-model'],
      ['Three-Customer Proof Run','Real-customer, delivery, evidence, repeat-purchase, and contribution-margin tracker.','./tools/proof-run.html','proof-run'],
      ['Signed Execution Request','Gasless member intent for paid high-touch GoalOS execution.','./tools/execution-request.html','execution-request'],
      ['Legal & Governance Index','Terms, privacy, membership, risk, security, claim boundary, and reuse notices.','../legal/LEGAL_INDEX.html','member-resources'],
      ['Deployment & Mainnet Acceptance','GitHub Pages, IPFS, ENS contenthash, and controlled-wallet verification.','../deploy/LIVE_MAINNET_ACCEPTANCE_CHECKLIST.md','member-resources']
    ];
    $('#resourceGrid').innerHTML=all.map(([name,desc,href,need])=>{const allowed=cap.has(need)||s.tier==='Sovereign';let action;if(!allowed)action='<span class="small muted">Profile capability not assigned</span>';else if(href==='machine')action='<button class="btn small" data-open-machine>Open in GoalOS</button>';else action=`<a class="btn small" href="${href}" ${href.endsWith('.xlsx')?'download':''}>Open / download</a>`;return `<article class="resource" style="opacity:${allowed?1:.55}"><h4>${esc(name)}</h4><p>${esc(desc)}</p>${action}</article>`;}).join('');
    document.querySelector('[data-open-machine]')?.addEventListener('click',()=>showTab('machine'));
  }
  async function downloadReceipt(){try{const r=await M.activationReceipt();M.downloadJson(`goalos-agi-club-activation-${session.label}.json`,r);}catch(e){alert(e.message||e);}}
  async function revalidate(){try{$('#auditConsole').textContent='Revalidating current Ethereum Mainnet ownership…';session=await M.validateSession({interactive:true});render(session);$('#auditConsole').textContent='PASS — current ownership, signature, legal versions, and profile were revalidated.';}catch(e){$('#auditConsole').textContent='FAIL — '+(e.message||e);}}
  function render(s){session=s;$('#memberName').textContent=s.name;$('#memberWallet').textContent=short(s.address);$('#memberPath').textContent=s.ownershipPath;$('#memberTier').textContent=s.tier;$('#sideMemberName').textContent=s.name;$('#sideMemberMeta').textContent=`${s.tier} · ${short(s.address)} · Ethereum Mainnet`;$('#goalos-member-badge').textContent=`${s.name} · ${s.tier}`;auditRows(s);resources(s);if(!$('#machineFrame').src)$('#machineFrame').src='./machine.html';}
  function bind(){$$('[data-tab]').forEach(b=>b.addEventListener('click',()=>showTab(b.dataset.tab)));$('#logoutBtn').addEventListener('click',()=>{M.clearSession();location.href='../index.html#access';});$('#downloadReceiptBtn').addEventListener('click',downloadReceipt);$('#downloadReceiptBtn2').addEventListener('click',downloadReceipt);$('#revalidateBtn').addEventListener('click',revalidate);}
  bind();window.addEventListener('goalos:member-verified',e=>render(e.detail));
})();
