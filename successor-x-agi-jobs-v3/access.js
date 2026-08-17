(() => {
'use strict';
const CFG=window.VSI3_CONFIG, CRYPTO=window.GoalOSCrypto;
const KEY='vsi3_access_v3';let memoryReceipt=null;const session={get:k=>{try{return sessionStorage.getItem(k)}catch{return k===KEY&&memoryReceipt?JSON.stringify(memoryReceipt):null}},set:(k,v)=>{try{sessionStorage.setItem(k,v)}catch{if(k===KEY)try{memoryReceipt=JSON.parse(v)}catch{}}},remove:k=>{try{sessionStorage.removeItem(k)}catch{}if(k===KEY)memoryReceipt=null}};
const $=s=>document.querySelector(s);
const normalizeAddress=v=>/^0x[0-9a-fA-F]{40}$/.test(String(v||''))?String(v).toLowerCase():null;
const encodeAddressWord=a=>a.replace(/^0x/,'').toLowerCase().padStart(64,'0');
const decodeAddressWord=data=>normalizeAddress('0x'+String(data||'').replace(/^0x/,'').slice(24,64));
const decodeUintWord=(data,index=0)=>BigInt('0x'+(String(data||'').replace(/^0x/,'').slice(index*64,(index+1)*64)||'0'));
const hexUtf8=text=>'0x'+[...new TextEncoder().encode(text)].map(v=>v.toString(16).padStart(2,'0')).join('');
const randomHex=(bytes=16)=>{const out=new Uint8Array(bytes);crypto.getRandomValues(out);return [...out].map(v=>v.toString(16).padStart(2,'0')).join('');};
const short=a=>a?`${a.slice(0,6)}…${a.slice(-4)}`:'—';
const status=(message,tone='neutral')=>{const el=$('#accessStatus');if(!el)return;el.className=`access-status ${tone}`;el.innerHTML=`<span class="status-dot"></span><span>${message}</span>`;};
const busy=value=>document.querySelectorAll('.access-action').forEach(b=>b.disabled=value);
async function rpc(method,params=[]){if(!window.ethereum)throw new Error('No injected Ethereum wallet was found. Open this page in a wallet-enabled browser.');return window.ethereum.request({method,params});}
async function ensureMainnet(){let chain=await rpc('eth_chainId');if(chain!==CFG.ethereumChainId){try{await rpc('wallet_switchEthereumChain',[{chainId:CFG.ethereumChainId}]);}catch{throw new Error('Switch the wallet to Ethereum Mainnet, then try again.');}chain=await rpc('eth_chainId');}if(chain!==CFG.ethereumChainId)throw new Error('Ethereum Mainnet is required.');return chain;}
async function connectWallet(){const accounts=await rpc('eth_requestAccounts');const account=normalizeAddress(accounts?.[0]);if(!account)throw new Error('The wallet did not return a valid Ethereum address.');await ensureMainnet();return account;}
async function ethCall(to,data){return rpc('eth_call',[{to,data},'latest']);}
async function tokenBalance(account){return BigInt(await ethCall(CFG.token.contract,'0x70a08231'+encodeAddressWord(account)));}
function requiredRaw(){return BigInt(CFG.token.minimumWhole)*10n**BigInt(CFG.token.decimals);}
function validateLabel(label){const v=String(label||'').trim().toLowerCase();if(!/^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/.test(v))throw new Error('Enter one ASCII label using letters, numbers or internal hyphens. Do not enter dots.');return v;}
async function currentClubOwner(label){
  const fullName=`${validateLabel(label)}.${CFG.ens.suffix}`;
  const node=CRYPTO.namehash(fullName);
  const registryOwner=decodeAddressWord(await ethCall(CFG.ens.registry,'0x02571be3'+node.slice(2)));
  if(!registryOwner||/^0x0{40}$/.test(registryOwner))return {fullName,node,registryOwner:null,effectiveOwner:null,wrapped:false,expiry:null};
  const wrapper=CFG.ens.nameWrappers.find(w=>w.toLowerCase()===registryOwner);
  if(!wrapper)return {fullName,node,registryOwner,effectiveOwner:registryOwner,wrapped:false,expiry:null};
  let result;
  try{result=await ethCall(wrapper,CRYPTO.selector('getData(uint256)')+node.slice(2));}
  catch{result=await ethCall(wrapper,'0x6352211e'+node.slice(2));}
  const clean=String(result).replace(/^0x/,'');
  const effectiveOwner=decodeAddressWord(clean);
  const expiry=clean.length>=192?Number(decodeUintWord(clean,2)):null;
  return {fullName,node,registryOwner,effectiveOwner,wrapped:true,expiry};
}
function expectedMessage(r){return [
  'Successor Omega x AGI Jobs v3 Access Receipt',
  `Application: ${r.appId}`,
  `Origin: ${r.origin}`,
  `Wallet: ${String(r.wallet).toLowerCase()}`,
  `Route: ${r.route}`,
  `Issued: ${r.issuedAt}`,
  `Expires: ${r.expiresAt}`,
  `Nonce: ${r.nonce}`,
  'Authority created: NONE',
  'No token approval, transfer, payment, staking, locking or custody is requested for access.'
].join('\n');}
async function signReceipt(route,account,details){
  const issuedAt=new Date().toISOString();
  const expiresAt=new Date(Date.now()+CFG.access.sessionMinutes*60000).toISOString();
  const origin=location.origin==='null'?'local-file':location.origin;
  const block=await rpc('eth_blockNumber');
  const receipt={schema:'SuccessorOmega.AccessReceipt.v3',appId:CFG.appId,version:CFG.version,route,wallet:account,chainId:CFG.ethereumChainId,origin,blockNumber:Number(BigInt(block)),issuedAt,expiresAt,nonce:randomHex(),authorityCreated:'NONE',details};
  receipt.message=expectedMessage(receipt);
  receipt.signature=await rpc('personal_sign',[hexUtf8(receipt.message),account]);
  return receipt;
}
function validTime(r){return r&&r.version===CFG.version&&r.appId===CFG.appId&&r.authorityCreated==='NONE'&&r.origin===(location.origin==='null'?'local-file':location.origin)&&Date.parse(r.expiresAt)>Date.now();}
function getReceipt(){try{return JSON.parse(session.get(KEY)||'null');}catch{return memoryReceipt;}}
function store(r){memoryReceipt=r;session.set(KEY,JSON.stringify(r));}
function unlock(r){
  document.body.classList.remove('access-locked');
  $('#accessGate')?.classList.add('hidden');
  const shell=$('#applicationShell');if(shell){shell.inert=false;shell.setAttribute('aria-hidden','false');}
  const chip=$('#accessChip');if(chip){chip.textContent=r.route==='LOCAL_QA'?'Local QA reference':`Qualified · ${short(r.wallet)}`;chip.className='chip pass';}
  window.dispatchEvent(new CustomEvent('vsi3:access-granted',{detail:r}));
}
function lock(reason='Fresh access verification is required.'){
  session.remove(KEY);
  document.body.classList.add('access-locked');
  $('#accessGate')?.classList.remove('hidden');
  const shell=$('#applicationShell');if(shell){shell.inert=true;shell.setAttribute('aria-hidden','true');}
  const chip=$('#accessChip');if(chip){chip.textContent='Access locked';chip.className='chip neutral';}
  status(reason,'neutral');
  window.dispatchEvent(new CustomEvent('vsi3:access-locked',{detail:{reason}}));
}
async function revalidate(r=getReceipt()){
  if(!validTime(r))throw new Error('The signed access session expired.');
  if(r.route==='LOCAL_QA')return r;
  await ensureMainnet();
  const accounts=await rpc('eth_accounts');const account=normalizeAddress(accounts?.[0]);
  if(!account||account!==normalizeAddress(r.wallet))throw new Error('The connected wallet changed.');
  if(r.route==='AGIALPHA_DIRECT_BALANCE'){
    const balance=await tokenBalance(account);if(balance<requiredRaw())throw new Error('The connected wallet no longer holds the required direct AGIALPHA balance.');
    r.details={...r.details,balanceRaw:balance.toString(),checkedAt:new Date().toISOString()};
  }else if(r.route==='AGI_CLUB_DIRECT_OWNER'){
    const result=await currentClubOwner(r.details?.label);if(result.effectiveOwner!==account)throw new Error('The connected wallet is no longer the current direct owner of that AGI Club name.');
    if(result.expiry&&result.expiry<=Math.floor(Date.now()/1000))throw new Error('The wrapped AGI Club name is expired.');
    r.details={...r.details,...result,checkedAt:new Date().toISOString()};
  }else throw new Error('Unknown access route.');
  store(r);return r;
}
async function grantAccess(r,message){
  if(CFG.protectedGatewayEndpoint){
    status('Eligibility qualifies. Establishing a protected server session…','working');
    const endpoint=String(CFG.protectedGatewayEndpoint).replace(/\/$/,'')+'/api/session';
    const response=await fetch(endpoint,{method:'POST',credentials:'include',headers:{'Content-Type':'application/json','X-VSI3-App':CFG.appId},body:JSON.stringify({accessReceipt:r})});
    let body={};try{body=await response.json();}catch{}
    if(!response.ok||!body.protectedUrl)throw new Error(body.detail||body.error||'The protected access gateway rejected the session.');
    status('Protected session established. Redirecting to the private institution…','pass');
    location.assign(body.protectedUrl);return;
  }
  store(r);unlock(r);status(message,'pass');
}
async function verifyToken(){busy(true);status('Connecting wallet and reading the direct Mainnet balance…','working');try{
  const account=await connectWallet(),balance=await tokenBalance(account);
  if(balance<requiredRaw())throw new Error(`Direct balance does not meet the ${Number(CFG.token.minimumWhole).toLocaleString('en-CA')} AGIALPHA threshold.`);
  status('Balance qualifies. Sign the access receipt to open the institution.','working');
  const r=await signReceipt('AGIALPHA_DIRECT_BALANCE',account,{contract:CFG.token.contract,balanceRaw:balance.toString(),minimumRaw:requiredRaw().toString(),directBalanceOnly:true});await grantAccess(r,'Access granted through direct AGIALPHA balance.');
}catch(e){status(e.message||String(e),'fail');}finally{busy(false);}}
async function verifyClub(){busy(true);status('Connecting wallet and resolving current direct AGI Club ownership…','working');try{
  const account=await connectWallet(),label=validateLabel($('#clubLabel')?.value),result=await currentClubOwner(label);
  if(result.effectiveOwner!==account)throw new Error(`This wallet is not the current direct owner of ${result.fullName}.`);
  if(result.expiry&&result.expiry<=Math.floor(Date.now()/1000))throw new Error('The wrapped AGI Club name is expired.');
  status('Ownership qualifies. Sign the access receipt to open the institution.','working');
  const r=await signReceipt('AGI_CLUB_DIRECT_OWNER',account,{label,name:result.fullName,node:result.node,wrapped:result.wrapped,expiry:result.expiry,directOwnerOnly:true});await grantAccess(r,'Access granted through current direct AGI Club ownership.');
}catch(e){status(e.message||String(e),'fail');}finally{busy(false);}}
function localQa(){const allowed=location.protocol==='file:'||['localhost','127.0.0.1'].includes(location.hostname)||new URLSearchParams(location.search).has('qa');if(!allowed)return;const now=new Date();const r={schema:'SuccessorOmega.AccessReceipt.v3',appId:CFG.appId,version:CFG.version,route:'LOCAL_QA',wallet:'0x0000000000000000000000000000000000000000',chainId:CFG.ethereumChainId,origin:location.origin==='null'?'local-file':location.origin,issuedAt:now.toISOString(),expiresAt:new Date(now.getTime()+3600000).toISOString(),authorityCreated:'NONE',details:{nonQualifying:true}};store(r);unlock(r);}
function details(){const el=$('#accessDisclosure');if(el)el.classList.toggle('hidden');}
async function restore(){if(CFG.protectedGatewayEndpoint)return;const r=getReceipt();if(!r)return;try{await revalidate(r);unlock(r);status('Signed eligibility session restored and revalidated.','pass');}catch(e){lock(e.message);}}
function bind(){
  $('#verifyTokenButton')?.addEventListener('click',verifyToken);
  $('#verifyClubButton')?.addEventListener('click',verifyClub);
  $('#accessDetailsButton')?.addEventListener('click',details);
  const q=$('#localQaButton');if(q&&(location.protocol==='file:'||['localhost','127.0.0.1'].includes(location.hostname)||new URLSearchParams(location.search).has('qa')))q.classList.remove('hidden');q?.addEventListener('click',localQa);
  window.ethereum?.on?.('accountsChanged',()=>lock('The connected wallet changed. Verify access again.'));
  window.ethereum?.on?.('chainChanged',()=>lock('The wallet network changed. Verify access again on Ethereum Mainnet.'));
  window.addEventListener('focus',()=>{const r=getReceipt();if(r&&r.route!=='LOCAL_QA')revalidate(r).catch(e=>lock(e.message));});
  setInterval(()=>{const r=getReceipt();if(r&&r.route!=='LOCAL_QA')revalidate(r).catch(e=>lock(e.message));},CFG.access.revalidateSeconds*1000);
  restore();
}
document.addEventListener('DOMContentLoaded',bind);
window.VSI3Access=Object.freeze({getReceipt,revalidate,lock,expectedMessage});
})();
