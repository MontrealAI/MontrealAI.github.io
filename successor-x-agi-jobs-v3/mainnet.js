(() => {
'use strict';
const CFG=window.VSI3_CONFIG, K=window.GoalOSCrypto;
const $=s=>document.querySelector(s);
const esc=v=>String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const short=v=>v?`${v.slice(0,6)}…${v.slice(-4)}`:'—';
const hex=n=>'0x'+BigInt(n).toString(16);
const int=h=>Number(BigInt(h||'0x0'));
const word=(clean,i)=>clean.slice(i*64,(i+1)*64).padStart(64,'0');
const uintWord=(clean,i=0)=>BigInt('0x'+(word(clean,i)||'0'));
const addressWord=(clean,i=0)=>'0x'+word(clean,i).slice(24);
const textAt=(clean,offsetWord)=>{try{const off=Number(BigInt('0x'+offsetWord))*2;const len=Number(BigInt('0x'+clean.slice(off,off+64)));const bytes=clean.slice(off+64,off+64+len*2);return new TextDecoder().decode(Uint8Array.from(bytes.match(/.{1,2}/g)||[],x=>parseInt(x,16)));}catch{return ''}};
const topicUint=t=>Number(BigInt(t));
const topicAddress=t=>'0x'+String(t).replace(/^0x/,'').slice(24);
let account=null;const storage={get:k=>{try{return localStorage.getItem(k)}catch{return null}},set:(k,v)=>{try{localStorage.setItem(k,v)}catch{}}};
async function rpc(method,params=[]){if(!window.ethereum)throw new Error('A wallet-enabled browser is required for live Mainnet reads.');return window.ethereum.request({method,params});}
async function ensureMainnet(){const chain=await rpc('eth_chainId');if(chain!==CFG.ethereumChainId)throw new Error('Switch the wallet to Ethereum Mainnet.');return chain;}
async function currentAccount(request=false){const a=await rpc(request?'eth_requestAccounts':'eth_accounts');account=a?.[0]?.toLowerCase()||null;return account;}
async function call(to,data,from){return rpc('eth_call',[{to,data,...(from?{from}: {})},'latest']);}
async function codeAt(address,block){return rpc('eth_getCode',[address,hex(block)]);}
const deploymentCacheKey=a=>`vsi3_deploy_block_${a.toLowerCase()}`;
async function discoverDeploymentBlock(address,latest,hint){
  const cached=Number(storage.get(deploymentCacheKey(address))||0);if(cached>0&&cached<=latest)return cached;
  if(hint){const code=await codeAt(address,hint).catch(()=>null);if(code&&code!=='0x'){storage.set(deploymentCacheKey(address),String(hint));return hint;}}
  let low=Math.max(0,latest-3_000_000),high=latest;
  const lowCode=await codeAt(address,low).catch(()=>null);
  if(lowCode&&lowCode!=='0x'){storage.set(deploymentCacheKey(address),String(low));return low;}
  const highCode=await codeAt(address,high);if(!highCode||highCode==='0x')throw new Error(`No contract code found at ${address}.`);
  while(low+1<high){const mid=Math.floor((low+high)/2),code=await codeAt(address,mid);if(code&&code!=='0x')high=mid;else low=mid;}
  storage.set(deploymentCacheKey(address),String(high));return high;
}
async function logsAdaptive(address,fromBlock,toBlock,onProgress){
  const out=[];let cursor=fromBlock,chunk=50_000;
  while(cursor<=toBlock){const end=Math.min(toBlock,cursor+chunk-1);try{
    const rows=await rpc('eth_getLogs',[{address,fromBlock:hex(cursor),toBlock:hex(end)}]);out.push(...rows);cursor=end+1;chunk=Math.min(100_000,Math.max(chunk,20_000)*2);onProgress?.(cursor,toBlock,out.length);
  }catch(e){if(chunk<=500)throw e;chunk=Math.max(500,Math.floor(chunk/2));}}
  return out;
}
const eventDefs={
  JobCreated:{sig:'JobCreated(uint256,address,uint256,uint256,string,uint8,bytes32,string)'},
  JobApplied:{sig:'JobApplied(uint256,address)'},
  CheckpointSubmitted:{sig:'CheckpointSubmitted(uint256,address,string)'},
  CheckpointFailed:{sig:'CheckpointFailed(uint256,address,address)'},
  JobCompletionRequested:{sig:'JobCompletionRequested(uint256,address,string)'},
  JobValidated:{sig:'JobValidated(uint256,address)'},
  JobDisapproved:{sig:'JobDisapproved(uint256,address)'},
  JobDisputed:{sig:'JobDisputed(uint256,address)'},
  JobCompleted:{sig:'JobCompleted(uint256,address,uint256)'},
  JobEmployerRefunded:{sig:'JobEmployerRefunded(uint256,address,address,uint256)'},
  JobExpired:{sig:'JobExpired(uint256,address,address,uint256)'},
  JobCancelled:{sig:'JobCancelled(uint256)'},
  DisputeResolvedWithCode:{sig:'DisputeResolvedWithCode(uint256,address,uint8,string)'},
  NFTIssued:{sig:'NFTIssued(uint256,address,string)'}
};
for(const [name,d] of Object.entries(eventDefs)){d.name=name;d.topic=K.keccak256(d.sig).toLowerCase();}
const byTopic=Object.fromEntries(Object.values(eventDefs).map(x=>[x.topic,x]));
function parseLog(log,source){
  const def=byTopic[String(log.topics?.[0]||'').toLowerCase()];if(!def)return {source,name:'OtherEvent',block:int(log.blockNumber),tx:log.transactionHash,logIndex:int(log.logIndex),raw:log};
  const clean=String(log.data||'0x').replace(/^0x/,'');const topics=log.topics||[];let jobId=topics[1]?topicUint(topics[1]):null;const base={source,name:def.name,jobId,block:int(log.blockNumber),tx:log.transactionHash,logIndex:int(log.logIndex)};
  if(def.name==='JobCreated')return {...base,employer:topicAddress(topics[2]),perJobAgentRoot:topics[3],payout:uintWord(clean,0).toString(),duration:Number(uintWord(clean,1)),specUri:textAt(clean,word(clean,2)),intakeMode:Number(uintWord(clean,3)),details:textAt(clean,word(clean,4))};
  if(['JobApplied','CheckpointSubmitted','JobCompletionRequested','JobValidated','JobDisapproved','JobDisputed','JobCompleted'].includes(def.name))base.actor=topics[2]?topicAddress(topics[2]):null;
  if(def.name==='CheckpointSubmitted'||def.name==='JobCompletionRequested')base.uri=textAt(clean,word(clean,0));
  if(def.name==='JobCompleted')base.reputationPoints=topics[3]?topicUint(topics[3]):Number(uintWord(clean,0));
  if(def.name==='JobEmployerRefunded'||def.name==='JobExpired'){base.employer=topicAddress(topics[2]);base.agent=topicAddress(topics[3]);base.amount=uintWord(clean,0).toString();}
  return base;
}
function reduceJobs(events){
  const jobs=new Map();const ordered=[...events].sort((a,b)=>a.block-b.block||a.logIndex-b.logIndex);
  for(const e of ordered){if(e.jobId===null||e.jobId===undefined)continue;const j=jobs.get(e.jobId)||{id:e.jobId,status:'UNKNOWN',approvals:0,disapprovals:0,events:[]};j.events.push(e);j.lastBlock=e.block;j.lastTx=e.tx;
    if(e.name==='JobCreated')Object.assign(j,{status:'OPEN',employer:e.employer,payout:e.payout,duration:e.duration,specUri:e.specUri,details:e.details,intakeMode:e.intakeMode,createdBlock:e.block});
    else if(e.name==='JobApplied')Object.assign(j,{status:'ASSIGNED',agent:e.actor});
    else if(e.name==='CheckpointSubmitted')j.status='CHECKPOINT';
    else if(e.name==='JobCompletionRequested')Object.assign(j,{status:'IN_REVIEW',completionUri:e.uri});
    else if(e.name==='JobValidated'){j.approvals++;j.status='IN_REVIEW';}
    else if(e.name==='JobDisapproved'){j.disapprovals++;j.status='IN_REVIEW';}
    else if(e.name==='JobDisputed')j.status='DISPUTED';
    else if(e.name==='JobCompleted')Object.assign(j,{status:'COMPLETED',agent:e.actor,reputationPoints:e.reputationPoints});
    else if(e.name==='JobEmployerRefunded')j.status='EMPLOYER_WIN';
    else if(e.name==='JobExpired')j.status='EXPIRED';
    else if(e.name==='JobCancelled')j.status='CANCELLED';
    jobs.set(e.jobId,j);
  }
  return [...jobs.values()].sort((a,b)=>b.id-a.id);
}
const formatToken=raw=>{try{return new Intl.NumberFormat('en-CA',{maximumFractionDigits:3}).format(Number(BigInt(raw)/10n**15n)/1000)+' AGIALPHA';}catch{return '—'}};
function setStatus(text,tone='neutral'){const el=$('#mainnetStatus');if(el){el.textContent=text;el.className=`inline-status ${tone}`;}}
function explorerAddress(a){return `${CFG.explorer}/address/${a}#code`;}
function explorerTx(t){return `${CFG.explorer}/tx/${t}`;}
function renderJobs(rows,meta){const el=$('#mainnetJobs');if(!el)return;
  if(!rows.length){el.innerHTML='<div class="empty-state"><strong>No decodable jobs were returned.</strong><span>The contract connection succeeded, but the selected range contains no recognized JobCreated events.</span></div>';return;}
  el.innerHTML=`<div class="table-wrap"><table><thead><tr><th>Job</th><th>State</th><th>Payout</th><th>Employer / agent</th><th>Validation</th><th>Evidence</th></tr></thead><tbody>${rows.map(j=>`<tr><td><strong>#${j.id}</strong><small>block ${j.createdBlock||j.lastBlock}</small></td><td><span class="state-badge ${j.status.toLowerCase()}">${j.status.replaceAll('_',' ')}</span></td><td>${j.payout?formatToken(j.payout):'—'}</td><td><code>${short(j.employer)}</code><small>${j.agent?short(j.agent):'unassigned'}</small></td><td>${j.approvals} approve · ${j.disapprovals} reject</td><td><a href="${explorerTx(j.lastTx)}" target="_blank" rel="noopener">latest tx ↗</a>${j.specUri?`<small title="${esc(j.specUri)}">spec ${esc(j.specUri.slice(0,38))}${j.specUri.length>38?'…':''}</small>`:''}</td></tr>`).join('')}</tbody></table></div><p class="micro">Live reconstruction from ${meta.events} contract logs through Ethereum block ${meta.latest.toLocaleString('en-CA')}. No indexer or hosted database is trusted as the authoritative source.</p>`;
}
async function refresh(){
  setStatus('Connecting to Ethereum Mainnet…','working');const button=$('#refreshMainnetButton');if(button)button.disabled=true;
  try{await ensureMainnet();await currentAccount(false);const latest=int(await rpc('eth_blockNumber'));
    const prime=CFG.contracts.agiJobManagerPrime;const start=await discoverDeploymentBlock(prime,latest);
    setStatus(`Reading Prime logs from block ${start.toLocaleString('en-CA')}…`,'working');
    const logs=await logsAdaptive(prime,start,latest,(c,e,n)=>setStatus(`Scanning Mainnet ${Math.min(100,Math.round((c-start)/(e-start+1)*100))}% · ${n} events`,'working'));
    const events=logs.map(l=>parseLog(l,'Prime'));const jobs=reduceJobs(events);renderJobs(jobs,{events:logs.length,latest,start});
    const countHex=await call(prime,K.selector('nextJobId()')).catch(()=>null);const count=countHex?Number(BigInt(countHex)):jobs.length;
    const code=await rpc('eth_getCode',[prime,'latest']);
    $('#mainnetSummary').innerHTML=`<div><span>Prime jobs</span><strong>${count.toLocaleString('en-CA')}</strong></div><div><span>Decoded jobs</span><strong>${jobs.length}</strong></div><div><span>Events scanned</span><strong>${logs.length}</strong></div><div><span>Contract code</span><strong>${code&&code!=='0x'?'LIVE':'MISSING'}</strong></div>`;
    setStatus(`Live · Ethereum Mainnet block ${latest.toLocaleString('en-CA')}`,'pass');
  }catch(e){setStatus(e.message||String(e),'fail');const el=$('#mainnetJobs');if(el)el.innerHTML=`<div class="empty-state fail"><strong>Live chain read unavailable.</strong><span>${esc(e.message||e)}</span><span>Connect a wallet on Ethereum Mainnet and retry. The packaged reference cycle remains independently inspectable offline.</span></div>`;
  }finally{if(button)button.disabled=false;}
}
function pad64(h){return String(h).replace(/^0x/,'').padStart(64,'0');}
function encodeUint(v){return BigInt(v).toString(16).padStart(64,'0');}
function encodeAddress(v){if(!/^0x[0-9a-fA-F]{40}$/.test(v))throw new Error('Invalid Ethereum address.');return v.slice(2).toLowerCase().padStart(64,'0');}
function encodeBytes32(v){const h=String(v).replace(/^0x/,'');if(!/^[0-9a-fA-F]{64}$/.test(h))throw new Error('Invalid bytes32 value.');return h.toLowerCase();}
function encodeString(v){const h=[...new TextEncoder().encode(String(v))].map(x=>x.toString(16).padStart(2,'0')).join('');return encodeUint(h.length/2)+h.padEnd(Math.ceil(h.length/64)*64,'0');}
function encodeBytes32Array(arr){return encodeUint(arr.length)+arr.map(encodeBytes32).join('');}
function encodeCall(signature,types,values){
  const head=[],tail=[];let tailBytes=0;const headBytes=types.length*32;
  for(let i=0;i<types.length;i++){const t=types[i],v=values[i];if(t==='string'||t==='bytes32[]'){const enc=t==='string'?encodeString(v):encodeBytes32Array(v);head.push(encodeUint(headBytes+tailBytes));tail.push(enc);tailBytes+=enc.length/2;}else if(t==='uint256'||t==='uint8')head.push(encodeUint(v));else if(t==='address')head.push(encodeAddress(v));else if(t==='bytes32')head.push(encodeBytes32(v));else throw new Error(`Unsupported ABI type ${t}`);}
  return K.selector(signature)+head.join('')+tail.join('');
}
async function sendTx(to,data,label){
  if(!$('#enableTransactions')?.checked)throw new Error('Enable transaction mode before preparing a write.');await ensureMainnet();const from=await currentAccount(true);if(!from)throw new Error('Connect a wallet.');
  const tx={from,to,data};const gas=await rpc('eth_estimateGas',[tx]);const accepted=window.confirm(`${label}\n\nContract: ${to}\nEstimated gas: ${Number(BigInt(gas)).toLocaleString('en-CA')}\n\nThis is a real Ethereum Mainnet transaction. Continue to your wallet?`);if(!accepted)throw new Error('Transaction cancelled before wallet submission.');return rpc('eth_sendTransaction',[{...tx,gas}]);
}
async function approveExact(){const raw=parseWholeToken($('#jobPayout')?.value);const data=encodeCall('approve(address,uint256)',['address','uint256'],[CFG.contracts.agiJobManagerPrime,raw]);return sendTx(CFG.token.contract,data,`Approve exactly ${$('#jobPayout').value} AGIALPHA for the Prime job payout`);}
function parseWholeToken(v){const s=String(v||'').trim();if(!/^\d+(?:\.\d{1,18})?$/.test(s))throw new Error('Enter a non-negative token amount with at most 18 decimal places.');const [w,f='']=s.split('.');return BigInt(w)*10n**18n+BigInt((f+'0'.repeat(18)).slice(0,18));}
async function createJob(){const uri=$('#jobSpecUri').value.trim(),details=$('#jobDetails').value.trim(),duration=BigInt($('#jobDuration').value||0),payout=parseWholeToken($('#jobPayout').value);if(!uri||duration<=0n||payout<=0n)throw new Error('Specification URI, payout and duration are required.');const data=encodeCall('createJob(string,uint256,uint256,string)',['string','uint256','uint256','string'],[uri,payout,duration,details]);return sendTx(CFG.contracts.agiJobManagerPrime,data,'Create a real AGI Job on Ethereum Mainnet');}
async function jobAction(){const id=BigInt($('#actionJobId').value||0),action=$('#jobAction').value,uri=$('#completionUri').value.trim();let data,label;if(action==='requestJobCompletion'){if(!uri)throw new Error('Completion URI is required.');data=encodeCall('requestJobCompletion(uint256,string)',['uint256','string'],[id,uri]);label=`Request completion for job #${id}`;}else{const signature=`${action}(uint256)`;data=encodeCall(signature,['uint256'],[id]);label=`${action} for job #${id}`;}return sendTx(CFG.contracts.agiJobManagerPrime,data,label);}
function txResult(promise){const out=$('#transactionResult');out.textContent='Preparing simulation and wallet request…';promise.then(hash=>{out.innerHTML=`Submitted: <a target="_blank" rel="noopener" href="${explorerTx(hash)}">${hash}</a>`;refresh();}).catch(e=>out.textContent=e.message||String(e));}
function bind(){
  $('#refreshMainnetButton')?.addEventListener('click',refresh);
  $('#approveExactButton')?.addEventListener('click',()=>txResult(approveExact()));
  $('#createJobButton')?.addEventListener('click',()=>txResult(createJob()));
  $('#submitJobActionButton')?.addEventListener('click',()=>txResult(jobAction()));
  $('#jobAction')?.addEventListener('change',e=>$('#completionUriField')?.classList.toggle('hidden',e.target.value!=='requestJobCompletion'));
  document.querySelectorAll('[data-contract]').forEach(a=>{const key=a.dataset.contract,addr=CFG.contracts[key]||CFG.token.contract;a.href=explorerAddress(addr);a.textContent=short(addr);a.title=addr;});
  window.addEventListener('vsi3:access-granted',()=>setTimeout(refresh,100));
  if(window.VSI3Access?.getReceipt?.())setTimeout(refresh,150);
}
document.addEventListener('DOMContentLoaded',bind);
window.VSI3Mainnet=Object.freeze({refresh,parseLog,reduceJobs,encodeCall});
})();
