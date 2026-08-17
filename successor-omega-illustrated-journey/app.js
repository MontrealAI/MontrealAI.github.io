(() => {
'use strict';

const CFG = window.SUCCESSOR_OMEGA_CONFIG;
const DELIVERY = window.SUCCESSOR_OMEGA_DELIVERY;
const $ = (selector, root=document) => root.querySelector(selector);
const $$ = (selector, root=document) => [...root.querySelectorAll(selector)];
const ACCESS_KEY = 'successor_omega_access_v1_0_2';
const ACTIVITY_KEY = 'successor_omega_activity_v1_0_2';
let manifest = null;
let keyBytes = null;
let session = null;
let downloading = false;
let periodicTimer = null;
let inactivityTimer = null;
let provider = null;

const esc = value => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
const normalizeAddress = value => /^0x[0-9a-fA-F]{40}$/.test(String(value || '')) ? String(value).toLowerCase() : null;
const shortAddress = value => value ? `${value.slice(0,6)}…${value.slice(-4)}` : '—';
const hexUtf8 = text => '0x' + [...new TextEncoder().encode(text)].map(value => value.toString(16).padStart(2,'0')).join('');
const b64ToBytes = value => Uint8Array.from(atob(value), char => char.charCodeAt(0));
const bytesToHex = value => [...new Uint8Array(value)].map(byte => byte.toString(16).padStart(2,'0')).join('');
const randomHex = (bytes=16) => { const out = new Uint8Array(bytes); crypto.getRandomValues(out); return bytesToHex(out); };
const encodeAddressWord = address => address.replace(/^0x/,'').toLowerCase().padStart(64,'0');
const decodeAddressWord = data => normalizeAddress(`0x${String(data).replace(/^0x/,'').slice(24,64)}`);
const decodeUintWord = (data,index=0) => BigInt(`0x${String(data).replace(/^0x/,'').slice(index*64,(index+1)*64) || '0'}`);
const byteLabel = count => count < 1024 ? `${count} B` : count < 1048576 ? `${(count/1024).toFixed(1)} KB` : `${(count/1048576).toFixed(count>104857600?0:1)} MB`;
const delay = milliseconds => new Promise(resolve => setTimeout(resolve,milliseconds));

function toast(message) {
  const element = $('#toast');
  element.textContent = message;
  element.classList.remove('hidden');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.add('hidden'), 3800);
}
function modal(title, html) {
  $('#modalTitle').textContent = title;
  $('#modalBody').innerHTML = html;
  $('#modal').classList.remove('hidden');
}
function setStatus(message, tone='neutral') {
  const element = $('#accessStatus');
  element.className = `access-status ${tone}`;
  element.innerHTML = `<span class="access-status-dot"></span><span>${esc(message)}</span>`;
}
function setBusy(value) {
  $$('.access-action').forEach(button => {
    button.dataset.label ??= button.textContent;
    button.disabled = value;
    button.textContent = value ? 'Verifying…' : button.dataset.label;
  });
}
function progress(label, percent) {
  $('#downloadProgress').classList.remove('hidden');
  $('#downloadLabel').textContent = label;
  $('#downloadPercent').textContent = `${Math.round(percent)}%`;
  $('#downloadBar').value = percent;
}
function formatUnits(value, decimals=18, maxFraction=4) {
  const negative = value < 0n;
  let raw = negative ? -value : value;
  const base = 10n ** BigInt(decimals);
  const whole = raw / base;
  let fraction = (raw % base).toString().padStart(decimals,'0').slice(0,maxFraction).replace(/0+$/,'');
  const grouped = new Intl.NumberFormat('en-CA').format(whole);
  return `${negative?'-':''}${grouped}${fraction?'.'+fraction:''}`;
}
function validateClubLabel(input) {
  const value = String(input || '').trim().toLowerCase();
  if (!/^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/.test(value)) throw new Error('Enter one ASCII label using letters, numbers or internal hyphens. Do not enter dots.');
  return value;
}

function chooseProvider() {
  if (provider) return provider;
  if (window.ethereum?.providers?.length) provider = window.ethereum.providers.find(item => item.isMetaMask) || window.ethereum.providers[0];
  else provider = window.ethereum || null;
  return provider;
}
async function rpc(method, params=[]) {
  const active = chooseProvider();
  if (!active?.request) throw new Error('No Ethereum wallet was found. Open this page in a wallet-enabled browser such as MetaMask, Rabby or Coinbase Wallet.');
  return active.request({method,params});
}
async function ensureMainnet() {
  let chainId = await rpc('eth_chainId');
  if (chainId !== CFG.ethereumChainId) {
    try { await rpc('wallet_switchEthereumChain',[{chainId:CFG.ethereumChainId}]); }
    catch { throw new Error('Switch the wallet to Ethereum Mainnet, then try again.'); }
    chainId = await rpc('eth_chainId');
  }
  if (chainId !== CFG.ethereumChainId) throw new Error('Ethereum Mainnet is required.');
  return chainId;
}
async function connectWallet() {
  const accounts = await rpc('eth_requestAccounts');
  const account = normalizeAddress(accounts?.[0]);
  if (!account) throw new Error('The wallet did not return a valid Ethereum address.');
  await ensureMainnet();
  return account;
}
async function ethCall(to, data) { return rpc('eth_call',[{to,data},'latest']); }
async function tokenBalance(account) {
  const data = '0x70a08231' + encodeAddressWord(account);
  return BigInt(await ethCall(CFG.token.contract,data));
}
function requiredTokenRaw() { return BigInt(CFG.token.minimumWhole) * 10n ** BigInt(CFG.token.decimals); }
async function currentClubOwner(clubLabel) {
  if (!window.GoalOSCrypto?.namehash) throw new Error('ENS verification library did not load. Refresh the page and try again.');
  const fullName = `${validateClubLabel(clubLabel)}.${CFG.ens.suffix}`;
  const node = window.GoalOSCrypto.namehash(fullName);
  const ownerData = '0x02571be3' + node.slice(2);
  const registryOwner = decodeAddressWord(await ethCall(CFG.ens.registry,ownerData));
  if (!registryOwner || /^0x0{40}$/.test(registryOwner)) return {fullName,node,registryOwner:null,effectiveOwner:null,wrapped:false,expiry:null};
  const wrapper = CFG.ens.nameWrappers.find(address => address.toLowerCase() === registryOwner);
  if (!wrapper) return {fullName,node,registryOwner,effectiveOwner:registryOwner,wrapped:false,expiry:null};
  const selector = window.GoalOSCrypto.selector('getData(uint256)');
  let result;
  try { result = await ethCall(wrapper,selector+node.slice(2)); }
  catch { result = await ethCall(wrapper,'0x6352211e'+node.slice(2)); }
  const clean = String(result).replace(/^0x/,'');
  const effectiveOwner = decodeAddressWord(clean);
  const expiry = clean.length >= 192 ? Number(decodeUintWord(clean,2)) : null;
  return {fullName,node,registryOwner,effectiveOwner,wrapped:true,expiry};
}

async function createSignedReceipt(route, account, details) {
  const issuedAt = new Date().toISOString();
  const expiresAt = new Date(Date.now()+CFG.access.sessionMinutes*60000).toISOString();
  const origin = location.origin === 'null' ? 'local-file' : location.origin;
  const path = location.pathname;
  const blockNumberHex = await rpc('eth_blockNumber');
  const receipt = {
    schema:'SuccessorOmega.AccessReceipt.v1', appId:CFG.appId, version:CFG.version,
    route, wallet:account, chainId:CFG.ethereumChainId, origin, path,
    blockNumber:Number(BigInt(blockNumberHex)), issuedAt, expiresAt,
    nonce:randomHex(18), authorityCreated:'NONE', details
  };
  const message = [
    'Successor Omega Access Receipt',
    `Application: ${CFG.appId}`,
    `Version: ${CFG.version}`,
    `Origin: ${origin}`,
    `Path: ${path}`,
    `Wallet: ${account}`,
    `Route: ${route}`,
    `Issued: ${issuedAt}`,
    `Expires: ${expiresAt}`,
    `Nonce: ${receipt.nonce}`,
    'Authority created: NONE',
    'No token approval, transfer, payment, staking, locking or custody is requested.'
  ].join('\n');
  setStatus('Eligibility verified. Sign the domain-bound access receipt in your wallet.','busy');
  receipt.signature = await rpc('personal_sign',[hexUtf8(message),account]);
  receipt.message = message;
  return receipt;
}
function saveReceipt(receipt) {
  session = receipt;
  sessionStorage.setItem(ACCESS_KEY,JSON.stringify(receipt));
  markActivity();
}
function loadReceipt() { try { return JSON.parse(sessionStorage.getItem(ACCESS_KEY)||'null'); } catch { return null; } }
function validReceipt(receipt) {
  const origin = location.origin === 'null' ? 'local-file' : location.origin;
  return Boolean(receipt && receipt.version===CFG.version && receipt.origin===origin && receipt.path===location.pathname && new Date(receipt.expiresAt).getTime()>Date.now());
}
async function loadDeliveryKey() {
  if (keyBytes) return keyBytes;
  if (!DELIVERY?.fragments?.length) throw new Error('Publication delivery material did not load. Refresh the page.');
  const bytes = b64ToBytes(DELIVERY.fragments.join(''));
  if (bytes.length !== 32) throw new Error('Publication delivery material is invalid.');
  const digest = await sha(bytes);
  if (digest !== DELIVERY.keySha256) throw new Error('Publication delivery material failed its integrity check.');
  keyBytes = bytes;
  return keyBytes;
}
function unlockApp(receipt) {
  session = receipt;
  document.body.classList.remove('access-locked');
  $('#accessGate').classList.add('hidden');
  const shell = $('#applicationShell');
  shell.removeAttribute('inert');
  shell.setAttribute('aria-hidden','false');
  $('#accessChip').textContent = `Access granted · ${shortAddress(receipt.wallet)}`;
  const routeText = receipt.route==='AGI_CLUB_DIRECT_OWNER' ? `AGI Club: ${receipt.details?.name || 'direct owner'}` : `AGIALPHA: ${receipt.details?.balanceFormatted || 'qualifying direct balance'}`;
  $('#sessionSummary').textContent = `${routeText}. Session expires ${new Date(receipt.expiresAt).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}.`;
  startChecks();
}
async function lockApp(reason='Access requires fresh verification.') {
  session = null;
  keyBytes = null;
  sessionStorage.removeItem(ACCESS_KEY);
  clearInterval(periodicTimer);
  clearInterval(inactivityTimer);
  document.body.classList.add('access-locked');
  $('#accessGate').classList.remove('hidden');
  const shell = $('#applicationShell');
  shell.setAttribute('inert','');
  shell.setAttribute('aria-hidden','true');
  setStatus(reason,'fail');
}
async function verifyTokenRoute() {
  setBusy(true);
  setStatus('Connecting wallet and reading the current direct AGIALPHA balance…','busy');
  try {
    const account = await connectWallet();
    const balance = await tokenBalance(account);
    const minimum = requiredTokenRaw();
    if (balance < minimum) throw new Error(`This wallet holds ${formatUnits(balance,CFG.token.decimals)} AGIALPHA. At least 1,000,000 direct AGIALPHA is required.`);
    const receipt = await createSignedReceipt('AGIALPHA_DIRECT_BALANCE',account,{contract:CFG.token.contract,balanceRaw:balance.toString(),balanceFormatted:formatUnits(balance,CFG.token.decimals),minimumRaw:minimum.toString(),direct:true});
    saveReceipt(receipt);
    unlockApp(receipt);
    setStatus('Current direct AGIALPHA balance verified. Publication unlocked.','pass');
  } catch (error) { setStatus(error.message||String(error),'fail'); }
  finally { setBusy(false); }
}
async function verifyClubRoute() {
  setBusy(true);
  setStatus('Connecting wallet and verifying current direct AGI Club ownership…','busy');
  try {
    const clubLabel = validateClubLabel($('#clubLabel').value);
    const account = await connectWallet();
    const result = await currentClubOwner(clubLabel);
    if (!result.effectiveOwner || result.effectiveOwner!==account) throw new Error(`${result.fullName} is not currently held directly by the connected wallet.`);
    if (result.expiry && result.expiry*1000<=Date.now()) throw new Error(`${result.fullName} is wrapped but its recorded expiry is no longer current.`);
    const receipt = await createSignedReceipt('AGI_CLUB_DIRECT_OWNER',account,{name:result.fullName,node:result.node,wrapped:result.wrapped,registryOwner:result.registryOwner,effectiveOwner:result.effectiveOwner,expiry:result.expiry,direct:true});
    saveReceipt(receipt);
    unlockApp(receipt);
    setStatus('Current direct AGI Club ownership verified. Publication unlocked.','pass');
  } catch (error) { setStatus(error.message||String(error),'fail'); }
  finally { setBusy(false); }
}
async function revalidateAccess(receipt=session||loadReceipt()) {
  if (!validReceipt(receipt)) { await lockApp('The access session expired. Verify eligibility again.'); return false; }
  try {
    const account = normalizeAddress((await rpc('eth_accounts'))?.[0]);
    const chainId = await rpc('eth_chainId');
    if (account!==receipt.wallet || chainId!==CFG.ethereumChainId) throw new Error('Wallet or network changed.');
    if (receipt.route==='AGIALPHA_DIRECT_BALANCE') {
      const balance = await tokenBalance(account);
      if (balance<requiredTokenRaw()) throw new Error('The direct AGIALPHA balance no longer meets the access threshold.');
      receipt.details.balanceRaw=balance.toString();
      receipt.details.balanceFormatted=formatUnits(balance,CFG.token.decimals);
    } else if (receipt.route==='AGI_CLUB_DIRECT_OWNER') {
      const result = await currentClubOwner(receipt.details.name.split('.')[0]);
      if (result.effectiveOwner!==account) throw new Error('Current direct AGI Club ownership is no longer verified.');
      if (result.expiry && result.expiry*1000<=Date.now()) throw new Error('The wrapped AGI Club name is expired.');
      receipt.details={...receipt.details,...result};
    } else throw new Error('Unknown access route.');
    saveReceipt(receipt);
    unlockApp(receipt);
    return true;
  } catch (error) {
    await lockApp(`${error.message||error} Access relocked.`);
    return false;
  }
}
function markActivity() { localStorage.setItem(ACTIVITY_KEY,String(Date.now())); }
function startChecks() {
  clearInterval(periodicTimer);
  clearInterval(inactivityTimer);
  periodicTimer=setInterval(()=>revalidateAccess(),CFG.access.revalidateMinutes*60000);
  inactivityTimer=setInterval(()=>{
    const last=Number(localStorage.getItem(ACTIVITY_KEY)||Date.now());
    if(Date.now()-last>CFG.access.inactivityMinutes*60000) lockApp('Session locked after inactivity. Verify eligibility again.');
  },30000);
}

async function fetchBytes(url,label,onProgress=null,retries=2) {
  let lastError;
  for(let attempt=0;attempt<=retries;attempt++) {
    try {
      const response=await fetch(url,{cache:'no-store'});
      if(!response.ok) throw new Error(`Publication file request failed (${response.status}).`);
      const total=Number(response.headers.get('Content-Length')||0);
      if(!response.body) {
        const output=new Uint8Array(await response.arrayBuffer());
        onProgress?.(output.length,total||output.length);
        return output;
      }
      const reader=response.body.getReader(),chunks=[];
      let loaded=0;
      while(true) {
        const result=await reader.read();
        if(result.done) break;
        chunks.push(result.value); loaded+=result.value.length;
        if(onProgress) onProgress(loaded,total);
        else progress(`Downloading ${label} · ${byteLabel(loaded)}${total?` of ${byteLabel(total)}`:''}`,total?Math.min(72,loaded/total*72):35);
      }
      const output=new Uint8Array(loaded); let offset=0;
      for(const chunk of chunks){output.set(chunk,offset);offset+=chunk.length;}
      return output;
    } catch(error) { lastError=error; if(attempt<retries) await delay(500*(attempt+1)); }
  }
  throw lastError;
}
async function sha(value) { return bytesToHex(await crypto.subtle.digest('SHA-256',value)); }
async function contentKey() { await loadDeliveryKey(); return crypto.subtle.importKey('raw',keyBytes,{name:'AES-GCM'},false,['decrypt']); }
function orderedParts(asset) {
  const parts=[...(asset.parts||[])].sort((a,b)=>a.index-b.index);
  const expected=asset.multipart?.partCount||parts.length;
  if(!parts.length||parts.length!==expected) throw new Error('Publication multipart manifest is incomplete.');
  parts.forEach((part,index)=>{if(part.index!==index+1||!part.file||!part.ivB64||!part.aad)throw new Error('Publication multipart manifest order is invalid.');});
  return parts;
}
async function decryptSingle(asset,key) {
  const encrypted=await fetchBytes(asset.file,asset.label);
  if(encrypted.length!==asset.ciphertextBytes||await sha(encrypted)!==asset.ciphertextSha256)throw new Error('Encrypted publication integrity verification failed.');
  progress(`Opening ${asset.label} locally…`,80);
  const plain=await crypto.subtle.decrypt({name:'AES-GCM',iv:b64ToBytes(asset.ivB64),additionalData:new TextEncoder().encode(asset.aad),tagLength:128},key,encrypted);
  progress(`Verifying ${asset.label}…`,94);
  if(plain.byteLength!==asset.plaintextBytes||await sha(plain)!==asset.plaintextSha256)throw new Error('Publication integrity verification failed after opening.');
  return new Blob([plain],{type:asset.mimeType});
}
async function decryptMultipart(asset,key) {
  const parts=orderedParts(asset),plainParts=[];
  let plainBytes=0;
  for(let index=0;index<parts.length;index++) {
    const part=parts[index],step=90/parts.length,base=index*step;
    const encrypted=await fetchBytes(part.file,`${asset.label} · part ${index+1}/${parts.length}`,(loaded,total)=>{
      const denominator=total||part.ciphertextBytes;
      progress(`Downloading ${asset.label} · part ${index+1}/${parts.length} · ${byteLabel(loaded)} of ${byteLabel(denominator)}`,base+Math.min(step*.70,loaded/denominator*step*.70));
    });
    if(encrypted.length!==part.ciphertextBytes||await sha(encrypted)!==part.ciphertextSha256)throw new Error(`Encrypted part ${index+1} integrity verification failed.`);
    progress(`Opening and verifying ${asset.label} · part ${index+1}/${parts.length}…`,base+step*.82);
    const plain=await crypto.subtle.decrypt({name:'AES-GCM',iv:b64ToBytes(part.ivB64),additionalData:new TextEncoder().encode(part.aad),tagLength:128},key,encrypted);
    if(plain.byteLength!==part.plaintextBytes||await sha(plain)!==part.plaintextSha256)throw new Error(`Publication part ${index+1} integrity verification failed.`);
    plainParts.push(plain); plainBytes+=plain.byteLength;
    progress(`Verified ${asset.label} · part ${index+1}/${parts.length}.`,base+step);
  }
  if(plainBytes!==asset.plaintextBytes||plainBytes!==asset.multipart?.wholePlaintextBytes)throw new Error('Reassembled publication size verification failed.');
  progress(`Assembling ${asset.label}…`,96);
  return new Blob(plainParts,{type:asset.mimeType});
}
async function decryptAsset(asset) {
  if(!(await revalidateAccess())) throw new Error('Fresh eligibility is required.');
  const key=await contentKey();
  const blob=Array.isArray(asset.parts)&&asset.parts.length?await decryptMultipart(asset,key):await decryptSingle(asset,key);
  progress(`${asset.label} is ready.`,100);
  return blob;
}
async function openAsset(id) {
  if(downloading)return;
  const asset=manifest?.assets?.find(item=>item.id===id);
  if(!asset)return toast('Publication asset not found.');
  downloading=true;
  const popup=asset.action==='open'?window.open('about:blank','_blank'):null;
  try {
    if(popup) popup.document.body.innerHTML='<p style="font:16px system-ui;padding:30px">Preparing verified publication…</p>';
    const blob=await decryptAsset(asset),url=URL.createObjectURL(blob);
    if(asset.action==='open') { if(popup) popup.location.href=url; else window.open(url,'_blank','noopener'); }
    else { const anchor=document.createElement('a');anchor.href=url;anchor.download=asset.filename;document.body.appendChild(anchor);anchor.click();anchor.remove(); }
    setTimeout(()=>URL.revokeObjectURL(url),15*60*1000);
    toast(`${asset.label} verified and ready.`);
  } catch(error) {
    if(popup)popup.close();
    toast(error.message||String(error));
    progress(error.message||String(error),0);
  } finally {
    downloading=false;
    setTimeout(()=>$('#downloadProgress').classList.add('hidden'),1200);
  }
}
function accessDetails() {
  modal('How access verification works',`<div class="callout"><strong>Access = current direct AGI Club owner OR current direct holder of at least 1,000,000 official AGIALPHA on Ethereum Mainnet.</strong></div><h3>AGI Club route</h3><p>The browser computes the ENS namehash of the exact single-label name <code>label.club.agi.eth</code>, reads the current owner from the ENS Registry, and—when wrapped—reads the effective owner and expiry from the NameWrapper. The connected wallet must be the current direct owner.</p><h3>AGIALPHA route</h3><p>The browser calls <code>balanceOf(connectedWallet)</code> on <code>${esc(CFG.token.contract)}</code>. Only the connected wallet’s current direct Mainnet balance counts.</p><h3>Access receipt</h3><p>After verification, the wallet signs a domain-bound receipt with a 30-minute expiry and the explicit statement <code>authorityCreated = NONE</code>. No transaction is submitted.</p><h3>Publication delivery</h3><p>Files remain packaged as authenticated AES-GCM objects and are opened locally only after the browser verifies current eligibility. GitHub Pages has no private server runtime, so this follows the same client-side wallet-gate model as the Specialist ASI Training Toolkit. It prevents accidental access through the interface, but it is not a server-side confidentiality boundary.</p><h3>Never requested</h3><p>No token approval, transfer, payment, staking, locking, burning, deposit, custody or transaction authority.</p>`);
}
async function init() {
  try {
    const response=await fetch(CFG.delivery.manifest,{cache:'no-store'});
    if(!response.ok)throw new Error('Publication manifest could not be loaded.');
    manifest=await response.json();
  } catch(error) { setStatus(error.message,'fail'); return; }
  setStatus('Connect an Ethereum wallet to verify current eligibility.','neutral');
  const receipt=loadReceipt();
  if(validReceipt(receipt)) {
    session=receipt;
    setStatus('Revalidating current on-chain eligibility…','busy');
    await revalidateAccess(receipt);
  }
  if('serviceWorker' in navigator && location.protocol==='https:') navigator.serviceWorker.register('sw.js').catch(()=>{});
}

$('#clubAccessButton').addEventListener('click',verifyClubRoute);
$('#tokenAccessButton').addEventListener('click',verifyTokenRoute);
$('#copyContract').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(CFG.token.contract);toast('Official AGIALPHA contract copied.');}catch{toast(CFG.token.contract);}});
$('#accessDetailsButton').addEventListener('click',accessDetails);
$('#modalClose').addEventListener('click',()=>$('#modal').classList.add('hidden'));
$('#modal').addEventListener('click',event=>{if(event.target.id==='modal')$('#modal').classList.add('hidden');});
$('#lockButton').addEventListener('click',()=>lockApp('Session locked. Verify current eligibility to return.'));
$$('.protected-action').forEach(button=>button.addEventListener('click',()=>openAsset(button.dataset.asset)));
['pointerdown','keydown','touchstart','scroll'].forEach(name=>addEventListener(name,markActivity,{passive:true}));
addEventListener('focus',()=>{if(session)revalidateAccess();});
addEventListener('pageshow',event=>{if(event.persisted&&session)revalidateAccess();});
addEventListener('keydown',event=>{if(event.key==='Escape')$('#modal').classList.add('hidden');});
const activeProvider=chooseProvider();
if(activeProvider?.on) {
  activeProvider.on('accountsChanged',()=>{if(session)lockApp('Wallet account changed. Verify again.');});
  activeProvider.on('chainChanged',()=>{if(session)lockApp('Wallet network changed. Ethereum Mainnet is required.');});
  activeProvider.on('disconnect',()=>{if(session)lockApp('Wallet disconnected. Verify again.');});
}
init();
})();
