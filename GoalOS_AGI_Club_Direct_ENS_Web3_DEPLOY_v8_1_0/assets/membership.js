/* GoalOS AGI Club direct-ENS membership core v8.1.0 */
(function(){
  'use strict';
  const scriptUrl = document.currentScript && document.currentScript.src ? document.currentScript.src : location.href;
  const ROOT = new URL('../', scriptUrl);
  const SESSION_KEY = 'goalos_agi_club_direct_ens_session_v8';
  const LABEL_KEY = 'goalos_agi_club_last_label_v8';
  const ENS_REGISTRY = '0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e';
  const NAME_WRAPPER = '0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401';
  const ZERO = '0x0000000000000000000000000000000000000000';
  const REGISTRY_ABI = ['function owner(bytes32 node) view returns (address)'];
  const WRAPPER_ABI = [
    'function ownerOf(uint256 id) view returns (address)',
    'function getData(uint256 id) view returns (address owner,uint32 fuses,uint64 expiry)'
  ];
  const DEFAULT_RUNTIME = {
    releaseVersion:'8.1.0', chainId:1, parentName:'club.agi.eth',
    termsVersion:'8.1.0-2026-07-18', privacyVersion:'8.1.0-2026-07-18',
    defaultTier:'Club', sessionMinutes:60, revalidateMinutes:15,
    tierManifestUrl:'./config/tier-manifest.json', tierManifestSigner:'', tierManifestRequiredWhenSignerConfigured:true,
    enterpriseUrl:'https://www.quebecartificialintelligence.com/agiclub',
    contactEmail:'secretariat@montreal.ai'
  };
  const TIER_CAPABILITIES = {
    Club:['complete-machine','proof-run','execution-request','financial-model','member-resources'],
    Pioneer:['complete-machine','proof-run','execution-request','financial-model','member-resources'],
    Business:['complete-machine','proof-run','execution-request','financial-model','member-resources','sales-systems','recurring-revenue'],
    Sovereign:['complete-machine','proof-run','execution-request','financial-model','member-resources','sales-systems','recurring-revenue','institution-audit','capability-reserve','treasury'],
    Agent:['complete-machine','proof-run','execution-request','financial-model','member-resources','proof-jobs','evidence-dockets','governed-automation'],
    Node:['complete-machine','proof-run','execution-request','financial-model','member-resources','validation','replay','sentinels','proof-audit']
  };
  function normalizeAddress(a){ try{return ethers.getAddress(a);}catch{return '';} }
  function sameAddress(a,b){return !!a && !!b && String(a).toLowerCase()===String(b).toLowerCase();}
  function isZero(a){return !a || sameAddress(a,ZERO);}
  function randomNonce(){
    const bytes=new Uint8Array(16); crypto.getRandomValues(bytes); return Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');
  }
  function canonicalize(value){
    if(Array.isArray(value)) return value.map(canonicalize);
    if(value && typeof value==='object'){
      const out={}; Object.keys(value).sort().forEach(k=>{out[k]=canonicalize(value[k]);}); return out;
    }
    return value;
  }
  function canonicalStringify(value){return JSON.stringify(canonicalize(value));}
  async function loadRuntime(){
    if(window.GOALOS_RUNTIME) return {...DEFAULT_RUNTIME,...window.GOALOS_RUNTIME};
    try{
      const url=new URL('config/runtime.json',ROOT);
      const r=await fetch(url,{cache:'no-store'}); if(!r.ok) throw new Error('runtime fetch failed');
      const loaded={...DEFAULT_RUNTIME,...await r.json()}; if(Number(loaded.chainId)!==1) throw new Error('GoalOS membership must use Ethereum Mainnet (chain ID 1).'); if(String(loaded.parentName||'').toLowerCase()!=='club.agi.eth') throw new Error('Unexpected AGI Club ENS parent configuration.'); return loaded;
    }catch(err){
      if(location.protocol==='file:') return {...DEFAULT_RUNTIME};
      throw new Error('GoalOS runtime configuration could not be loaded or validated. Reload the official deployment or contact the operator.');
    }
  }
  function parseDirectName(raw,parentName='club.agi.eth'){
    const input=String(raw||'').trim();
    if(!input) throw new Error('Enter your AGI Club label, for example “elite”.');
    const suffix='.'+parentName.toLowerCase();
    let full=input.includes('.')?input:`${input}.${parentName}`;
    try{ full=ethers.ensNormalize(full); }catch{ throw new Error('That ENS name is not valid. Enter only your direct AGI Club label.'); }
    const low=full.toLowerCase();
    if(!low.endsWith(suffix)) throw new Error(`Only direct *.${parentName} names activate this edition.`);
    const label=full.slice(0,-suffix.length);
    if(!label || label.includes('.')) throw new Error(`Enter one direct name only, such as elite.${parentName}.`);
    const parentLabels=parentName.split('.');
    if(full.split('.').length!==parentLabels.length+1) throw new Error(`Nested names are not accepted. Use one direct *.${parentName} name.`);
    const normalized=ethers.ensNormalize(`${label}.${parentName}`);
    return {label:normalized.slice(0,-suffix.length),name:normalized,node:ethers.namehash(normalized)};
  }
  function createProvider(){
    if(!window.ethereum) throw new Error('MetaMask was not detected. Open this page inside MetaMask or install the MetaMask extension.');
    return new ethers.BrowserProvider(window.ethereum,'any');
  }
  async function currentChainId(provider){ return Number((await provider.getNetwork()).chainId); }
  async function ensureMainnet(provider,interactive=true){
    let id=await currentChainId(provider);
    if(id===1) return 1;
    if(!interactive) throw new Error('Switch MetaMask to Ethereum Mainnet.');
    try{await window.ethereum.request({method:'wallet_switchEthereumChain',params:[{chainId:'0x1'}]});}
    catch{throw new Error('Ethereum Mainnet is required. Approve the network switch in MetaMask.');}
    id=await currentChainId(new ethers.BrowserProvider(window.ethereum,'any'));
    if(id!==1) throw new Error('Ethereum Mainnet is required.');
    return id;
  }
  async function connectMetaMask(interactive=true){
    const provider=createProvider();
    const method=interactive?'eth_requestAccounts':'eth_accounts';
    const accounts=await provider.send(method,[]);
    if(!accounts||!accounts[0]) throw new Error(interactive?'MetaMask did not return an account.':'Reconnect MetaMask to continue.');
    await ensureMainnet(provider,interactive);
    const address=normalizeAddress(accounts[0]);
    return {provider,address,signer:interactive?await provider.getSigner():null};
  }
  async function verifyOwnership(nameOrParsed,address,provider){
    const runtime=await loadRuntime();
    const parsed=typeof nameOrParsed==='string'?parseDirectName(nameOrParsed,runtime.parentName):nameOrParsed;
    const account=normalizeAddress(address);
    if(!account) throw new Error('The connected wallet address is invalid.');
    const registry=new ethers.Contract(ENS_REGISTRY,REGISTRY_ABI,provider);
    const registryOwner=normalizeAddress(await registry.owner(parsed.node));
    if(isZero(registryOwner)) throw new Error(`${parsed.name} is not currently registered in the ENS Registry.`);
    if(sameAddress(registryOwner,account)){
      return {...parsed,owner:account,registryOwner,wrapped:false,expiry:0,ownershipPath:'ENS Registry'};
    }
    if(sameAddress(registryOwner,NAME_WRAPPER)){
      const wrapper=new ethers.Contract(NAME_WRAPPER,WRAPPER_ABI,provider);
      const tokenId=BigInt(parsed.node);
      let data; try{data=await wrapper.getData(tokenId);}catch{throw new Error('The wrapped ENS ownership record could not be read.');}
      const dataOwner=normalizeAddress(data.owner??data[0]);
      const expiry=Number(data.expiry??data[2]??0);
      let nftOwner=''; try{nftOwner=normalizeAddress(await wrapper.ownerOf(tokenId));}catch{}
      const effective=nftOwner||dataOwner;
      const now=Math.floor(Date.now()/1000);
      if(expiry>0 && expiry<=now) throw new Error(`${parsed.name} has expired in the ENS Name Wrapper.`);
      if(!sameAddress(effective,account)) throw new Error(`The connected wallet is not the current owner of ${parsed.name}.`);
      if(dataOwner && !sameAddress(dataOwner,account)) throw new Error(`The Name Wrapper ownership record for ${parsed.name} does not match this wallet.`);
      return {...parsed,owner:account,registryOwner,wrapped:true,expiry,ownershipPath:'ENS Name Wrapper'};
    }
    throw new Error(`The connected wallet is not the current owner of ${parsed.name}.`);
  }
  function buildSiweMessage({runtime,name,address,nonce,issuedAt,expiresAt}){
    const domain=location.host||'local.goalos';
    const uri=location.href.split('#')[0];
    return `${domain} wants you to sign in with your Ethereum account:\n${address}\n\nActivate GoalOS Autonomous Money Machine Ω for ${name}. This proves wallet control only. It is not a transaction, token approval, contract, payment, or asset transfer.\n\nURI: ${uri}\nVersion: 1\nChain ID: 1\nNonce: ${nonce}\nIssued At: ${issuedAt}\nExpiration Time: ${expiresAt}\nResources:\n- ens:${name}\n- goalos:terms:${runtime.termsVersion}\n- goalos:privacy:${runtime.privacyVersion}`;
  }
  async function resolveTier(name,node){
    const runtime=await loadRuntime();
    let tier=runtime.defaultTier||'Club', source='direct-owner-default', status='active', manifest=null;
    const signerExpected=normalizeAddress(runtime.tierManifestSigner||'');
    if(!signerExpected) return {tier,capabilities:TIER_CAPABILITIES[tier]||TIER_CAPABILITIES.Club,source,status,manifestVerified:false};
    try{
      const url=new URL(runtime.tierManifestUrl||'./config/tier-manifest.json',ROOT);
      const res=await fetch(url,{cache:'no-store'}); if(!res.ok) throw new Error('manifest unavailable');
      manifest=await res.json();
      const recovered=normalizeAddress(ethers.verifyMessage(canonicalStringify(manifest.payload),manifest.signature));
      if(!sameAddress(recovered,signerExpected)) throw new Error('manifest signer mismatch');
      if(!manifest.payload||manifest.payload.schema!=='goalos.agi-club.tier-manifest.v1') throw new Error('manifest schema mismatch'); if(Number(manifest.payload.chainId)!==1 || String(manifest.payload.parentName).toLowerCase()!==String(runtime.parentName).toLowerCase()) throw new Error('manifest scope mismatch'); if(manifest.payload.issuedAt && Date.parse(manifest.payload.issuedAt)>Date.now()+300000) throw new Error('manifest issue time is in the future');
      if(manifest.payload.expiresAt && Date.parse(manifest.payload.expiresAt)<=Date.now()) throw new Error('manifest expired');
      const entry=(manifest.payload.members||[]).find(m=>String(m.node||'').toLowerCase()===String(node).toLowerCase()||String(m.name||'').toLowerCase()===String(name).toLowerCase());
      if(entry){ status=entry.status||'active'; if(status!=='active') throw new Error('This AGI Club membership is disabled by the signed operator manifest.'); tier=entry.tier||tier; if(!Object.prototype.hasOwnProperty.call(TIER_CAPABILITIES,tier)) throw new Error('manifest tier is not recognized'); source='operator-signed-manifest'; }
      return {tier,capabilities:TIER_CAPABILITIES[tier]||TIER_CAPABILITIES.Club,source,status,manifestVerified:true};
    }catch(err){
      console.error('Tier manifest verification failed:',err);
      if(runtime.tierManifestRequiredWhenSignerConfigured!==false) throw new Error('The operator-signed capability manifest could not be verified. Access is paused until the operator repairs the signed manifest.');
      return {tier,capabilities:TIER_CAPABILITIES[tier]||TIER_CAPABILITIES.Club,source,status,manifestVerified:false,manifestError:String(err.message||err)};
    }
  }
  async function createSession({parsed,ownership,address,signer,tierInfo}){
    const runtime=await loadRuntime();
    const issuedAt=new Date().toISOString();
    const expiresAt=new Date(Date.now()+Number(runtime.sessionMinutes||60)*60_000).toISOString();
    const nonce=randomNonce();
    const message=buildSiweMessage({runtime,name:parsed.name,address,nonce,issuedAt,expiresAt});
    const signature=await signer.signMessage(message);
    const recovered=normalizeAddress(ethers.verifyMessage(message,signature));
    if(!sameAddress(recovered,address)) throw new Error('MetaMask signature verification failed.');
    const session={schema:'goalos.agi-club.session.v1',releaseVersion:runtime.releaseVersion,origin:location.origin,activationUri:location.href.split('#')[0],name:parsed.name,label:parsed.label,node:parsed.node,address:normalizeAddress(address),ownershipPath:ownership.ownershipPath,wrapped:ownership.wrapped,wrappedExpiry:ownership.expiry||0,tier:tierInfo.tier,capabilities:tierInfo.capabilities,tierSource:tierInfo.source,manifestVerified:!!tierInfo.manifestVerified,termsVersion:runtime.termsVersion,privacyVersion:runtime.privacyVersion,issuedAt,expiresAt,nonce,message,signature};
    sessionStorage.setItem(SESSION_KEY,JSON.stringify(session));
    localStorage.setItem(LABEL_KEY,parsed.label);
    return session;
  }
  function readSession(){try{return JSON.parse(sessionStorage.getItem(SESSION_KEY)||'null');}catch{return null;}}
  function clearSession(){sessionStorage.removeItem(SESSION_KEY);}
  function lastLabel(){return localStorage.getItem(LABEL_KEY)||'';}
  async function validateSession({interactive=false}={}){
    const session=readSession(); if(!session) throw new Error('No verified AGI Club session is available.');
    const runtime=await loadRuntime();
    if(session.schema!=='goalos.agi-club.session.v1') {clearSession(); throw new Error('The saved membership session format is invalid.');}
    if(session.releaseVersion!==runtime.releaseVersion){clearSession();throw new Error('The GoalOS release changed. Verify membership again.');}
    if(session.origin && session.origin!==location.origin){clearSession();throw new Error('The saved membership session belongs to a different site origin.');}
    if(Date.parse(session.expiresAt)<=Date.now()) {clearSession(); throw new Error('Your verified session expired. Verify the ENS name again.');}
    const recovered=normalizeAddress(ethers.verifyMessage(session.message,session.signature));
    if(!sameAddress(recovered,session.address)) {clearSession(); throw new Error('The saved membership signature is invalid.');}
    if(session.termsVersion!==runtime.termsVersion||session.privacyVersion!==runtime.privacyVersion){clearSession();throw new Error('The legal terms changed. Review and accept the current versions.');}
    const {provider,address}=await connectMetaMask(interactive);
    if(!sameAddress(address,session.address)){clearSession();throw new Error('MetaMask is connected to a different wallet.');}
    const parsed=parseDirectName(session.name,runtime.parentName);
    const ownership=await verifyOwnership(parsed,address,provider);
    const tierInfo=await resolveTier(parsed.name,parsed.node);
    const refreshed={...session,ownershipPath:ownership.ownershipPath,wrapped:ownership.wrapped,wrappedExpiry:ownership.expiry||0,tier:tierInfo.tier,capabilities:tierInfo.capabilities,tierSource:tierInfo.source,manifestVerified:!!tierInfo.manifestVerified,lastValidatedAt:new Date().toISOString()};
    sessionStorage.setItem(SESSION_KEY,JSON.stringify(refreshed));
    return refreshed;
  }
  function downloadJson(filename,obj){const blob=new Blob([JSON.stringify(obj,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  async function activationReceipt(){const session=await validateSession({interactive:false});return {schema:'goalos.agi-club.activation-receipt.v1',generatedAt:new Date().toISOString(),membership:{name:session.name,node:session.node,wallet:session.address,ownershipPath:session.ownershipPath,wrapped:session.wrapped,wrappedExpiry:session.wrappedExpiry,tier:session.tier,tierSource:session.tierSource},legal:{termsVersion:session.termsVersion,privacyVersion:session.privacyVersion},session:{issuedAt:session.issuedAt,expiresAt:session.expiresAt,nonce:session.nonce,message:session.message,signature:session.signature},boundary:'This receipt proves a local wallet signature and an onchain ownership check at the recorded time. It is not a contract, payment, revenue guarantee, security certification, or permanent entitlement.'};}
  window.GoalOSMembership={ROOT,SESSION_KEY,LABEL_KEY,ENS_REGISTRY,NAME_WRAPPER,ZERO,DEFAULT_RUNTIME,TIER_CAPABILITIES,loadRuntime,parseDirectName,createProvider,connectMetaMask,ensureMainnet,currentChainId,verifyOwnership,resolveTier,createSession,readSession,clearSession,lastLabel,validateSession,activationReceipt,downloadJson,canonicalStringify,normalizeAddress,sameAddress};
})();
