import {readFile,access} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(here,'..');
const read=async p=>readFile(path.join(root,p),'utf8');
const json=async p=>JSON.parse(await read(p));
const assert=(condition,message)=>{if(!condition) throw new Error(message)};

const requiredFiles=[
  'index.html','README.md','ACTIVATE.md','MCAP-001.md','COVENANT.md',
  'manifest.json','registry.json','ens-records.json','agent-context.json',
  'chronicle.jsonl','verifiers.json','admissions.json','policy.json',
  'rollback.json','treasury.json','proof/index.json',
  'schemas/root-manifest.schema.json','schemas/registry.schema.json',
  'schemas/ens-records.schema.json','schemas/agent-context.schema.json'
];

for(const file of requiredFiles) await access(path.join(root,file));

const [manifest,registry,records,context,verifiers,admissions,policy,rollback,treasury,proof]=await Promise.all([
  json('manifest.json'),json('registry.json'),json('ens-records.json'),json('agent-context.json'),
  json('verifiers.json'),json('admissions.json'),json('policy.json'),json('rollback.json'),json('treasury.json'),json('proof/index.json')
]);

const docs=[manifest,registry,context,verifiers,admissions,policy,rollback,treasury,proof];
for(const doc of docs){
  assert(doc.version==='0.1.0','version mismatch');
  if('specification' in doc) assert(doc.specification==='MCAP-001','specification mismatch');
}

assert(manifest.canonicalRoot==='asi.eth','manifest root mismatch');
assert(registry.root==='asi.eth','registry root mismatch');
assert(records.root==='asi.eth','record-set root mismatch');
assert(context.ens==='asi.eth','agent context root mismatch');
assert(manifest.doctrine.continuity==='Memory may cross.','continuity doctrine changed');
assert(manifest.doctrine.authority==='Authority must requalify.','authority doctrine changed');

const names=registry.entries.map(x=>x.ens);
assert(new Set(names).size===names.length,'duplicate registry ENS entry');
assert(names.includes('successor.asi.eth'),'successor.asi.eth missing');
assert(names.includes('proof.asi.eth'),'proof.asi.eth missing');
assert(names.includes('verifier.asi.eth'),'verifier.asi.eth missing');
assert(names.includes('chronicle.asi.eth'),'chronicle.asi.eth missing');
assert(names.includes('admission.asi.eth'),'admission.asi.eth missing');

const active=registry.entries.filter(x=>x.status==='active');
assert(active.length===0,'Genesis release must not claim active ENS components before activation evidence is published');
assert(admissions.admissions.length===0,'Genesis release must not contain operational admissions');
assert(verifiers.verifiers.length===0,'Genesis release must not claim admitted independent verifiers');
assert(policy.defaultAuthority==='A0','Genesis authority must remain A0');
assert(policy.activeGrants.length===0,'Genesis release must not contain active authority grants');
assert(treasury.economicLayers.treasuryAddresses.length===0,'Genesis release must not claim a live treasury');
assert(proof.proofs.length===0,'Genesis release must not claim completed proofs');

const recordKeys=records.records.map(x=>x.key);
for(const key of ['description','url','org.montrealai.mcap.id','org.montrealai.mcap.manifest','org.montrealai.mcap.covenant','org.montrealai.mcap.registry','org.montrealai.mcap.successor']){
  assert(recordKeys.includes(key),`required ENS record missing: ${key}`);
}
assert(new Set(recordKeys).size===recordKeys.length,'duplicate ENS record key');

const chronicle=(await read('chronicle.jsonl')).trim().split(/\n+/).map((line,i)=>{
  try{return JSON.parse(line)}catch{throw new Error(`invalid Chronicle JSON on line ${i+1}`)}
});
assert(chronicle.length>=1,'Chronicle must contain a Genesis event');
assert(chronicle[0].eventType==='genesis-publication','first Chronicle event must be genesis-publication');
assert(chronicle[0].authority==='A0','Genesis Chronicle authority must be A0');

const html=await read('index.html');
for(const text of ['ASI.ETH','The addressable root of Machine Civilizations','Memory may cross.','Authority must requalify.','Not a swarm.','An institution.']){
  assert(html.includes(text),`canonical interface missing: ${text}`);
}

console.log(`ASI.ETH MCAP-001 v${manifest.version} validation passed`);
console.log(`${requiredFiles.length} files present; ${registry.entries.length} namespace roles; ${records.records.length} prepared ENS records; ${chronicle.length} Chronicle event(s)`);
