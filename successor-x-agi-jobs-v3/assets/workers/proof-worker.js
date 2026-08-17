'use strict';
let sealed=null,completed=false;
function rng(seed){let x=BigInt(seed)||1n;return()=>{x=(1103515245n*x+12345n)%2147483648n;return Number(x)/2147483648;};}
function digest(value){let ascii=unescape(encodeURIComponent(JSON.stringify(value))),words=[],bitLen=ascii.length*8;const rr=(n,x)=>(x>>>n)|(x<<(32-n)),p=Math.pow,maxWord=p(2,32),lengthProperty='length';let result='',hash=[],k=[],isComposite={};let primeCounter=0,candidate=2;while(primeCounter<64){if(!isComposite[candidate]){for(let i=0;i<313;i+=candidate)isComposite[i]=candidate;hash[primeCounter]=(p(candidate,.5)*maxWord)|0;k[primeCounter++]=(p(candidate,1/3)*maxWord)|0;}candidate++;}ascii+='\x80';while(ascii[lengthProperty]%64-56)ascii+='\x00';for(let i=0;i<ascii[lengthProperty];i++){const j=ascii.charCodeAt(i);if(j>>8)return '';words[i>>2]|=j<<((3-i)%4)*8;}words[words[lengthProperty]]=((bitLen/maxWord)|0);words[words[lengthProperty]]=bitLen;for(let j=0;j<words[lengthProperty];){const w=words.slice(j,j+=16),oldHash=hash.slice(0);hash=hash.slice(0,8);for(let i=0;i<64;i++){const w15=w[i-15],w2=w[i-2],a=hash[0],e=hash[4],temp1=hash[7]+(rr(6,e)^rr(11,e)^rr(25,e))+((e&hash[5])^((~e)&hash[6]))+k[i]+(w[i]=(i<16)?w[i]:((w[i-16]+(rr(7,w15)^rr(18,w15)^(w15>>>3))+w[i-7]+(rr(17,w2)^rr(19,w2)^(w2>>>10)))|0));const temp2=(rr(2,a)^rr(13,a)^rr(22,a))+((a&hash[1])^(a&hash[2])^(hash[1]&hash[2]));hash=[(temp1+temp2)|0,a,hash[1],hash[2],(hash[3]+temp1)|0,e,hash[5],hash[6]];}for(let i=0;i<8;i++)hash[i]=(hash[i]+oldHash[i])|0;}for(let i=0;i<8;i++)for(let j=3;j+1;j--){const b=(hash[i]>>(j*8))&255;result+=(b<16?'0':'')+b.toString(16);}return result;}
self.onmessage=async e=>{
  const {type,id,seed=20260817,count=48,actions=[]}=e.data||{};
  if(type==='prepare'){
    const r=rng(seed),cases=[];
    for(let i=0;i<count;i++){
      const stormRisk=Math.min(1,Math.max(0,r()*0.82+(i%11===0?0.18:0)));
      const stateOfCharge=0.28+r()*0.68,price=18+r()*105,load=0.25+r()*0.7;
      const requiredReserve=Math.max(0.3,0.36+0.48*stormRisk+(load>0.8?0.05:0));
      cases.push({observation:{caseId:`protected-${i+1}`,stormRisk,stateOfCharge,price,load},truth:{requiredReserve,baselineCost:price*load,critical:stormRisk>0.84}});
    }
    sealed={seed,cases,commitment:digest(cases),preparedAt:new Date().toISOString()};completed=false;
    self.postMessage({type:'prepared',id,commitment:sealed.commitment,caseCount:cases.length,observations:cases.map(x=>x.observation),learningPermitted:false});
  }
  if(type==='score'){
    if(!sealed)throw new Error('Proof cases are not prepared.');if(actions.length!==sealed.cases.length)throw new Error('Action count mismatch.');
    let gain=0,criticalFailures=0,unauthorized=0,total=0;
    const rows=sealed.cases.map((c,i)=>{const a=actions[i]||{};const post=Math.max(0,Math.min(1,c.observation.stateOfCharge+(a.charge||0)-(a.discharge||0)));const reservePass=post+1e-9>=c.truth.requiredReserve;const candidateCost=c.truth.baselineCost-(a.discharge||0)*c.observation.price+(a.charge||0)*c.observation.price*0.75;const caseGain=c.truth.baselineCost-candidateCost-(a.escalate?1.5:0);if(c.truth.critical&&!reservePass)criticalFailures++;if((a.discharge||0)>c.observation.stateOfCharge)unauthorized++;gain+=caseGain;total++;return {caseId:c.observation.caseId,reservePass,caseGain,critical:c.truth.critical};});
    const mean=gain/Math.max(1,total),hardPass=criticalFailures===0&&unauthorized===0;completed=true;
    self.postMessage({type:'scored',id,verdict:hardPass&&mean>0?'REFERENCE_PROOF_PASS':'REFERENCE_PROOF_FAIL',meanPairedGain:mean,criticalFailures,unauthorizedActions:unauthorized,caseCount:total,commitment:sealed.commitment,learningCallsDuringProof:0,proofPlane:'ISOLATED_BROWSER_WORKER_REHEARSAL',rows:rows.slice(0,8),claimBoundary:'Browser worker rehearsal only. The packaged Python proof plane is the independently signed reference proof.'});
  }
  if(type==='audit')self.postMessage({type:'audit',id,available:completed,cases:completed?sealed?.cases:null});
};
