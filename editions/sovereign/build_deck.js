const pptxgen = require('pptxgenjs');
const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'GoalOS / MONTREAL.AI';
pptx.subject = 'GoalOS Partner MASTERCLASS Sovereign Edition';
pptx.title = 'GoalOS Partner MASTERCLASS — Sovereign Edition';
pptx.company = 'MONTREAL.AI / QUEBEC.AI';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Georgia', bodyFontFace: 'Arial', lang: 'en-US'
};
pptx.defineSlideMaster({
  title:'MASTER',
  background:{color:'02040A'},
  objects:[
    {rect:{x:0,y:0,w:13.333,h:0.12,fill:{color:'6EE7FF'},line:{color:'6EE7FF',transparency:100}}},
    {text:{text:'GOALOS PARTNER MASTERCLASS · SOVEREIGN EDITION',options:{x:0.45,y:0.2,w:7.6,h:0.22,fontFace:'Arial',fontSize:7.5,bold:true,color:'F3CE78',charSpacing:1.2,margin:0}}},
    {text:{text:'Vincent Boucher · QUEBEC.AI & MONTREAL.AI',options:{x:9.0,y:0.2,w:3.85,h:0.22,fontFace:'Arial',fontSize:7.5,color:'B9C7DF',align:'right',margin:0}}},
    {line:{x:0.45,y:7.16,w:12.43,h:0,line:{color:'314158',width:0.6}}},
    {text:{text:'AI creates output. GoalOS creates proof.',options:{x:0.48,y:7.22,w:6,h:0.18,fontFace:'Georgia',italic:true,fontSize:7.5,color:'8FA0BA',margin:0}}},
    {text:{text:'CONFIDENTIAL PARTNER BRIEFING',options:{x:9.0,y:7.22,w:3.8,h:0.18,fontFace:'Arial',fontSize:7.5,bold:true,color:'8FA0BA',align:'right',margin:0}}}
  ],
  slideNumber:{x:12.85,y:7.2,color:'8FA0BA',fontFace:'Arial',fontSize:8}
});

const C={bg:'02040A',panel:'0C1324',panel2:'111B31',navy:'0B1D33',white:'FBFDFF',muted:'9DACC5',gold:'F3CE78',gold2:'B78324',cyan:'70EDFF',violet:'AA83FF',green:'89F7C2',red:'FF8198',line:'2A3952',amber:'FFD78C'};
const S=pptx.ShapeType;

function slide(){return pptx.addSlide('MASTER')}
function txt(slide,text,x,y,w,h,opt={}){slide.addText(text,{x,y,w,h,fontFace:opt.fontFace||'Arial',fontSize:opt.fontSize||16,color:opt.color||C.white,bold:opt.bold||false,italic:opt.italic||false,align:opt.align||'left',valign:opt.valign||'mid',margin:opt.margin===undefined?0:opt.margin,breakLine:opt.breakLine||false,fit:'shrink',...opt})}
function title(slide,kicker,headline,sub=''){
  txt(slide,kicker.toUpperCase(),0.55,0.55,6.0,0.28,{fontSize:9,bold:true,color:C.gold,charSpacing:1.4});
  txt(slide,headline,0.55,0.88,12.1,1.0,{fontFace:'Georgia',fontSize:34,bold:true,color:C.white,margin:0,breakLine:false});
  if(sub) txt(slide,sub,0.58,1.88,11.9,0.5,{fontSize:14,color:'CAD7EA',margin:0});
}
function pill(slide,text,x,y,w,color=C.gold){slide.addShape(S.roundRect,{x,y,w,h:0.34,rectRadius:0.16,fill:{color,transparency:86},line:{color,transparency:45,width:0.8}});txt(slide,text,x+0.08,y+0.01,w-0.16,0.30,{fontSize:8.5,bold:true,color,align:'center'});}
function panel(slide,x,y,w,h,opts={}){slide.addShape(S.roundRect,{x,y,w,h,rectRadius:0.16,fill:{color:opts.fill||C.panel,transparency:opts.transparency||0},line:{color:opts.line||C.line,width:opts.width||0.8,transparency:opts.lineTransparency||0},shadow:opts.shadow?{type:'outer',color:'000000',opacity:0.28,blur:2,angle:45,distance:2}:undefined});}
function metric(slide,x,y,w,h,value,label,color=C.cyan){panel(slide,x,y,w,h,{fill:'09111F',line:C.line});txt(slide,String(value),x+0.12,y+0.08,w-0.24,h*0.46,{fontFace:'Georgia',fontSize:25,bold:true,color});txt(slide,label,x+0.12,y+h*0.52,w-0.24,h*0.32,{fontSize:9.5,color:C.muted});}
function flow(slide,items,x,y,w,h,accent=C.cyan){const gap=0.08;const bw=(w-gap*(items.length-1))/items.length;items.forEach((it,i)=>{const bx=x+i*(bw+gap);panel(slide,bx,y,bw,h,{fill:i%2?'10182B':'0B1425',line:i===items.length-1?C.green:C.line});txt(slide,String(i+1).padStart(2,'0'),bx+0.08,y+0.07,bw-0.16,0.2,{fontSize:7.8,bold:true,color:C.gold});txt(slide,it,bx+0.08,y+0.29,bw-0.16,h-0.34,{fontSize:10.5,bold:true,align:'center'});if(i<items.length-1){slide.addShape(S.chevron,{x:bx+bw-0.015,y:y+h/2-0.08,w:0.18,h:0.16,fill:{color:accent},line:{color:accent}})}})}
function bullets(slide,items,x,y,w,h,opt={}){const runs=[];items.forEach((it,i)=>{runs.push({text:it,options:{bullet:{indent:14},hanging:3,breakLine:true}})});slide.addText(runs,{x,y,w,h,fontFace:'Arial',fontSize:opt.fontSize||14,color:opt.color||C.white,breakLine:false,margin:0,paraSpaceAfterPt:8,fit:'shrink'});}
function bar(slide,label,value,max,x,y,w,color=C.cyan,sub=''){txt(slide,label,x,y,w*0.28,0.28,{fontSize:10,bold:true});slide.addShape(S.roundRect,{x:x+w*0.30,y:y+0.03,w:w*0.55,h:0.18,fill:{color:'26344A'},line:{color:'26344A'}});slide.addShape(S.roundRect,{x:x+w*0.30,y:y+0.03,w:w*0.55*Math.max(0,value/max),h:0.18,fill:{color},line:{color}});txt(slide,String(value),x+w*0.87,y,w*0.12,0.28,{fontSize:10,bold:true,color,align:'right'});if(sub)txt(slide,sub,x+w*0.30,y+0.25,w*0.65,0.2,{fontSize:7.5,color:C.muted});}
function note(slide,text,x,y,w,h,color=C.gold){slide.addShape(S.roundRect,{x,y,w,h,fill:{color,transparency:90},line:{color,transparency:52,width:0.8},rectRadius:0.16});txt(slide,text,x+0.15,y+0.08,w-0.3,h-0.16,{fontFace:'Georgia',fontSize:12,bold:true,color:C.white,align:'center'});}

// 1 Cover
{
 const s=slide();
 s.addShape(S.ellipse,{x:8.6,y:0.8,w:4.1,h:4.1,fill:{color:C.violet,transparency:80},line:{color:C.violet,transparency:100}});
 s.addShape(S.ellipse,{x:9.4,y:1.4,w:2.5,h:2.5,fill:{color:C.cyan,transparency:82},line:{color:C.cyan,transparency:100}});
 s.addShape(S.ellipse,{x:10.1,y:2.0,w:1.2,h:1.2,fill:{color:C.gold,transparency:65},line:{color:C.gold,transparency:100}});
 txt(s,'GOALOS',0.65,0.65,3.1,0.45,{fontSize:20,bold:true,color:C.gold,charSpacing:2});
 txt(s,'Partner\nMASTERCLASS',0.65,1.25,8.3,2.2,{fontFace:'Georgia',fontSize:46,bold:true,color:C.white,margin:0,breakLine:true});
 txt(s,'SOVEREIGN EDITION',0.72,3.52,4.2,0.32,{fontSize:12,bold:true,color:C.cyan,charSpacing:2});
 txt(s,'The institution that makes intelligence earn authority.',0.72,4.03,8.1,0.7,{fontFace:'Georgia',fontSize:24,italic:true,color:'DDE7F8'});
 note(s,'Mission 1 must not merely be remembered. It must earn the right to make Mission 2 better.',0.72,5.05,8.2,0.72,C.gold);
 txt(s,'Vincent Boucher\nPresident, QUEBEC.AI & MONTREAL.AI',0.72,6.12,5.5,0.55,{fontSize:12,bold:true,color:C.white});
 pill(s,'18 MODULES · INTERACTIVE · RSI · REAL-TASK EVIDENCE',8.35,5.35,4.2,C.cyan);
}
// 2 Executive proposition
{
 const s=slide();title(s,'Executive proposition','AI capability is abundant. Institutional permission to rely on it is not.','GoalOS converts autonomous work into proof-carrying, replayable, rollback-ready institutional capability.');
 const cols=[['Models','Candidate output','Text, code, plans, hypotheses, tool proposals.'],['Agents','Bounded action','Tools, environments, search, execution.'],['GoalOS','Earned authority','Proof, validation, memory, settlement, challenge, rollback.'],['Institution','Durable capability','Scoped reuse, cryptographic state, measurable Mission 2 lift.']];
 cols.forEach((c,i)=>{panel(s,0.55+i*3.13,2.6,2.82,2.28,{fill:i===2?'12243B':'0C1324',line:i===2?C.cyan:C.line});pill(s,c[0].toUpperCase(),0.72+i*3.13,2.78,1.35,i===2?C.cyan:C.gold);txt(s,c[1],0.72+i*3.13,3.25,2.48,0.38,{fontFace:'Georgia',fontSize:21,bold:true});txt(s,c[2],0.72+i*3.13,3.78,2.45,0.72,{fontSize:12,color:C.muted});});
 note(s,'Discovery engines improve results. GoalOS governs the right to evolve.',2.1,5.45,9.1,0.74,C.violet);
}
// 3 Reader map
{
 const s=slide();title(s,'Role-adaptive experience','One institution. Seven partner lenses.','The masterclass changes its language, value thesis, entry path, and proof metric for the person in the room.');
 const roles=[['Board','Decision readiness'],['Chief AI','Capability transfer'],['Research','Reproduced advantage'],['Risk','Proof coverage'],['Infrastructure','Replay capacity'],['Validator','Accuracy + independence'],['Capital','Verified-work throughput']];
 roles.forEach((r,i)=>{const col=i<4?i: i-4;const row=i<4?0:1;const x=0.62+col*3.08,y=2.45+row*1.58;panel(s,x,y,2.82,1.30,{fill:row?'10172A':'0C1324',line:i===0?C.gold:C.line});txt(s,r[0],x+0.16,y+0.16,2.5,0.32,{fontFace:'Georgia',fontSize:18,bold:true,color:i===0?C.gold:C.white});txt(s,r[1],x+0.16,y+0.65,2.45,0.28,{fontSize:11,color:C.muted});});
 note(s,'Bring one consequential objective, one executive sponsor, and one independent reviewer.',1.1,6.08,11.1,0.62,C.gold);
}
// 4 Category gap
{
 const s=slide();title(s,'The authority gap','What most AI platforms still cannot answer.','The missing layer is not generation. It is an evidentiary constitution for reliance.');
 const q=['What was authorized?','What was attempted?','What proof exists?','Who replayed it?','What failed?','What remains risky?','What may be reused?','How is it revoked?'];
 q.forEach((t,i)=>{const x=0.65+(i%4)*3.08,y=2.35+Math.floor(i/4)*1.25;panel(s,x,y,2.78,0.98,{fill:i<4?'0D1628':'10172A',line:i===6?C.green:C.line});txt(s,t,x+0.14,y+0.18,2.5,0.55,{fontSize:15,bold:true,align:'center'});});
 note(s,'A prompt requests an answer. A proof loop decides what must be proven before the answer can matter.',1.0,5.65,11.3,0.76,C.gold);
}
// 5 Proof loop
{
 const s=slide();title(s,'Canonical operating loop','Objective → proof → governed capability → better future mission.','Authority advances only when evidence survives the next gate.');
 flow(s,['Objective','Proof Mission','AGI Jobs','Evidence','Chronicle','Validated Skill','Graph Root','Mission 2'],0.55,2.5,12.25,1.15,C.cyan);
 panel(s,1.5,4.25,2.1,0.75,{fill:'26131D',line:C.red});txt(s,'Proof Debt',1.7,4.44,1.7,0.24,{fontSize:16,bold:true,color:C.red,align:'center'});
 s.addShape(S.line,{x:2.55,y:4.22,w:0,h:-0.5,line:{color:C.red,width:1.2,dash:'dash'}});
 s.addShape(S.line,{x:2.55,y:5.02,w:7.7,h:0,line:{color:C.gold,width:1.2,dash:'dash',beginArrowType:'none',endArrowType:'triangle'}});
 txt(s,'repair queue / blocked claim / fresh proof',3.15,4.76,5.5,0.24,{fontSize:9.5,color:C.gold,italic:true,align:'center'});
 note(s,'No Chronicle entry, no future-mission influence.',3.3,5.55,6.8,0.62,C.violet);
}
// 6 Authority stack
{
 const s=slide();title(s,'Authority stack','Desire, work, evidence, memory, and commitment remain separate.','No lower layer may silently bypass the gate above it.');
 const stack=[['01 Objective','Intent, decision, success, failure'],['02 Mission','Scope, proof level, budget, risk, rollback'],['03 Proof Work','Claims, Proof Debt, custom jobs'],['04 Evidence','Bundles, provenance, replay, cost, risk'],['05 Memory','Chronicle, scope, version, utility'],['06 Commitment','Roots, signatures, ZK receipts, lineage']];
 stack.forEach((r,i)=>{const y=2.15+i*0.68;panel(s,1.2,y,10.9,0.54,{fill:i%2?'10172A':'0C1324',line:i===4?C.green:i===5?C.gold:C.line});txt(s,r[0],1.4,y+0.09,2.2,0.28,{fontSize:12,bold:true,color:i===4?C.green:i===5?C.gold:C.white});txt(s,r[1],3.25,y+0.09,8.35,0.28,{fontSize:12,color:C.muted});});
}
// 7 Mission OS
{
 const s=slide();title(s,'Mission OS','Set the objective. GoalOS runs until proof is done.','The flagship deliverable is a Governed Decision State—not a static report.');
 const items=['Mission Contract','Claims matrix','Source provenance','Contradiction register','Evidence Docket','Verifier report','Risk ledger','Executive brief','Decision deck','Action graph','Chronicle entry','Capability package'];
 items.forEach((t,i)=>{const x=0.65+(i%4)*3.08,y=2.25+Math.floor(i/4)*1.05;panel(s,x,y,2.78,0.82,{fill:i===4?'10263A':'0C1324',line:i===4?C.cyan:C.line});txt(s,t,x+0.12,y+0.17,2.54,0.42,{fontSize:13,bold:true,align:'center',color:i===4?C.cyan:C.white});});
 note(s,'The shortest path from uncertainty to justified action.',2.2,5.82,8.95,0.62,C.gold);
}
// 8 Mission Contract
{
 const s=slide();title(s,'Mission Contract','Freeze the institutional boundary before agents optimize the work.','Objective, proof level, risk, validators, blocked claims, and rollback become explicit.');
 panel(s,0.65,2.25,5.55,3.95,{fill:'07111F',line:C.cyan});
 const code=`{\n  "objective": "Authorize a bounded AI deployment?",\n  "proof_level": "L3 External Review",\n  "risk_class": "High",\n  "required_validators": [\n    "independent reviewer", "AGI Node council"\n  ],\n  "blocked_claims": [\n    "guaranteed ROI", "unrestricted autonomy"\n  ],\n  "rollback": ["quarantine", "revoke on root mismatch"]\n}`;
 txt(s,code,0.9,2.48,5.0,3.38,{fontFace:'Courier New',fontSize:11.5,color:'DDFBFF',valign:'top'});
 panel(s,6.55,2.25,6.1,3.95,{fill:'0C1324',line:C.line});
 bullets(s,['Prevents ambiguous optimization','Makes success and failure falsifiable','Calibrates validator burden','Protects private intelligence','Blocks claims before proof','Defines Mission 2 metric','Makes rollback mandatory'],6.85,2.62,5.45,2.9,{fontSize:15});
}
// 9 Proof Debt & Jobs
{
 const s=slide();title(s,'Proof factory','Unsupported claims become explicit work—not hidden assumptions.','The job factory starts empty and creates only mission-specific proof contracts.');
 const jobs=[['Baseline Analyst','Counterfactual + uncertainty'],['Safety Sentinel','Failure modes + rollback'],['Evidence Auditor','Source reality + variance'],['Boundary Reviewer','Data + role + jurisdiction'],['Economics Analyst','Cost + sensitivity'],['Transfer Evaluator','Fresh Mission 2 comparison']];
 jobs.forEach((j,i)=>{const x=0.65+(i%3)*4.12,y=2.3+Math.floor(i/3)*1.42;panel(s,x,y,3.78,1.12,{fill:'0C1324',line:i===5?C.green:C.line});pill(s,String(i+1).padStart(2,'0'),x+0.14,y+0.12,0.62,C.gold);txt(s,j[0],x+0.9,y+0.12,2.65,0.28,{fontSize:14,bold:true});txt(s,j[1],x+0.9,y+0.52,2.65,0.28,{fontSize:10.5,color:C.muted});});
 note(s,'An AGI Job is a proof contract—not a todo item.',2.5,5.55,8.3,0.62,C.violet);
}
// 10 Evidence docket
{
 const s=slide();title(s,'Evidence Docket','A proof room that lets a reviewer reconstruct the mission.','What happened, what failed, what remains blocked, who can replay, and what may be reused.');
 const metrics=[['24','artifacts',C.cyan],['8','supported claims',C.green],['4','blocked claims',C.red],['3','contradictions',C.gold],['6','replay manifests',C.violet],['2','rollback paths',C.amber]];
 metrics.forEach((m,i)=>metric(s,0.65+(i%3)*4.12,2.25+Math.floor(i/3)*1.35,3.78,1.1,m[0],m[1],m[2]));
 note(s,'Evidence must become inspectable before it becomes authority.',2.1,5.45,9.2,0.65,C.gold);
}
// 11 AGI Nodes
{
 const s=slide();title(s,'AGI Node validation council','Independent commit-reveal validation at global scale.','Nominal identities do not count as independent when one operator, model, cloud, or key controls them.');
 const nodes=[['Montreal','Independent assurance','Model-A'],['London','Research council','Model-B'],['Frankfurt','Security validator','Model-C'],['Singapore','Infrastructure partner','Model-D'],['Tokyo','External replay lab','Model-E']];
 nodes.forEach((n,i)=>{const x=0.55+i*2.55;panel(s,x,2.45,2.28,2.35,{fill:'0C1324',line:i===4?C.green:C.line});pill(s,n[0].toUpperCase(),x+0.15,2.65,1.05,i===4?C.green:C.cyan);txt(s,n[1],x+0.15,3.16,1.98,0.55,{fontSize:13,bold:true,align:'center'});txt(s,n[2],x+0.15,3.83,1.98,0.28,{fontSize:10,color:C.muted,align:'center'});txt(s,'COMMIT → REVEAL',x+0.15,4.28,1.98,0.24,{fontSize:8.5,bold:true,color:C.gold,align:'center'});});
 note(s,'Five labels ≠ five independent validators.',3.5,5.5,6.3,0.62,C.red);
}
// 12 Chronicle
{
 const s=slide();title(s,'Chronicle','The memory firewall between evidence and durable capability.','Admit, admit with scope, repair, reject, quarantine, supersede, revoke, or retire.');
 flow(s,['Candidate','Evidence Docket','Chronicle Gate','Validated Skill','Future Mission'],0.8,2.4,11.75,1.1,C.green);
 const gates=[['Evidence','strength threshold'],['Replay','reconstruction ready'],['Validation','independent quorum'],['Risk','within envelope'],['Boundary','scope + privacy + rollback']];
 gates.forEach((g,i)=>{panel(s,0.75+i*2.5,4.1,2.22,1.05,{fill:'0C1324',line:i===4?C.green:C.line});txt(s,g[0],0.9+i*2.5,4.28,1.9,0.25,{fontSize:14,bold:true,color:i===4?C.green:C.white,align:'center'});txt(s,g[1],0.9+i*2.5,4.66,1.9,0.22,{fontSize:9.5,color:C.muted,align:'center'});});
 note(s,'No Chronicle entry, no future-mission influence.',3.0,5.65,7.35,0.62,C.gold);
}
// 13 VSG
{
 const s=slide();title(s,'Validated Skill Graph','Governed capability memory—not a prompt library.','A durable skill carries scope, method, evidence, tests, validators, risk, version, freshness, utility, and rollback.');
 panel(s,4.65,2.35,4.0,1.25,{fill:'14223A',line:C.green});txt(s,'VALIDATED SKILL',4.95,2.7,3.4,0.34,{fontFace:'Georgia',fontSize:23,bold:true,color:C.green,align:'center'});
 const leaves=[['Identity + scope',0.7,2.2],['Method + replay',0.7,4.2],['Evidence + validators',9.65,2.2],['Policy + lifecycle',9.65,4.2],['Utility + lineage',4.65,4.65]];
 leaves.forEach((l,i)=>{panel(s,l[1],l[2],3.0,1.0,{fill:'0C1324',line:i===4?C.gold:C.line});txt(s,l[0],l[1]+0.16,l[2]+0.22,2.68,0.38,{fontSize:15,bold:true,align:'center'});s.addShape(S.line,{x:l[1]<4?l[1]+3:l[1],y:l[2]+0.5,w:l[1]<4?1.0:-1.0,h:(l[2]<3?0.55:-0.55),line:{color:C.violet,width:1.2,endArrowType:'triangle'}})});
}
// 14 Merkle privacy ZK
{
 const s=slide();title(s,'Merkle, privacy, and ZK','Commit exact institutional state without publishing plaintext private intelligence.','Chronicle decides authority. Merkle roots preserve the state that earned it.');
 const layers=[['Private capability','methods, traces, reviewer notes',C.violet],['Proof artifacts','dockets, bundles, attestations',C.cyan],['Commitment state','typed Merkle forest',C.gold],['Authority state','Chronicle decision',C.green],['Future mission','bounded prior',C.amber]];
 layers.forEach((l,i)=>{const x=0.55+i*2.55;panel(s,x,2.45,2.27,1.55,{fill:'0C1324',line:l[2]});txt(s,l[0],x+0.12,2.7,2.03,0.36,{fontSize:13.5,bold:true,color:l[2],align:'center'});txt(s,l[1],x+0.15,3.27,1.97,0.38,{fontSize:9.5,color:C.muted,align:'center'});if(i<4)s.addShape(S.chevron,{x:x+2.21,y:3.03,w:0.28,h:0.20,fill:{color:l[2]},line:{color:l[2]}})});
 note(s,'Inclusion is not truth. A root is a commitment layer; Chronicle is the authority layer.',1.3,5.1,10.7,0.74,C.red);
}
// 15 Bonded authority
{
 const s=slide();title(s,'Bonded authority','$AGIALPHA becomes professional protocol security behind consequential transitions.','Nearly invisible user friction; capital at risk whenever work asks to become paid, trusted, challenged, reusable, or promoted.');
 const rows=[['Begin work','Worker Proof Bond'],['Become paid','ProofBundle + Validator Bonds'],['Become trusted','Claim Authority Bond'],['Open challenge','Challenger Bond'],['Become reusable','Capability Reuse Bond'],['Operate node','Node Operational Reserve'],['Upgrade GoalOS','Proposal + Canary + Rollback']];
 rows.forEach((r,i)=>{const y=2.15+i*0.56;txt(s,r[0],0.85,y,3.15,0.28,{fontSize:12.5,bold:true});s.addShape(S.line,{x:4.0,y:y+0.14,w:1.15,h:0,line:{color:C.gold,width:1.2,endArrowType:'triangle'}});txt(s,r[1],5.35,y,6.6,0.28,{fontSize:12.5,color:i===4?C.green:C.white,bold:i===4});});
 pill(s,'0% PROTOCOL TRANSFER TAX',8.4,5.95,2.05,C.green);pill(s,'$0 TREASURY TARGET',10.58,5.95,1.8,C.cyan);
}
// 16 RSI pipeline
{
 const s=slide();title(s,'RSI sovereign invention lab','Recursive self-improvement as deterministic, auditable institutional state transition.','Artifacts are schema-bound, baselines are mandatory, ledgers are append-only, and promotion is mechanically gated.');
 flow(s,['TARGET','EMIT','FILTER','ATLAS','TEST-PLAN','EVAL','INSERT','PROMOTE'],0.55,2.55,12.25,1.12,C.violet);
 const guards=[['RISK','Prohibited domains + safety gate',C.red],['EVIDENCE','Confidence cannot inflate without execution',C.cyan],['BASELINE','Incumbent + neighbor + null',C.gold],['PERSISTENCE','Advantage under shocks',C.green]];
 guards.forEach((g,i)=>{panel(s,0.65+i*3.08,4.3,2.78,1.38,{fill:'0C1324',line:g[2]});txt(s,g[0],0.82+i*3.08,4.5,2.44,0.28,{fontSize:13,bold:true,color:g[2],align:'center'});txt(s,g[1],0.82+i*3.08,4.96,2.44,0.38,{fontSize:10,color:C.muted,align:'center'});});
}
// 17 Search control
{
 const s=slide();title(s,'Search control ≠ outcome authority','Exploration can be open-ended. Promotion cannot.','OMNI or any search controller may allocate attention and compute; it never grants memory, settlement, or rollout authority.');
 panel(s,0.7,2.35,5.85,3.45,{fill:'111027',line:C.violet});txt(s,'SEARCH / ALLOCATION',1.0,2.65,5.2,0.4,{fontFace:'Georgia',fontSize:24,bold:true,color:C.violet,align:'center'});bullets(s,['Interestingness','Novelty','Diversity','Stepping stones','Resource allocation','Candidate generation'],1.15,3.35,4.95,1.8,{fontSize:16});
 panel(s,6.78,2.35,5.85,3.45,{fill:'0E1D23',line:C.green});txt(s,'OUTCOME AUTHORITY',7.08,2.65,5.2,0.4,{fontFace:'Georgia',fontSize:24,bold:true,color:C.green,align:'center'});bullets(s,['Risk gate','Evidence contact','Baseline advantage','Reproduction','Stress + persistence','Scope + rollback'],7.25,3.35,4.95,1.8,{fontSize:16});
 note(s,'A high score cannot bypass a hard gate.',4.05,6.0,5.2,0.56,C.red);
}
// 18 Move 37
{
 const s=slide();title(s,'Move-37 breakthrough control','High novelty creates a higher burden of proof.','Breakthroughs are admitted as audited state transitions—not as narratives.');
 const steps=[['Recognize','Thresholds + risk + ECI'],['Reproduce','Fixed seeds + baseline + hashes'],['Stress-test','Policy shocks + sensitivity'],['Persistence','Advantage remains positive'],['Dossier','Scope + evidence + rollback']];
 steps.forEach((st,i)=>{const y=2.25+i*0.76;panel(s,0.72,y,7.0,0.58,{fill:'0C1324',line:i===4?C.green:C.line});txt(s,String(i+1),0.88,y+0.12,0.35,0.25,{fontSize:11,bold:true,color:C.gold});txt(s,st[0],1.35,y+0.09,1.5,0.28,{fontSize:13,bold:true});txt(s,st[1],2.85,y+0.09,4.5,0.28,{fontSize:11,color:C.muted});});
 panel(s,8.15,2.25,4.45,3.8,{fill:'140F2C',line:C.violet});txt(s,'HIGH NOVELTY\n⇒\nHIGHER SKEPTICISM',8.55,2.75,3.65,1.2,{fontFace:'Georgia',fontSize:27,bold:true,color:C.violet,align:'center'});txt(s,'No probe, no breakthrough claim.\nNo persistence, no promotion.',8.65,4.45,3.45,0.65,{fontSize:14,bold:true,color:C.white,align:'center'});
}
// 19 Self hosting
{
 const s=slide();title(s,'Constitutional self-hosting','GoalOS may improve GoalOS only through GoalOS.','No privileged developer exception. Every constitutional change needs evidence, canaries, monitoring, and rollback.');
 flow(s,['Upgrade Proposal','Baseline','Candidate','Invariant Tests','Council','1% Canary','25% Canary','Promote / Rollback'],0.55,2.55,12.25,1.12,C.gold);
 bullets(s,['Proposal bond','Evaluator independence','Constitutional invariants','Delayed outcomes','Rollback target','New graph epoch'],1.15,4.2,5.1,1.35,{fontSize:16});
 note(s,'No proof, no evolution. No eval, no propagation. No rollback, no release.',6.2,4.45,5.6,0.9,C.red);
}
// 20 decisive test
{
 const s=slide();title(s,'The decisive Mission 1 → Mission 2 test','Mission 1 must make fresh Mission 2 work measurably better.','Equal model, tools, time, compute, evaluator, stopping rule, and context budget. Only the prior differs.');
 const arms=[['Fresh control','No prior capability',C.muted],['Raw memory','Transcript / notes',C.gold],['Validated Skill','Chronicle-admitted capability',C.green],['Ungated candidate','Rejected or unreviewed memory',C.red]];
 arms.forEach((a,i)=>{panel(s,0.65+i*3.08,2.35,2.78,2.0,{fill:'0C1324',line:a[2]});txt(s,a[0],0.82+i*3.08,2.62,2.44,0.42,{fontFace:'Georgia',fontSize:18,bold:true,color:a[2],align:'center'});txt(s,a[1],0.9+i*3.08,3.35,2.28,0.48,{fontSize:11,color:C.muted,align:'center'});});
 note(s,'Validated Skill > Raw Memory > Fresh Control; ungated memory must not silently propagate.',1.0,5.15,11.3,0.74,C.gold);
}
// 21 transfer result
{
 const s=slide();title(s,'Interactive transfer result','The mechanism passes inside the partner simulation.','A rejected prior harms performance; tampering revokes inherited authority.');
 bar(s,'Fresh Control',74,100,0.9,2.4,7.4,C.muted);bar(s,'Raw Memory',79,100,0.9,3.05,7.4,C.gold);bar(s,'Validated Skill',92,100,0.9,3.70,7.4,C.green);bar(s,'Ungated Candidate',61,100,0.9,4.35,7.4,C.red);
 metric(s,9.0,2.45,3.3,1.15,'+18','points vs fresh',C.green);metric(s,9.0,3.85,3.3,1.15,'38%','less proof-design time',C.cyan);note(s,'External independent replay remains a separate gate.',8.9,5.45,3.55,0.62,C.red);
}
// 22 REAL001
{
 const s=slide();title(s,'REAL-001 evidence room','A consequential public software-maintenance transfer case.','A validated repository-reliability skill reached the correct safe decision with less evidence-search effort.');
 bar(s,'Fresh Control',92.86,100,0.8,2.35,7.2,C.muted);bar(s,'Raw Memory',92.86,100,0.8,3.0,7.2,C.gold);bar(s,'Validated Skill',98.57,100,0.8,3.65,7.2,C.green);bar(s,'Ungated Candidate',0,100,0.8,4.30,7.2,C.red,'actual score: -15.00');
 panel(s,8.65,2.25,3.95,2.75,{fill:'0E1D23',line:C.green});txt(s,'CORRECT SAFE DECISION',8.9,2.56,3.45,0.28,{fontSize:10,bold:true,color:C.green,align:'center'});txt(s,'CONTROLLED\nMIGRATION\nREQUIRED',8.92,3.05,3.4,1.2,{fontFace:'Georgia',fontSize:25,bold:true,align:'center'});
 pill(s,'FIRST-PARTY RECONSTRUCTION PASS',8.85,5.3,3.6,C.cyan);pill(s,'EXTERNAL REPLAY PENDING',8.85,5.75,3.6,C.gold);
}
// 23 Value
{
 const s=slide();title(s,'Partner value architecture','Reduce proof debt, failure exposure, review latency, and reinvention.','Increase decision readiness, validator confidence, and reusable governed capability.');
 metric(s,0.65,2.25,2.9,1.25,'$30.7M','illustrative annual Proof Debt exposure',C.red);metric(s,3.8,2.25,2.9,1.25,'$9.2M','scenario expected-loss reduction',C.green);metric(s,6.95,2.25,2.9,1.25,'242','decision-cycle days released',C.cyan);metric(s,10.1,2.25,2.55,1.25,'17','missions receiving validated capability',C.gold);
 const value=['Decision impact','Proof integrity','Actionability','Reuse'];const costs=['Cost','Risk','Latency','Proof Debt'];
 txt(s,'Mission Value =',1.2,4.25,2.0,0.4,{fontFace:'Georgia',fontSize:23,bold:true,color:C.gold});txt(s,value.join(' × '),3.05,4.25,8.0,0.4,{fontFace:'Georgia',fontSize:20,bold:true,color:C.white,align:'center'});txt(s,'÷',11.0,4.25,0.45,0.4,{fontSize:22,bold:true,color:C.gold,align:'center'});txt(s,costs.join(' + '),3.05,5.0,8.0,0.4,{fontFace:'Georgia',fontSize:18,color:C.muted,align:'center'});
 note(s,'Scenario model—not a forecast, valuation guarantee, or financial recommendation.',2.05,5.75,9.2,0.58,C.red);
}
// 24 Paths
{
 const s=slide();title(s,'Partnership pathways','Start with a proof mission. Compound into infrastructure.','Each pathway creates evidence before expanding authority.');
 const paths=[['Proof Mission Sprint','One consequential objective → Governed Decision State'],['Enterprise Proof OS','Install proof levels, Chronicle, replay, rollback, and skill memory'],['Private VSG','Govern reusable capability across sensitive institutional work'],['Managed AGI Node Council','Independent validation, replay, identity, and assurance'],['Cross-Institution Proof Network','Portable proof-bearing capability without plaintext leakage']];
 paths.forEach((p,i)=>{const y=2.1+i*0.78;panel(s,0.75,y,11.8,0.58,{fill:i===0?'16233A':'0C1324',line:i===0?C.gold:C.line});txt(s,String(i+1).padStart(2,'0'),0.95,y+0.13,0.5,0.24,{fontSize:10,bold:true,color:C.gold});txt(s,p[0],1.55,y+0.09,3.1,0.28,{fontSize:13,bold:true});txt(s,p[1],4.75,y+0.09,7.35,0.28,{fontSize:11.5,color:C.muted});});
}
// 25 Integration map
{
 const s=slide();title(s,'Partner integration architecture','GoalOS sits above existing capability and below institutional consequence.','No one model, agent, or vendor needs to become the source of institutional truth.');
 const layers=[['Existing capability','models, agents, tools, experts, enterprise systems'],['Mission OS','intent, scope, proof level, risk, validators, rollback'],['Proof fabric','AGI Jobs, ProofBundles, Evidence Dockets'],['Validation','AGI Nodes, replay, challenge, independence'],['Memory + commitment','Chronicle, VSG, Merkle, ZK, revocation'],['Operations + economics','bonded authority, settlement adapters, pilot metrics']];
 layers.forEach((l,i)=>{const y=1.95+i*0.74;panel(s,1.05,y,11.15,0.57,{fill:i===1?'10263A':i===4?'13221F':'0C1324',line:i===1?C.cyan:i===4?C.green:C.line});txt(s,l[0],1.25,y+0.11,2.55,0.28,{fontSize:12.5,bold:true,color:i===1?C.cyan:i===4?C.green:C.white});txt(s,l[1],3.9,y+0.11,7.95,0.28,{fontSize:11,color:C.muted});});
}
// 26 pilot
{
 const s=slide();title(s,'Thirty-day partner pilot','One objective. One sponsor. One independent reviewer. One Mission 2 test.','The pilot earns the right to expand; it does not predeclare success.');
 const weeks=[['WEEK 1','Freeze authority','Mission Contract, proof level, claim boundary, baseline, privacy, rollback.'],['WEEK 2','Produce proof','Custom jobs, bundles, provenance, contradictions, cost and risk.'],['WEEK 3','Validate + challenge','Independent replay, AGI Nodes, falsification, Chronicle recommendation.'],['WEEK 4','Deliver + transfer','Decision State, Skill Passport, Merkle packet, Mission 2 comparison.']];
 weeks.forEach((w,i)=>{panel(s,0.55+i*3.2,2.3,2.92,3.15,{fill:'0C1324',line:i===3?C.green:C.line});pill(s,w[0],0.75+i*3.2,2.55,0.95,i===3?C.green:C.gold);txt(s,w[1],0.75+i*3.2,3.08,2.52,0.5,{fontFace:'Georgia',fontSize:19,bold:true,align:'center'});txt(s,w[2],0.78+i*3.2,3.85,2.46,1.1,{fontSize:11,color:C.muted,align:'center'});});
 note(s,'The pilot’s final deliverable is a measured authority transition—not a presentation.',2.1,5.85,9.2,0.58,C.gold);
}
// 27 diligence
{
 const s=slide();title(s,'Trust, due diligence, and claim boundary','Sophisticated partners trust systems that state exactly what is evidenced and what remains open.','Evidence maturity is part of the product.');
 const rows=[['Mission Contract','Evidenced'],['Evidence Docket','Evidenced'],['Independent validator quorum','Demonstrated in simulation'],['Chronicle + Merkle tamper revocation','Demonstrated'],['REAL-001 first-party reconstruction','Pass'],['Independent external replay','Pending'],['General real-world compounding','Unproven'],['Production authority','Not granted']];
 rows.forEach((r,i)=>{const y=2.05+i*0.48;txt(s,r[0],0.75,y,6.1,0.26,{fontSize:11.5,bold:true});pill(s,r[1].toUpperCase(),8.3,y-0.02,3.65,r[1].includes('Pending')||r[1].includes('Unproven')?C.gold:r[1].includes('Not')?C.red:C.green);});
 note(s,'No label overrides facts. No token creates immunity. No code overrides mandatory law.',1.55,6.1,10.2,0.55,C.red);
}
// 28 close
{
 const s=slide();
 txt(s,'THE PARTNER ASK',0.65,0.75,3.5,0.32,{fontSize:11,bold:true,color:C.gold,charSpacing:2});
 txt(s,'Bring one consequential objective.',0.65,1.35,11.7,0.78,{fontFace:'Georgia',fontSize:39,bold:true,color:C.white});
 txt(s,'Leave with proof — and the first capability that has earned the right to make the next mission better.',0.65,2.28,11.6,1.1,{fontFace:'Georgia',fontSize:28,italic:true,color:'DCE8FA'});
 flow(s,['Objective','Proof Mission','Evidence Docket','Chronicle','Validated Skill','Mission 2'],1.0,4.0,11.3,1.0,C.gold);
 note(s,'AI creates output. GoalOS creates proof. The institution decides what earns memory.',1.4,5.6,10.5,0.72,C.gold);
 txt(s,'Vincent Boucher · President, QUEBEC.AI & MONTREAL.AI',0.72,6.55,8.5,0.3,{fontSize:13,bold:true,color:C.white});
}

pptx.writeFile({fileName:'/mnt/data/goalos-partner-masterclass-sovereign-v2/presentation/goalos-partner-masterclass-sovereign-v2-deck.pptx'});
