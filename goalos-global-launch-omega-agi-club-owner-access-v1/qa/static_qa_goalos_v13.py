import json,re,hashlib,subprocess,tempfile,time,sys
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader
ROOT=Path('/mnt/data/GoalOS_Global_Launch_Omega_AGI_CLUB_GITHUB_PAGES_FINAL_v13_0_0_GL1_2026-07-24/goalos-global-launch-omega-agi-club-owner-access')
INDEX=ROOT/'index.html'
V12=ROOT/'GoalOS_Global_Launch_Omega_AGI_CLUB_GITHUB_PAGES_v12_0_0_GL1.html'
OUT=Path('/mnt/data/goalos_v13_release_assets/qa');OUT.mkdir(parents=True,exist_ok=True)
controls=[]
def ck(name,cond,detail=''):
    controls.append({'name':name,'passed':bool(cond),'detail':detail})
    print(('PASS' if cond else 'FAIL'),name,detail,flush=True)
text=INDEX.read_text('utf-8',errors='replace')
soup=BeautifulSoup(text,'html.parser')
ids=[x.get('id') for x in soup.find_all(attrs={'id':True})]
dups=sorted({x for x in ids if ids.count(x)>1})
ck('Canonical index exists',INDEX.is_file(),INDEX.stat().st_size)
ck('Unique DOM identifiers',not dups,dups[:10])
ck('Canonical product title','GoalOS Global Launch Ω' in (soup.title.get_text() if soup.title else ''))
ck('Version 13 release marker','v13.0.0-GL1' in text)
ck('Apex Institution Formation edition marker','Apex Institution Formation' in text)
required_sections=['top','institution-formation','goalos-chat','mission-contract','global-launch','apex-council','launch-missions','proof-governance','graph-governance','expansion-office','research-proof','omega-prime','legal-center']
for sid in required_sections: ck('Integrated section '+sid,soup.find(id=sid) is not None)
ck('Premium pre-access preview',soup.select_one('.v13-preview-copy h1') is not None)
ck('Six premium preview metrics',len(soup.select('.v13-preview-stats>div'))==6,len(soup.select('.v13-preview-stats>div')))
ck('Four institutional preview cards',len(soup.select('.v13-preview-capabilities article'))==4,len(soup.select('.v13-preview-capabilities article')))
ck('Main flagship hero',soup.select_one('.v13-main-hero') is not None)
ck('Four live institution cards',len(soup.select('.v13-live-card'))==4,len(soup.select('.v13-live-card')))
ck('Apex deliberation controls',all(soup.find(id=x) is not None for x in ['v13SearchBudget','v13ObjectiveLens','v13RunDeliberation','v13FalsifyChampion','v13DeliberationState','v13DeliberationChampion']))
# Embedded data
omega=soup.find(id='omega-data')
raw=(omega.string or omega.get_text()).strip();data=json.loads(raw)
ck('70 jurisdiction nodes',len(data.get('jurisdictions',[]))==70,len(data.get('jurisdictions',[])))
ck('463 governed routes',len(data.get('instruments',[]))==463,len(data.get('instruments',[])))
ck('476 official and mapped sources',len(data.get('sources',[]))==476,len(data.get('sources',[])))
ck('20 controlled transaction categories',len(data.get('transactions',[]))==20,len(data.get('transactions',[])))
# External dependencies
script_src=[x.get('src') for x in soup.find_all('script',src=True)]
style_href=[x.get('href') for x in soup.find_all('link',href=True) if 'stylesheet' in (x.get('rel') or [])]
ck('No external script dependencies',not script_src,script_src)
ck('No external stylesheet dependencies',not style_href,style_href)
ck('No server API dependency','/api/access/' not in text)
ck('Official Ethereum Mainnet ENS Name Wrapper present','0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401'.lower() in text.lower())
ck('Exact direct club name rule present','club.agi.eth' in text and 'exact' in text.lower())
# JS syntax
js_fail=[];js_count=0
with tempfile.TemporaryDirectory() as td:
    for i,x in enumerate(soup.find_all('script')):
        if x.get('src'): continue
        typ=(x.get('type') or '').lower()
        if typ and typ not in ('text/javascript','application/javascript','module'): continue
        code=x.string if x.string is not None else x.get_text()
        if not code.strip(): continue
        p=Path(td)/f'{i:02d}.js';p.write_text(code,'utf-8');js_count+=1
        r=subprocess.run(['node','--check',str(p)],capture_output=True,text=True)
        if r.returncode: js_fail.append({'script':i,'error':r.stderr[-1200:]})
ck('All executable inline scripts pass syntax',not js_fail,{'count':js_count,'failures':js_fail})
# Local links
html_files=list(ROOT.rglob('*.html'));unresolved=[];checked=0
for hp in html_files:
    hs=BeautifulSoup(hp.read_text('utf-8',errors='replace'),'html.parser')
    for tag,attr in [('a','href'),('img','src'),('script','src'),('link','href')]:
        for n in hs.find_all(tag):
            v=(n.get(attr) or '').strip()
            if not v or v.startswith(('#','http://','https://','mailto:','tel:','data:','blob:','javascript:','ipfs:','ens:')): continue
            v=v.split('#',1)[0].split('?',1)[0]
            if not v: continue
            target=(hp.parent/v).resolve();checked+=1
            try: target.relative_to(ROOT.resolve())
            except ValueError: unresolved.append((str(hp.relative_to(ROOT)),v,'escapes root'));continue
            if not target.exists(): unresolved.append((str(hp.relative_to(ROOT)),v,'missing'))
ck('All packaged local links resolve',not unresolved,{'checked':checked,'errors':unresolved[:20]})
# File size
overs=[(str(p.relative_to(ROOT)),p.stat().st_size) for p in ROOT.rglob('*') if p.is_file() and p.stat().st_size>=25*1024*1024]
ck('Every publication file below 25 MB',not overs,overs)
# Paper and research
paper=ROOT/'research/GoalOS_Global_Launch_Omega_v13_Apex_Institution_Formation_Edition.pdf'
orig=ROOT/'research/GoalOS_Global_Launch_Omega_The_Verified_Operating_System_for_Building_Anything_Anywhere.pdf'
latex=ROOT/'research/GoalOS_Global_Launch_Omega_LaTeX_Paper_COMPLETE_v1_0_0.zip'
feedback=ROOT/'research/CATEGORY_AND_PRODUCT_CONSTITUTION_FEEDBACK.md'
addendum=ROOT/'research/GoalOS_Global_Launch_Omega_v13_Apex_Search_Addendum.md'
ck('v13 Apex paper packaged',paper.is_file() and paper.stat().st_size>400000,paper.stat().st_size if paper.exists() else 0)
pages=len(PdfReader(str(paper)).pages) if paper.exists() else 0
ck('v13 Apex paper is 22 pages',pages==22,pages)
ck('Original 20-page paper preserved',orig.is_file() and len(PdfReader(str(orig)).pages)==20,orig.stat().st_size if orig.exists() else 0)
ck('LaTeX paper source preserved',latex.is_file() and latex.stat().st_size>2_000_000,latex.stat().st_size if latex.exists() else 0)
ck('Category/product feedback preserved',feedback.is_file() and feedback.stat().st_size>20_000,feedback.stat().st_size if feedback.exists() else 0)
ck('Apex Search addendum packaged',addendum.is_file() and addendum.stat().st_size>1500,addendum.stat().st_size if addendum.exists() else 0)
ck('Premium paper cover packaged',(ROOT/'research/paper_cover_v13.png').is_file())
ck('Apex Search paper preview packaged',(ROOT/'preview/GoalOS_Global_Launch_Omega_v13_Apex_Opportunity_Search.png').is_file())
# Doctrines
visible=soup.get_text(' ',strip=True).lower()
phrases=['proof before authority','evidence before memory','the category is not business formation','institution formation','verified launch value','proof debt','transaction law','runway law','succession law','final institutional law']
for phrase in phrases: ck('Doctrine integrated: '+phrase,phrase in visible)
ck('Launch-stack category promise integrated',('the world is your launch stack' in text.lower()) or ('turns the world into your launch stack' in text.lower()))
# Preservation
v12s=BeautifulSoup(V12.read_text('utf-8',errors='replace'),'html.parser')
v12_ids={x.get('id') for x in v12s.find_all(attrs={'id':True})};v13_ids=set(ids);missing=sorted(v12_ids-v13_ids)
ck('All v12 interface identifiers preserved',not missing,{'v12':len(v12_ids),'v13':len(v13_ids),'missing':missing[:20]})
# Copies synced
base=hashlib.sha256(INDEX.read_bytes()).hexdigest()
for cp in [ROOT/'404.html',ROOT/'GoalOS_Global_Launch_Omega_AGI_CLUB_GITHUB_PAGES_v13_0_0_GL1.html',Path('/mnt/data/GoalOS_Global_Launch_Omega_AGI_CLUB_GITHUB_PAGES_v13_0_0_GL1.html')]:
    ck(cp.name+' synchronized with index',cp.is_file() and hashlib.sha256(cp.read_bytes()).hexdigest()==base)
# Source refs
srcids={str(x.get('Source ID') or x.get('source_id') or '') for x in data.get('sources',[])}
missing_src=[]
for inst in data.get('instruments',[]):
    refs=inst.get('sourceIds') or inst.get('Source IDs') or []
    if isinstance(refs,str): refs=[x.strip() for x in re.split(r'[/,;]',refs) if x.strip()]
    for r in refs:
        if r and r not in srcids: missing_src.append((inst.get('Instrument ID') or inst.get('id'),r))
ck('Instrument source references resolve',not missing_src,missing_src[:20])
# Key governance assets
for rel in ['governance/LEGAL_INDEX.html','governance/AGI_CLUB_OWNER_ACCESS_LICENSE.html','governance/TRANSACTION_EVIDENCE_GATE.html','governance/NO_SECRETS_BEFORE_CONTRACT.html','GITHUB_PAGES_DEPLOYMENT_GUIDE.md','START_HERE.html','V13_ARCHITECTURE_IMPLEMENTATION.md']:
    ck('Required governance/release artifact '+rel,(ROOT/rel).is_file())
result={'release':'GoalOS Global Launch Ω v13.0.0-GL1','testedAt':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'passed':sum(x['passed'] for x in controls),'failed':sum(not x['passed'] for x in controls),'total':len(controls),'linkCount':checked,'htmlFiles':len(html_files),'v12Ids':len(v12_ids),'v13Ids':len(v13_ids),'controls':controls}
(OUT/'static_qa.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),'utf-8')
print(json.dumps(result,indent=2,ensure_ascii=False))
raise SystemExit(1 if result['failed'] else 0)
