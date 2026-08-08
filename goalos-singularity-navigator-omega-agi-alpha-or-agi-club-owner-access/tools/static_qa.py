from __future__ import annotations
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import unquote, urlsplit
import json, re, subprocess, tempfile, hashlib, os, sys

BASE=Path('/mnt/data/GoalOS_Singularity_Navigator_Omega_AGI_CLUB_GITHUB_PAGES_FINAL_v2_0_0_SN3_2026-07-26/goalos-singularity-navigator-omega-agi-club-owner-access')
STANDALONE=Path('/mnt/data/GoalOS_Singularity_Navigator_Omega_AGI_CLUB_GITHUB_PAGES_v2_0_0_SN3.html')
OLD=Path('/mnt/data/GoalOS_Singularity_Navigator_Omega_AGI_CLUB_GITHUB_PAGES_FINAL_v2_0_0_SN2_2026-07-26/goalos-singularity-navigator-omega-agi-club-owner-access/index.html')
results=[]
def check(name, ok, detail=None):
    results.append({'name':name,'passed':bool(ok),'detail':detail})

def refs_for(html_path):
    text=html_path.read_text(encoding='utf-8',errors='ignore')
    soup=BeautifulSoup(text,'html.parser')
    refs=[]
    for tag,attr in [('a','href'),('img','src'),('script','src'),('link','href'),('iframe','src'),('source','src')]:
        for e in soup.find_all(tag):
            v=e.get(attr)
            if v: refs.append((tag,attr,v))
    for style in soup.find_all('style'):
        for v in re.findall(r'url\((?:["\']?)([^"\')]+)',style.get_text()):
            refs.append(('style','url',v))
    return text,soup,refs

# structure and required files
required=['index.html','404.html','START_HERE.html','README.md','VERSION.json','STATIC_ACCESS_SECURITY_MODEL.md','GITHUB_PAGES_DEPLOYMENT_GUIDE.md','governance/LEGAL_INDEX.html','governance/AGI_CLUB_OWNER_ACCESS_LICENSE.html','governance/AGI_CLUB_ACCESS_CONTROL.html','governance/ACCESS_CONTROL_LIMITATIONS.html','schemas/agi_club_owner_access_receipt_v2.schema.json','research/GoalOS_Singularity_Navigation_Omega_Tools_to_Ride_the_Singularity_by_Vincent_Boucher_v2_0_0.pdf','assets/frontier_map.webp','manifest.webmanifest','sw.js','documentation/SN3_ADVANCEMENTS.md']
missing=[r for r in required if not (BASE/r).is_file()]
check('required publication files',not missing,missing)
check('standalone exists',STANDALONE.is_file(),STANDALONE.stat().st_size if STANDALONE.exists() else None)
check('all publication files below 25 MB',all(p.stat().st_size<25*1024*1024 for p in BASE.rglob('*') if p.is_file()),max((p.stat().st_size for p in BASE.rglob('*') if p.is_file()),default=0))
check('standalone below 25 MB',STANDALONE.stat().st_size<25*1024*1024,STANDALONE.stat().st_size)

# HTML files
all_html=list(BASE.rglob('*.html'))
check('HTML documents present',len(all_html)>=7,len(all_html))
local_checked=0; unresolved=[]; anchor_errors=[]; duplicate_ids=[]
external_scripts=[]; external_styles=[]
for f in all_html:
    text,soup,refs=refs_for(f)
    ids=[e.get('id') for e in soup.find_all(id=True)]
    dups=sorted({x for x in ids if ids.count(x)>1})
    if dups: duplicate_ids.append((str(f.relative_to(BASE)),dups))
    idset=set(ids)
    for tag,attr,raw in refs:
        val=raw.strip()
        if val.startswith(('data:','blob:','mailto:','tel:','javascript:')): continue
        if tag=='style' and '%23' in val: continue
        if val.startswith(('http://','https://')):
            if tag=='script': external_scripts.append(val)
            if tag=='link': external_styles.append(val)
            continue
        if val.startswith('#'):
            if val[1:] and val[1:] not in idset: anchor_errors.append((str(f.relative_to(BASE)),val))
            continue
        parts=urlsplit(val)
        rel=unquote(parts.path)
        if not rel: continue
        target=(f.parent/rel).resolve()
        local_checked+=1
        if not target.exists(): unresolved.append((str(f.relative_to(BASE)),val,str(target)))
        if parts.fragment and target.suffix.lower() in ('.html','.htm') and target.exists():
            ts=BeautifulSoup(target.read_text(encoding='utf-8',errors='ignore'),'html.parser')
            if not ts.find(id=parts.fragment): anchor_errors.append((str(f.relative_to(BASE)),val))
check('duplicate HTML IDs',not duplicate_ids,duplicate_ids)
check('local links and assets resolve',not unresolved,{'checked':local_checked,'errors':unresolved})
check('local anchors resolve',not anchor_errors,anchor_errors)
check('external JavaScript dependencies',not external_scripts,external_scripts)
check('external stylesheet dependencies',not external_styles,external_styles)

# main app identity, access and no-removal preservation
new_text,new_soup,_=refs_for(BASE/'index.html')
old_soup=BeautifulSoup(OLD.read_text(encoding='utf-8',errors='ignore'),'html.parser')
old_ids={e.get('id') for e in old_soup.find_all(id=True)}
new_ids={e.get('id') for e in new_soup.find_all(id=True)}
check('predecessor interface identifiers preserved',old_ids<=new_ids,{'predecessor':len(old_ids),'current':len(new_ids),'missing':sorted(old_ids-new_ids),'added':len(new_ids-old_ids)})
for needle in ['0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e','0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401','label.club.agi.eth','sessionMinutes:30','recheckMinutes:5','personal_sign','accountsChanged','chainChanged','disconnect']:
    check(f'access control contains {needle}',needle in new_text)
check('no server access API dependency','/api/access/' not in new_text)
check('CSP permits only self/data/blob images',"img-src 'self' data: blob:" in new_text)
check('CSP blocks network connections',"connect-src 'none'" in new_text)
check('release version identified','v2.0.0-SN3' in new_text)
check('canonical path identified','/goalos-singularity-navigator-omega-agi-club-owner-access/' in new_text)

# JSON files
json_errors=[]
for f in BASE.rglob('*.json'):
    try: json.loads(f.read_text(encoding='utf-8'))
    except Exception as e: json_errors.append((str(f.relative_to(BASE)),str(e)))
check('JSON documents parse',not json_errors,json_errors)
schema=json.loads((BASE/'schemas/agi_club_owner_access_receipt_v2.schema.json').read_text())
patterns=[schema['properties'][k]['pattern'] for k in ('name','address','node')]
check('receipt schema regexes normalized',all('{{' not in x and '}}' not in x for x in patterns),patterns)
try:
    import jsonschema
    jsonschema.Draft202012Validator.check_schema(schema)
    check('receipt schema is valid Draft 2020-12',True)
except Exception as e:
    check('receipt schema is valid Draft 2020-12',False,str(e))

# JavaScript syntax in site and standalone
js_results=[]
for f in [BASE/'index.html',BASE/'404.html',STANDALONE]:
    soup=BeautifulSoup(f.read_text(encoding='utf-8',errors='ignore'),'html.parser')
    scripts=[x.string or x.get_text() for x in soup.find_all('script') if not x.get('src')]
    for i,js in enumerate(scripts,1):
        with tempfile.NamedTemporaryFile('w',suffix='.js',delete=False,encoding='utf-8') as t:
            t.write(js); name=t.name
        p=subprocess.run(['node','--check',name],capture_output=True,text=True)
        os.unlink(name)
        js_results.append({'file':f.name,'script':i,'passed':p.returncode==0,'stderr':p.stderr.strip()})
check('executable inline JavaScript syntax',all(x['passed'] for x in js_results),js_results)

# PDF and image basics
paper=BASE/'research/GoalOS_Singularity_Navigation_Omega_Tools_to_Ride_the_Singularity_by_Vincent_Boucher_v2_0_0.pdf'
try:
    p=subprocess.run(['pdfinfo',str(paper)],capture_output=True,text=True,check=True)
    m=re.search(r'^Pages:\s+(\d+)',p.stdout,re.M)
    pages=int(m.group(1)) if m else None
    check('founding paper parses and has 140 pages',pages==140,pages)
except Exception as e: check('founding paper parses and has 140 pages',False,str(e))

out={'passed':all(r['passed'] for r in results),'assertions':len(results),'results':results}
Path('/tmp/sn_static_qa.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
if not out['passed']: sys.exit(1)
