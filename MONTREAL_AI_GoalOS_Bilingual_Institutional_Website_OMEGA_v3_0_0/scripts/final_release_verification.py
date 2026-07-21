#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import datetime, json, re, subprocess, sys, zipfile
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'FINAL_VERIFICATION_REPORT.json'
checks=[]
errors=[]

def record(name, ok, detail='', metrics=None):
    checks.append({'name':name,'status':'PASS' if ok else 'FAIL','detail':detail,'metrics':metrics or {}})
    if not ok: errors.append(f'{name}: {detail}')

def load_json(rel): return json.loads((ROOT/rel).read_text('utf-8'))

# Existing QA reports
for rel,label in [
    ('qa/STATIC_VERIFICATION_REPORT.json','Static site verification'),
    ('qa/ACCESSIBILITY_AND_CONTENT_QA_REPORT.json','Accessibility and content QA'),
    ('qa/BROWSER_QA_REPORT.json','Browser rendering QA'),
    ('qa/NAVIGATION_QA_REPORT.json','Navigation and local preview QA')]:
    try:
        d=load_json(rel)
        if rel.endswith('NAVIGATION_QA_REPORT.json'):
            ok=d.get('status')=='PASS' and not d.get('local_path_issues') and not d.get('http_page_failures') and not d.get('journey_failures')
            metrics={'pages':d.get('public_pages'),'internal_links':d.get('internal_links_checked'),'http_pages_passed':d.get('http_pages_passed'),'journeys':len(d.get('journeys',[]))}
        elif 'errors' in d:
            ok=not d.get('errors') and not d.get('warnings')
            metrics={'errors':len(d.get('errors',[])),'warnings':len(d.get('warnings',[])),'pages':d.get('public_pages')}
        else:
            fail_keys=['render_failures','runtime_error_scenarios','overflow_failures','broken_image_scenarios','language_failures','h1_failures','structure_failures','mobile_menu_failures']
            ok=all(not d.get(k) for k in fail_keys)
            metrics={k:len(d.get(k,[])) for k in fail_keys}|{'pages':d.get('pages_tested'),'scenarios':d.get('scenarios')}
        record(label,ok,rel,metrics)
    except Exception as e: record(label,False,str(e))

# JavaScript and Python syntax
p=subprocess.run(['node','--check',str(ROOT/'goalos-assets/js/site.js')],capture_output=True,text=True)
record('Public JavaScript syntax',p.returncode==0,(p.stderr or p.stdout).strip())
py_files=sorted((ROOT/'scripts').glob('*.py'))+sorted((ROOT/'BUILD_SOURCE').glob('*.py'))
failed=[]
for f in py_files:
    r=subprocess.run([sys.executable,'-m','py_compile',str(f)],capture_output=True,text=True)
    if r.returncode: failed.append({'file':f.relative_to(ROOT).as_posix(),'error':(r.stderr or r.stdout).strip()})
record('Python source compilation',not failed,'All Python sources compile' if not failed else str(failed),{'files':len(py_files)})

# JSON parsing
json_files=sorted(ROOT.rglob('*.json')); bad=[]
for f in json_files:
    try: json.loads(f.read_text('utf-8'))
    except Exception as e: bad.append({'file':f.relative_to(ROOT).as_posix(),'error':str(e)})
record('JSON parsing',not bad,'All JSON parses' if not bad else str(bad),{'files':len(json_files)})

# PDF preflight
pdfs=sorted(ROOT.rglob('*.pdf')); bad=[]; pages=0
for f in pdfs:
    r=subprocess.run(['pdfinfo',str(f)],capture_output=True,text=True)
    if r.returncode:
        bad.append({'file':f.relative_to(ROOT).as_posix(),'error':r.stderr.strip()})
    else:
        m=re.search(r'^Pages:\s+(\d+)',r.stdout,re.M); pages+=int(m.group(1)) if m else 0
record('PDF preflight',not bad,'All PDFs readable' if not bad else str(bad),{'files':len(pdfs),'total_pages':pages})

# ZIP/XLSX integrity
archives=sorted([p for p in ROOT.rglob('*') if p.suffix.lower() in {'.zip','.xlsx','.pptx','.docx'}]); bad=[]
for f in archives:
    try:
        with zipfile.ZipFile(f) as z:
            corrupt=z.testzip()
            if corrupt: bad.append({'file':f.relative_to(ROOT).as_posix(),'corrupt_member':corrupt})
            names=set(z.namelist())
            if f.suffix.lower()=='.xlsx' and '[Content_Types].xml' not in names: bad.append({'file':f.relative_to(ROOT).as_posix(),'error':'missing [Content_Types].xml'})
    except Exception as e: bad.append({'file':f.relative_to(ROOT).as_posix(),'error':str(e)})
record('Office and ZIP integrity',not bad,'All archives pass integrity checks' if not bad else str(bad),{'files':len(archives)})

# Image integrity
images=sorted(p for p in ROOT.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}); bad=[]
for f in images:
    try:
        with Image.open(f) as im: im.verify()
    except Exception as e: bad.append({'file':f.relative_to(ROOT).as_posix(),'error':str(e)})
record('Image integrity',not bad,'All images decode' if not bad else str(bad),{'files':len(images)})

# Secret and placeholder scan
text_ext={'.html','.js','.css','.json','.md','.txt','.yml','.yaml','.xml','.py','.svg','.webmanifest'}
secret_re=re.compile(r'(?i)(BEGIN [A-Z ]*PRIVATE KEY|aws_secret_access_key\s*=|(?:api[_-]?key|secret|token)\s*[:=]\s*["\'][A-Za-z0-9_\-]{28,}["\']|sk-[A-Za-z0-9]{20,})')
placeholder_re=re.compile(r'(?i)(REPLACE_WITH_REAL_SECRET|YOUR_API_KEY_HERE|INSERT_PRIVATE_KEY)')
findings=[]
for f in ROOT.rglob('*'):
    if not f.is_file() or f.suffix.lower() not in text_ext: continue
    if '__pycache__' in f.parts or f.resolve()==Path(__file__).resolve(): continue
    try: t=f.read_text('utf-8')
    except Exception: continue
    if secret_re.search(t) or placeholder_re.search(t): findings.append(f.relative_to(ROOT).as_posix())
record('Credential and production-placeholder scan',not findings,'No credential-pattern findings' if not findings else ', '.join(findings),{'text_files_scanned':sum(1 for f in ROOT.rglob('*') if f.is_file() and f.suffix.lower() in text_ext)})

# Required release surfaces
required=[
 'index.html','en/index.html','goalos/index.html','en/goalos/index.html',
 'proof-missions/index.html','en/proof-missions/index.html',
 'proof-missions/ai-deployment/index.html','proof-missions/ai-compliance/index.html','proof-missions/rfp-to-revenue/index.html','proof-missions/diligence-capital/index.html','proof-missions/venture-office/index.html',
 'goalos-legal/index.html','en/goalos-legal/index.html','goalos-assets/css/site.css','goalos-assets/js/site.js',
 'source-register.json','sitemap.xml','manifest.webmanifest','.nojekyll','README.md','DEPLOYMENT.md','PRODUCTION_READINESS_DECISION.md','LEGAL_ACTIVATION_CHECKLIST.md','LIVE_DEPLOYMENT_ACCEPTANCE.md','start_local_preview.py','START_HERE.html','LOCAL_PREVIEW_README.md'
]
missing=[x for x in required if not (ROOT/x).exists()]
record('Required release surface',not missing,'All required surfaces present' if not missing else ', '.join(missing),{'required':len(required)})

# Counts and exact state
htmls=[p for p in ROOT.rglob('*.html') if p.relative_to(ROOT).as_posix()!='START_HERE.html' and not p.relative_to(ROOT).as_posix().startswith(('goalos-documents/','BUILD_SOURCE/','qa/'))]
fr=[p for p in htmls if not p.relative_to(ROOT).as_posix().startswith('en/')]
en=[p for p in htmls if p.relative_to(ROOT).as_posix().startswith('en/')]
record('Bilingual page count',len(htmls)==74 and len(fr)==37 and len(en)==37,f'{len(htmls)} pages; {len(fr)} FR; {len(en)} EN',{'pages':len(htmls),'french':len(fr),'english':len(en)})

report={
 'schema':'montrealai.goalos.final-verification.v3',
 'release':'MONTREAL_AI_GoalOS_Bilingual_Institutional_Website_OMEGA_v3_0_0',
 'version':'3.0.0',
 'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'decision':'PASS' if not errors else 'FAIL',
 'checks':checks,
 'errors':errors,
 'activation_boundary':{
  'static_release_candidate_ready':not errors,
  'live_github_pages_deployment_performed':False,
  'live_https_acceptance_required':True,
  'independent_legal_review_required':True,
  'independent_security_review_required_for_activated_services':True,
  'commercial_proof_earned':False,
  'customer_outcome_verified':False,
  'repeat_purchase_earned':False,
  'fresh_transfer_improvement_earned':False
 }
}
OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n','utf-8')
print(f'FINAL RELEASE VERIFICATION: {report["decision"]} · {len(checks)} checks · {len(errors)} failures')
if errors:
    print('\n'.join(errors)); sys.exit(1)
