from pathlib import Path
import json, hashlib, zipfile, datetime, subprocess, re, shutil, os
base=Path('/mnt/data')
slug='goalos-partner-masterclass-sovereign-v2'
out=base/slug

# README
readme=f'''# GoalOS Partner MASTERCLASS — Sovereign Edition

An 18-module partner-facing operating institution for proof-carrying autonomy, Mission OS, Proof Debt, custom AGI Jobs, Evidence Dockets, AGI Node commit-reveal validation, Chronicle, Validated Skill Graphs, Merkle and privacy controls, bonded authority, recursive self-improvement, Move-37 breakthrough handling, Mission 1 -> Mission 2 transfer, REAL-001, partner economics, architecture mapping, due diligence, and a 30-day pilot.

## Start

Open `{slug}.html` in a modern browser. The application is self-contained and requires no server, account, wallet, model API, or external asset.

## Presentation formats

- Executive 15 minutes
- Boardroom 30 minutes
- Working session 90 minutes
- Self-paced certification

## Core empirical law

Mission 1 must not merely be remembered. It must earn the right to make Mission 2 measurably better under equal constraints.

## Claim boundary

The package combines interactive simulations, reference architecture, and REAL-001 first-party evidence. Independent external replay, general real-world RSI, production authority, guaranteed outcomes, token value, legal certification, and achieved AGI/ASI are not claimed.
'''
(out/'README.md').write_text(readme,encoding='utf-8')

# Counts and tests.
browser=json.loads((out/'qa'/'browser-test.json').read_text())
slide_count=len(list((out/'presentation'/'rendered').glob('*.png')))
handbook_count=len(list((out/'docs'/'handbook-render').glob('page-*.png')))
html=(out/f'{slug}.html').read_text(encoding='utf-8')
external_assets=re.findall(r'<(?:script|link|img)[^>]+(?:src|href)=["\']https?://',html,re.I)
# JS syntax
from bs4 import BeautifulSoup
soup=BeautifulSoup(html,'html.parser')
js='\n'.join(sc.get_text() for sc in soup.find_all('script'))
tmp=out/'qa'/'_masterclass.js'; tmp.write_text(js)
node=subprocess.run(['node','--check',str(tmp)],capture_output=True,text=True)
tmp.unlink(missing_ok=True)
# PowerPoint test
pptx_test=subprocess.run(['python','/home/oai/skills/slides/container_tools/slides_test.py',str(out/'presentation'/f'{slug}-deck.pptx')],capture_output=True,text=True)
checks={
 'standalone_exists':(out/f'{slug}.html').exists(),
 'standalone_javascript':node.returncode==0,
 'standalone_external_assets_zero':len(external_assets)==0,
 'all_18_modules_render':browser.get('all_modules_rendered') is True,
 'browser_console_errors_zero':len(browser.get('errors',[]))==0,
 'grand_demo_mission2_pass':browser.get('grand_demo_mission2_pass') is True,
 'grand_demo_chronicle_admit':browser.get('grand_demo_chronicle')=='ADMIT_WITH_SCOPE',
 'grand_demo_six_custom_jobs':browser.get('grand_demo_jobs')==6,
 'grand_demo_24_evidence_artifacts':browser.get('grand_demo_evidence_artifacts')==24,
 'tamper_blocks_mission2':browser.get('tamper_blocks_mission2') is True,
 'deck_28_slides':slide_count==28,
 'deck_overflow_test':pptx_test.returncode==0,
 'deck_pdf_exists':(out/'presentation'/f'{slug}-deck.pdf').exists(),
 'handbook_docx_exists':(out/'docs'/f'{slug}-handbook.docx').exists(),
 'handbook_pdf_exists':(out/'docs'/f'{slug}-handbook.pdf').exists(),
 'handbook_13_pages':handbook_count==13,
 'real001_evidence_included':(out/'evidence'/'goalos-real-001-proof-pack.json').exists(),
 'autonomous_workflow_included':(out/'workflows'/'goalos-real-001-autonomous-github-action.yml').exists(),
 'independent_replay_workflow_included':(out/'workflows'/'goalos-real-001-independent-replay-action.yml').exists(),
 'canonical_source_library_included':len(list((out/'sources').glob('*.pdf')))>=7,
}
validation={'title':'GoalOS Partner MASTERCLASS — Sovereign Edition','version':'2.0','generatedAt':datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),'status':'pass' if all(checks.values()) else 'partial','checks':checks,'browser':browser,'slideCount':slide_count,'handbookPages':handbook_count,'claimBoundary':'Interactive simulations and first-party evidence do not constitute independent external validation, production authority, general RSI, token valuation, legal certification, or achieved AGI/ASI.'}
(out/'qa'/f'{slug}-validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
qa='''GOALOS PARTNER MASTERCLASS — SOVEREIGN EDITION QA REPORT

Status: {status}
Interactive modules: 18
Executive deck: {slides} slides
Handbook: {pages} pages
Required external interface assets: {assets}
JavaScript syntax: {js}
Browser console / page errors: {errors}
Grand demonstration: {demo}
Chronicle decision: {chronicle}
Tamper-triggered Mission 2 block: {tamper}
PowerPoint overflow test: {pptx}
Visual QA: home, transfer, Merkle room, full slide contact sheet, and handbook contact sheet inspected.

Claim boundary: the package contains simulations, reference architecture, and REAL-001 first-party evidence. Independent external replay and stronger production or general-RSI claims remain gated.
'''.format(status=validation['status'].upper(),slides=slide_count,pages=handbook_count,assets=len(external_assets),js='PASS' if node.returncode==0 else 'FAIL',errors='0' if not browser['errors'] else len(browser['errors']),demo='PASS' if browser['grand_demo_mission2_pass'] else 'FAIL',chronicle=browser['grand_demo_chronicle'],tamper='PASS' if browser['tamper_blocks_mission2'] else 'FAIL',pptx='PASS' if pptx_test.returncode==0 else 'FAIL')
(out/'qa'/f'{slug}-QA-REPORT.txt').write_text(qa,encoding='utf-8')

# Hash and manifest helpers.
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def zip_dir(dst, paths, root=out):
    if dst.exists(): dst.unlink()
    with zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in paths:
            p=Path(p)
            if p.is_dir():
                for f in sorted(p.rglob('*')):
                    if f.is_file() and 'rendered' not in f.parts and 'handbook-render' not in f.parts:
                        z.write(f,f.relative_to(root))
            elif p.exists(): z.write(p,p.relative_to(root))

presentation_zip=base/f'{slug}-presentation-pack.zip'
zip_dir(presentation_zip,[out/f'{slug}.html',out/f'{slug}-download-center.html',out/f'{slug}-preview.png',out/'presentation',out/'docs',out/'qa'],out)
interactive_zip=base/f'{slug}-interactive.zip'
zip_dir(interactive_zip,[out/f'{slug}.html',out/f'{slug}-download-center.html',out/'evidence',out/'workflows'],out)

everything_zip=base/f'{slug}-everything.zip'
if everything_zip.exists(): everything_zip.unlink()
with zipfile.ZipFile(everything_zip,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
    for f in sorted(out.rglob('*')):
        if f.is_file() and 'rendered' not in f.parts and 'handbook-render' not in f.parts:
            z.write(f,f.relative_to(out.parent))

# Manifest.
finals=[]
for f in sorted(out.rglob('*')):
    if f.is_file() and 'rendered' not in f.parts and 'handbook-render' not in f.parts:
        finals.append(f)
for f in [presentation_zip,interactive_zip,everything_zip]: finals.append(f)
manifest={'title':'GoalOS Partner MASTERCLASS — Sovereign Edition','version':'2.0','generatedAt':validation['generatedAt'],'modules':18,'slides':slide_count,'handbookPages':handbook_count,'artifacts':[]}
for f in finals:
    manifest['artifacts'].append({'file':str(f.relative_to(base)),'bytes':f.stat().st_size,'sha256':sha(f)})
manifest_path=out/f'{slug}-manifest.json'; manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
# Checksums after manifest.
checksum_files=[f for f in sorted(out.rglob('*')) if f.is_file() and 'rendered' not in f.parts and 'handbook-render' not in f.parts]+[presentation_zip,interactive_zip,everything_zip]
(out/f'{slug}-SHA256SUMS.txt').write_text('\n'.join(f'{sha(f)}  {f.relative_to(base)}' for f in checksum_files)+'\n',encoding='utf-8')

# Convenience root copies.
root_copies={
 out/f'{slug}.html':base/f'{slug}.html',
 out/f'{slug}-download-center.html':base/f'{slug}-download-center.html',
 out/f'{slug}-preview.png':base/f'{slug}-preview.png',
 out/'presentation'/f'{slug}-deck.pptx':base/f'{slug}-deck.pptx',
 out/'presentation'/f'{slug}-deck.pdf':base/f'{slug}-deck.pdf',
 out/'docs'/f'{slug}-handbook.docx':base/f'{slug}-handbook.docx',
 out/'docs'/f'{slug}-handbook.pdf':base/f'{slug}-handbook.pdf',
}
for src,dst in root_copies.items(): shutil.copy2(src,dst)
print(json.dumps({'validation':validation['status'],'everything':str(everything_zip),'presentation':str(presentation_zip),'interactive':str(interactive_zip)},indent=2))
