#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import hashlib, html, json, shutil, tempfile, zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/data/MONTREAL_AI_GoalOS_Bilingual_Institutional_Website_OMEGA_v3_0_0_Downloads')
RELEASE=ROOT.name
VERSION='3.0.0'
FIXED_DT=(2026,7,20,0,0,0)
OUT.mkdir(parents=True,exist_ok=True)

SKIP_PARTS={'__pycache__','.DS_Store'}
SKIP_SUFFIX={'.pyc','.pyo'}
MANIFEST_EXCLUDE={'FINAL_RELEASE_MANIFEST.json','SHA256SUMS.txt'}

def eligible(p:Path)->bool:
    rel=p.relative_to(ROOT)
    return p.is_file() and not any(x in SKIP_PARTS for x in rel.parts) and p.suffix.lower() not in SKIP_SUFFIX

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def release_root(entries:list[dict])->str:
    canonical=''.join(f"{x['sha256']}  {x['size']}  {x['path']}\n" for x in entries).encode()
    return hashlib.sha256(canonical).hexdigest()

def all_files()->list[Path]:
    return sorted((p for p in ROOT.rglob('*') if eligible(p)),key=lambda p:p.relative_to(ROOT).as_posix())

def overlay_files()->list[Path]:
    result=[]
    excluded_roots={'BUILD_SOURCE','qa'}
    excluded_names={'START_HERE.html','START_LOCAL_PREVIEW.command','START_LOCAL_PREVIEW.bat','LOCAL_PREVIEW_README.md','start_local_preview.py'}
    for p in all_files():
        rel=p.relative_to(ROOT)
        if rel.parts and rel.parts[0] in excluded_roots: continue
        if rel.as_posix() in excluded_names: continue
        result.append(p)
    return result

def entries_for(files:list[Path],root=ROOT)->list[dict]:
    return [{'path':p.relative_to(root).as_posix(),'size':p.stat().st_size,'sha256':sha(p)} for p in files]

def write_json(p:Path,obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n','utf-8')

# Overlay manifest is included in the complete release, but does not self-list.
overlay_pre=[p for p in overlay_files() if p.name not in {'SITE_OVERLAY_MANIFEST.json','FINAL_RELEASE_MANIFEST.json','SHA256SUMS.txt'}]
overlay_entries=entries_for(overlay_pre)
overlay_manifest={
 'schema':'montrealai.goalos.site-overlay-manifest.v3',
 'release':RELEASE,'version':VERSION,'generated_at':'2026-07-20T00:00:00Z',
 'repository':'MontrealAI/MontrealAI.github.io','branch':'master','target':'repository root',
 'deployment_method':'non-destructive overlay; add or replace listed paths; do not delete unlisted legacy paths',
 'previous_root_index_blob_sha_observed':'bddf438cab761f6b5566c2f00b92dc77e4f0a032',
 'live_deployment_performed':False,
 'files':overlay_entries,'overlay_root':release_root(overlay_entries),
 'instructions':['Create a review branch from current master.','Extract this overlay at repository root.','Inspect the complete diff and preserve the previous commit for rollback.','Run the included static, accessibility/content and navigation verification scripts.','Preview over HTTPS; complete legal, communications and security review; merge only after authorization.']
}
write_json(ROOT/'SITE_OVERLAY_MANIFEST.json',overlay_manifest)

# Final payload manifest intentionally excludes itself and the text checksum file.
payload=[p for p in all_files() if p.name not in MANIFEST_EXCLUDE]
complete_entries=entries_for(payload)
complete_manifest={
 'schema':'montrealai.goalos.final-release-manifest.v3','release':RELEASE,'version':VERSION,'generated_at':'2026-07-20T00:00:00Z',
 'release_root':release_root(complete_entries),'file_count':len(complete_entries),'total_bytes':sum(x['size'] for x in complete_entries),
 'public_site':{'pages':74,'french_pages':37,'english_pages':37,'primary_language':'fr-CA','english_mirror':'/en/'},
 'public_surface':{'accounts':False,'server_forms':False,'analytics':False,'wallet_connection':False,'payments':False,'transactions':False,'public_chatbot':False,'external_runtime_dependencies':False,'customer_data_processing':False},
 'activation_boundary':{'static_release_candidate_ready':True,'live_deployment_performed':False,'live_https_acceptance_required':True,'independent_legal_review_required':True,'independent_security_review_required_for_activated_services':True,'commercial_proof_earned':False,'customer_outcome_verified':False,'repeat_purchase_earned':False,'fresh_transfer_improvement_earned':False,'valuation_claimed':False},
 'files':complete_entries
}
write_json(ROOT/'FINAL_RELEASE_MANIFEST.json',complete_manifest)
(ROOT/'SHA256SUMS.txt').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in complete_entries),'utf-8')

# Regenerate after the overlay manifest and checksum files exist.
overlay_now=[p for p in overlay_files() if p.name not in {'SITE_OVERLAY_MANIFEST.json','FINAL_RELEASE_MANIFEST.json','SHA256SUMS.txt'}]
overlay_entries_final=entries_for(overlay_now)
overlay_manifest.update({'files':overlay_entries_final,'overlay_root':release_root(overlay_entries_final),'file_count':len(overlay_entries_final),'total_bytes':sum(x['size'] for x in overlay_entries_final)})
write_json(ROOT/'SITE_OVERLAY_MANIFEST.json',overlay_manifest)
payload=[p for p in all_files() if p.name not in MANIFEST_EXCLUDE]
complete_entries=entries_for(payload)
complete_manifest.update({'release_root':release_root(complete_entries),'file_count':len(complete_entries),'total_bytes':sum(x['size'] for x in complete_entries),'files':complete_entries})
write_json(ROOT/'FINAL_RELEASE_MANIFEST.json',complete_manifest)
(ROOT/'SHA256SUMS.txt').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in complete_entries),'utf-8')

# Deterministic ZIP utilities.
def zip_bytes(entries:list[tuple[Path,str]],dest:Path):
    dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for source,arc in sorted(entries,key=lambda x:x[1]):
            info=zipfile.ZipInfo(arc,FIXED_DT); info.compress_type=zipfile.ZIP_DEFLATED; info.create_system=3
            mode=0o100755 if source.stat().st_mode & 0o111 else 0o100644
            info.external_attr=(mode & 0xFFFF)<<16; info.flag_bits|=0x800
            z.writestr(info,source.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)

def verify_zip(zp:Path,expected:list[tuple[Path,str]])->dict:
    expected_map={arc:sha(src) for src,arc in expected}
    with zipfile.ZipFile(zp) as z:
        names=z.namelist(); corrupt=z.testzip(); unsafe=[n for n in names if n.startswith('/') or '..' in Path(n).parts or '\\' in n]
        duplicates=sorted({n for n in names if names.count(n)>1}); missing=sorted(set(expected_map)-set(names)); extra=sorted(set(names)-set(expected_map))
        mismatched=[n for n,h in expected_map.items() if n in names and hashlib.sha256(z.read(n)).hexdigest()!=h]
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zp) as z: z.extractall(td)
        extract_bad=[]
        for src,arc in expected:
            q=Path(td)/arc
            if not q.exists() or sha(q)!=sha(src): extract_bad.append(arc)
    with tempfile.TemporaryDirectory() as td:
        rebuilt=Path(td)/zp.name; zip_bytes(expected,rebuilt); deterministic=sha(rebuilt)==sha(zp)
    return {'zip_test':corrupt is None,'unsafe_paths':unsafe,'duplicate_entries':duplicates,'missing_entries':missing,'extra_entries':extra,'mismatched_entries':mismatched,'clean_extraction_mismatches':extract_bad,'deterministic_rebuild':deterministic,'entry_count':len(expected),'size':zp.stat().st_size,'sha256':sha(zp)}

# Package selections.
overlay=overlay_files()
packages={
 'MONTREAL_AI_GoalOS_GITHUB_PAGES_DEPLOY_OVERLAY_v3_0_0.zip':[(p,p.relative_to(ROOT).as_posix()) for p in overlay],
 'MONTREAL_AI_GoalOS_COMPLETE_INSTITUTIONAL_WEBSITE_RELEASE_v3_0_0.zip':[(p,f'{RELEASE}/{p.relative_to(ROOT).as_posix()}') for p in all_files()],
}

legal_top=['README.md','DEPLOYMENT.md','PRODUCTION_READINESS_DECISION.md','LEGAL_ACTIVATION_CHECKLIST.md','LIVE_DEPLOYMENT_ACCEPTANCE.md','SECURITY.md','PRIVACY_SURFACE.md','CLAIM_AUTHORITY_LADDER.md','source-register.json','FINAL_VERIFICATION_REPORT.json','FINAL_RELEASE_MANIFEST.json','SITE_OVERLAY_MANIFEST.json','SHA256SUMS.txt']
legal_files=[ROOT/x for x in legal_top]
for prefix in ['goalos-legal','en/goalos-legal','goalos-documents/research']:
    legal_files += [p for p in (ROOT/prefix).rglob('*') if eligible(p)]
legal_files=[p for p in legal_files if eligible(p)]
packages['MONTREAL_AI_GoalOS_LEGAL_AND_GOVERNANCE_REVIEW_PACK_v3_0_0.zip']=[(p,f'MONTREAL_AI_GoalOS_Legal_and_Governance_Review_v3_0_0/{p.relative_to(ROOT).as_posix()}') for p in sorted(set(legal_files),key=lambda p:p.relative_to(ROOT).as_posix())]

source_files=[]
for prefix in ['BUILD_SOURCE','scripts','.github','qa']:
    q=ROOT/prefix
    if q.exists(): source_files += [p for p in q.rglob('*') if eligible(p)]
source_files += [ROOT/x for x in ['README.md','RELEASE_NOTES.md','CHANGELOG.md','DEPLOYMENT.md','PRODUCTION_READINESS_DECISION.md','FINAL_VERIFICATION_REPORT.json','FINAL_RELEASE_MANIFEST.json','SITE_OVERLAY_MANIFEST.json','SHA256SUMS.txt','source-register.json']]
source_files=[p for p in source_files if eligible(p)]
packages['MONTREAL_AI_GoalOS_SOURCE_AND_QA_PACK_v3_0_0.zip']=[(p,f'MONTREAL_AI_GoalOS_Source_and_QA_v3_0_0/{p.relative_to(ROOT).as_posix()}') for p in sorted(set(source_files),key=lambda p:p.relative_to(ROOT).as_posix())]

brand_files=[]
for prefix in ['goalos-assets/img/brand','goalos-assets/img/business-model','goalos-documents/business-model']:
    brand_files += [p for p in (ROOT/prefix).rglob('*') if eligible(p)]
for rel in ['goalos-assets/img/og-card.jpg','goalos-assets/icons/favicon.svg','manifest.webmanifest','business-model/index.html','en/business-model/index.html','autonomous-money-machine/index.html','en/autonomous-money-machine/index.html']:
    q=ROOT/rel
    if q.exists() and eligible(q): brand_files.append(q)
packages['MONTREAL_AI_GoalOS_BRAND_AND_BUSINESS_MODEL_ASSETS_v3_0_0.zip']=[(p,f'MONTREAL_AI_GoalOS_Brand_and_Business_Model_v3_0_0/{p.relative_to(ROOT).as_posix()}') for p in sorted(set(brand_files),key=lambda p:p.relative_to(ROOT).as_posix())]

# Remove stale outputs and build.
for p in OUT.iterdir():
    if p.is_file(): p.unlink()
verification={}; package_records=[]
purposes={
 'MONTREAL_AI_GoalOS_GITHUB_PAGES_DEPLOY_OVERLAY_v3_0_0.zip':'Extract directly into the reviewed MontrealAI.github.io repository root; no wrapper directory.',
 'MONTREAL_AI_GoalOS_COMPLETE_INSTITUTIONAL_WEBSITE_RELEASE_v3_0_0.zip':'Complete website, documents, products, local preview, source, QA screenshots, reports and integrity records.',
 'MONTREAL_AI_GoalOS_LEGAL_AND_GOVERNANCE_REVIEW_PACK_v3_0_0.zip':'Focused bilingual legal, privacy, security, claim, research and activation-review materials.',
 'MONTREAL_AI_GoalOS_SOURCE_AND_QA_PACK_v3_0_0.zip':'Build provenance, verification scripts, CI, browser screenshots, QA reports and integrity records.',
 'MONTREAL_AI_GoalOS_BRAND_AND_BUSINESS_MODEL_ASSETS_v3_0_0.zip':'Institutional brand assets, Business Model pages, executive brief, strategy, playbook, decks, model and launch plan.'
}
for name,entries in packages.items():
    dest=OUT/name; zip_bytes(entries,dest); v=verify_zip(dest,entries); verification[name]=v
    package_records.append({'file':name,'purpose':purposes[name],**{k:v[k] for k in ['size','sha256','entry_count','deterministic_rebuild']}})

package_ok=all(v['zip_test'] and not v['unsafe_paths'] and not v['duplicate_entries'] and not v['missing_entries'] and not v['extra_entries'] and not v['mismatched_entries'] and not v['clean_extraction_mismatches'] and v['deterministic_rebuild'] for v in verification.values())
package_report={'schema':'montrealai.goalos.package-verification.v3','release':RELEASE,'generated_at':'2026-07-20T00:00:00Z','decision':'PASS' if package_ok else 'FAIL','packages':verification}
write_json(OUT/'PACKAGE_VERIFICATION.json',package_report)
download_manifest={'schema':'montrealai.goalos.download-manifest.v3','release':RELEASE,'version':VERSION,'release_root':complete_manifest['release_root'],'overlay_root':overlay_manifest['overlay_root'],'packages':package_records,'activation_boundary':complete_manifest['activation_boundary']}
write_json(OUT/'DOWNLOAD_MANIFEST.json',download_manifest)
(OUT/'DOWNLOAD_SHA256SUMS.txt').write_text(''.join(f"{x['sha256']}  {x['file']}\n" for x in package_records),'utf-8')
for name in ['FINAL_RELEASE_MANIFEST.json','SITE_OVERLAY_MANIFEST.json','FINAL_VERIFICATION_REPORT.json','SHA256SUMS.txt']:
    shutil.copy2(ROOT/name,OUT/name)

rows=''.join(f"<article><h2>{html.escape(x['file'])}</h2><p>{html.escape(x['purpose'])}</p><dl><div><dt>Size</dt><dd>{x['size']/1024/1024:.1f} MB</dd></div><div><dt>Files</dt><dd>{x['entry_count']}</dd></div><div><dt>SHA-256</dt><dd><code>{x['sha256']}</code></dd></div></dl><a href=\"{html.escape(x['file'])}\">Download / Télécharger</a></article>" for x in package_records)
center=f'''<!doctype html><html lang="fr-CA"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MONTREAL.AI × GoalOS — Download Center v3.0.0</title><style>:root{{--navy:#061426;--gold:#d6a63f;--ivory:#fffaf0;--ink:#102942}}*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,system-ui,sans-serif;background:radial-gradient(circle at 85% 5%,#ecd9a766,transparent 28%),linear-gradient(160deg,#fffdf8,#f4ead6);color:var(--ink)}}main{{width:min(1180px,calc(100% - 30px));margin:auto;padding:54px 0 80px}}header{{background:linear-gradient(145deg,#04101f,#0a2946);color:white;padding:38px;border-radius:28px;box-shadow:0 26px 80px #071b3033}}h1{{font:800 clamp(2.4rem,7vw,5.6rem)/.92 Georgia,serif;margin:.15em 0}}header p{{max-width:900px;line-height:1.65;color:#dce8f4}}.law{{margin-top:18px;border-left:3px solid var(--gold);padding:12px 14px;background:#ffffff10;border-radius:0 12px 12px 0}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:20px}}article{{background:white;border:1px solid #d9c9a6;border-radius:22px;padding:22px;box-shadow:0 12px 38px #071b3010}}h2{{font:700 1.18rem Georgia,serif;overflow-wrap:anywhere}}p{{line-height:1.6}}dl div{{display:grid;grid-template-columns:90px 1fr;gap:10px;padding:7px 0;border-top:1px solid #eee}}dt{{font-weight:700}}dd{{margin:0;min-width:0}}code{{font-size:.72rem;overflow-wrap:anywhere}}a{{display:inline-block;margin-top:12px;padding:12px 16px;border-radius:12px;background:linear-gradient(135deg,#efcb76,#bd7e16);color:#1c1306;text-decoration:none;font-weight:800}}.links{{margin-top:24px;padding:20px;border:1px solid #d9c9a6;border-radius:20px;background:#fff}}.links a{{background:var(--navy);color:white;margin:4px}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}header{{padding:26px 20px}}}}</style></head><body><main><header><small>MONTREAL.AI × GOALOS · OMEGA RELEASE</small><h1>DOWNLOAD CENTER</h1><p>Site institutionnel français-primaire, miroir anglais complet, modèle économique optimal, cinq classes de missions, Mission Lab non confidentiel, centre juridique bilingue, registre de preuve, rempart de capacité privée, sources, QA, manifestes et paquets déterministes.</p><div class="law"><b>No secrets before contract.</b> The public website is a non-confidential publication and qualification surface.</div><p><b>Release root:</b> <code>{complete_manifest['release_root']}</code><br><b>Overlay root:</b> <code>{overlay_manifest['overlay_root']}</code></p></header><section class="grid">{rows}</section><section class="links"><h2>Integrity / Intégrité</h2><a href="DOWNLOAD_MANIFEST.json">Download manifest</a><a href="PACKAGE_VERIFICATION.json">Package verification</a><a href="FINAL_VERIFICATION_REPORT.json">Final verification</a><a href="FINAL_RELEASE_MANIFEST.json">Release manifest</a><a href="SITE_OVERLAY_MANIFEST.json">Overlay manifest</a><a href="DOWNLOAD_SHA256SUMS.txt">ZIP checksums</a><a href="SHA256SUMS.txt">Payload checksums</a></section></main></body></html>'''
(OUT/'DOWNLOAD_CENTER.html').write_text(center,'utf-8')
metadata_checks={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
write_json(OUT/'FINAL_PACKAGE_VERIFICATION.json',{'schema':'montrealai.goalos.final-package-verification.v3','release':RELEASE,'generated_at':'2026-07-20T00:00:00Z','decision':'PASS' if package_ok else 'FAIL','release_root':complete_manifest['release_root'],'overlay_root':overlay_manifest['overlay_root'],'download_file_count':len(metadata_checks),'download_files':metadata_checks,'package_verification':'PACKAGE_VERIFICATION.json'})
print(json.dumps({'decision':'PASS' if package_ok else 'FAIL','release_root':complete_manifest['release_root'],'overlay_root':overlay_manifest['overlay_root'],'packages':package_records},indent=2))
if not package_ok: raise SystemExit(1)
