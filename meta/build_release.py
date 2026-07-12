from pathlib import Path
import hashlib, json, zipfile, os, sys
ROOT=Path('/mnt/data/GoalOS_Partner_MASTERCLASS_MASTERPIECE_v3')
ZIP=Path('/mnt/data/GoalOS_Partner_MASTERCLASS_MASTERPIECE_v3_EVERYTHING.zip')
EXCLUDE_PREFIXES=(
 'qa/deck_render/','qa/guide_render/','qa/guide_render_final/','qa/complete_book_spotcheck_render/','qa/deck_pdf/'
)
EXCLUDE_FILES={
 'ARTIFACT_INDEX.json','SHA256SUMS.txt','qa/complete_book_spotcheck.pdf',
 'meta/launcher.png','meta/launcher3.png','meta/launcher-mobile.png'
}

def include(rel:str)->bool:
    if rel in EXCLUDE_FILES:return False
    if any(rel.startswith(p) for p in EXCLUDE_PREFIXES):return False
    return True

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def cat(rel):
    top=rel.split('/',1)[0]
    if '/' not in rel:
        return 'root'
    return top

# Artifacts before index/checksum
files=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file():
        rel=p.relative_to(ROOT).as_posix()
        if include(rel):
            files.append({'path':rel,'category':cat(rel),'bytes':p.stat().st_size,'sha256':sha(p)})
index={
 'schema':'goalos.masterpieceArtifactIndex.v1',
 'release':'GOS-PM-MASTERPIECE-2026.07-v3.0',
 'generatedAt':'2026-07-12',
 'root':'GoalOS_Partner_MASTERCLASS_MASTERPIECE_v3',
 'scope':'Files included in the EVERYTHING release archive, excluding bulk page-render QA intermediates.',
 'counts':{'files':len(files),'bytes':sum(x['bytes'] for x in files)},
 'primaryArtifacts':[
  'START_HERE.html',
  'GoalOS_Partner_MASTERCLASS_MASTERPIECE.html',
  'GoalOS_Partner_MASTERCLASS_MASTERPIECE_Complete_Book.pdf',
  'presentation/GoalOS_Partner_MASTERCLASS_MASTERPIECE_Executive_Deck.pptx',
  'presentation/GoalOS_Partner_MASTERCLASS_MASTERPIECE_Executive_Deck.pdf',
  'guides/GoalOS_Partner_MASTERCLASS_MASTERPIECE_Operating_Guide.docx',
  'guides/GoalOS_Partner_MASTERCLASS_MASTERPIECE_Operating_Guide.pdf',
  'meta/VALIDATION_REPORT.md'
 ],
 'artifacts':files
}
(ROOT/'ARTIFACT_INDEX.json').write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
# checksums include index but exclude checksum itself
paths=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file():
        rel=p.relative_to(ROOT).as_posix()
        if rel=='SHA256SUMS.txt':continue
        if include(rel) or rel=='ARTIFACT_INDEX.json':paths.append((rel,p))
(ROOT/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {rel}\n' for rel,p in paths),encoding='utf-8')
# zip includes index + checksums
if ZIP.exists():ZIP.unlink()
with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT).as_posix()
        if include(rel) or rel in ('ARTIFACT_INDEX.json','SHA256SUMS.txt'):
            z.write(p,arcname=f'{ROOT.name}/{rel}')
print(json.dumps({'zip':str(ZIP),'files':len(zipfile.ZipFile(ZIP).infolist()),'bytes':ZIP.stat().st_size,'sha256':sha(ZIP)},indent=2))
