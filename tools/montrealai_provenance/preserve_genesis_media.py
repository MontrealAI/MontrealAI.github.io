from __future__ import annotations
import csv, hashlib, io, json, os, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import requests
from PIL import Image, ImageOps

INPUT=Path(os.environ.get('ITEMS_CSV','baseline/normalized/items.csv'))
OUT=Path(os.environ.get('OUTPUT_DIR','genesis_556_media'))
WORKERS=int(os.environ.get('WORKERS','8'))
TIMEOUT=int(os.environ.get('HTTP_TIMEOUT','60'))
UA='MONTREAL.AI Genesis 556 Preservation/4.0 (+https://montreal.ai)'

def sha256_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def sha256_file(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()

def fetch_one(r:dict[str,str])->dict[str,Any]:
 n=r['artwork_number'].zfill(3);url=r['image_url'];last=''
 for attempt in range(8):
  try:
   resp=requests.get(url,headers={'User-Agent':UA,'Accept':'image/avif,image/webp,image/apng,image/*,*/*;q=0.8'},timeout=TIMEOUT)
   resp.raise_for_status();raw=resp.content
   if len(raw)<1000:raise RuntimeError(f'body too small: {len(raw)}')
   im=Image.open(io.BytesIO(raw));im.load();fmt=(im.format or 'JPEG').upper();ext={'JPEG':'jpg','PNG':'png','WEBP':'webp','GIF':'gif'}.get(fmt,'bin')
   raw_path=OUT/'media/original'/f'{n}.{ext}';raw_path.parent.mkdir(parents=True,exist_ok=True);raw_path.write_bytes(raw)
   rgb=ImageOps.exif_transpose(im).convert('RGB')
   web=rgb.copy();web.thumbnail((1400,1400),Image.Resampling.LANCZOS);web_path=OUT/'media/web'/f'{n}.webp';web_path.parent.mkdir(parents=True,exist_ok=True);web.save(web_path,'WEBP',quality=91,method=6)
   thumb=ImageOps.fit(rgb,(360,360),method=Image.Resampling.LANCZOS);thumb_path=OUT/'media/thumbs'/f'{n}.webp';thumb_path.parent.mkdir(parents=True,exist_ok=True);thumb.save(thumb_path,'WEBP',quality=84,method=6)
   meta={k:r.get(k,'') for k in r};meta.update({'artwork_number':n,'downloaded_source_url':url,'original_media_path':raw_path.relative_to(OUT).as_posix(),'web_media_path':web_path.relative_to(OUT).as_posix(),'thumbnail_path':thumb_path.relative_to(OUT).as_posix(),'original_sha256':sha256_bytes(raw),'web_sha256':sha256_file(web_path),'thumbnail_sha256':sha256_file(thumb_path),'original_bytes':len(raw),'web_bytes':web_path.stat().st_size,'thumbnail_bytes':thumb_path.stat().st_size,'width':rgb.width,'height':rgb.height,'detected_format':fmt,'http_content_type':resp.headers.get('content-type',''),'status':'PASS'})
   mp=OUT/'metadata'/f'{n}.json';mp.parent.mkdir(parents=True,exist_ok=True);mp.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
   return meta
  except Exception as e:
   last=str(e);time.sleep(min(30,1.5*(2**attempt)))
 return {'artwork_number':n,'image_url':url,'status':'FAIL','error':last}

def main()->int:
 if OUT.exists():shutil.rmtree(OUT)
 OUT.mkdir(parents=True)
 with INPUT.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=556:raise RuntimeError(f'Expected 556 item rows, got {len(rows)}')
 # OpenSea has two raw titles #316. The token whose embedded nonce is 322 occupies canonical position #317.
 for r in rows:
  nonce=(int(r['token_id_decimal'])>>40)&((1<<56)-1)
  r['raw_opensea_artwork_number']=r['artwork_number']
  if nonce==322:
   r['artwork_number']='317';r['canonical_name']='Crypto AI Art #317';r['canonicalization_note']='Raw OpenSea title #316 preserved; canonical position #317 assigned from token identity.'
  else:r['canonical_name']=f"Crypto AI Art #{r['artwork_number'].zfill(3)}"
 if {r['artwork_number'].zfill(3) for r in rows}!={f'{i:03d}' for i in range(556)}:raise RuntimeError('Canonical identity validation failed')
 results=[]
 with ThreadPoolExecutor(max_workers=WORKERS) as ex:
  fut={ex.submit(fetch_one,r):r['artwork_number'] for r in rows}
  for i,f in enumerate(as_completed(fut),1):
   x=f.result();results.append(x);print(f"{i}/556 {x['artwork_number']} {x['status']}",flush=True)
 results.sort(key=lambda x:int(x['artwork_number']))
 fields=sorted({k for r in results for k in r})
 with (OUT/'MEDIA_MANIFEST.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
 (OUT/'MEDIA_MANIFEST.json').write_text(json.dumps({'schema':'montrealai.genesis-556.media-preservation.v1','records':results},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 passed=[r for r in results if r['status']=='PASS'];qa={'schema':'montrealai.genesis-556.media-preservation.qa.v1','records':len(results),'passed':len(passed),'failed':len(results)-len(passed),'unique_artwork_numbers':len({r['artwork_number'] for r in results}),'unique_original_hashes':len({r.get('original_sha256') for r in passed}),'total_original_bytes':sum(int(r['original_bytes']) for r in passed),'total_web_bytes':sum(int(r['web_bytes']) for r in passed),'total_thumbnail_bytes':sum(int(r['thumbnail_bytes']) for r in passed),'acceptance_gate':'PASS' if len(passed)==556 else 'FAIL'}
 (OUT/'QA_REPORT.json').write_text(json.dumps(qa,indent=2)+'\n')
 all_files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
 (OUT/'SHA256SUMS').write_text('\n'.join(f'{sha256_file(p)}  {p.relative_to(OUT).as_posix()}' for p in all_files)+'\n')
 print(json.dumps(qa,indent=2));return 0 if qa['acceptance_gate']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
