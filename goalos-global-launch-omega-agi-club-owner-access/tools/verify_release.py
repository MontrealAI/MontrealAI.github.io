from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
errors=[]
for line in (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
    if not line.strip(): continue
    digest,rel=line.split('  ',1); p=root/rel
    if not p.exists(): errors.append(f'missing {rel}'); continue
    got=hashlib.sha256(p.read_bytes()).hexdigest()
    if got!=digest: errors.append(f'checksum mismatch {rel}')
m=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
for rec in m['files']:
    p=root/rec['path']
    if not p.exists(): errors.append(f'manifest missing {rec["path"]}'); continue
    if p.stat().st_size!=rec['size']: errors.append(f'manifest size {rec["path"]}')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=rec['sha256']: errors.append(f'manifest hash {rec["path"]}')
print('PASS' if not errors else 'FAIL')
for e in errors: print(e)
sys.exit(1 if errors else 0)
