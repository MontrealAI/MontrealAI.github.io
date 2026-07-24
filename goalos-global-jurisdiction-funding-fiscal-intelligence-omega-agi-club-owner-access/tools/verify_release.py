#!/usr/bin/env python3
from pathlib import Path
import hashlib,sys
root=Path(__file__).resolve().parents[1]
ledger=root/'SHA256SUMS.txt'
errors=[]
for line in ledger.read_text().splitlines():
    if not line.strip(): continue
    digest,rel=line.split('  ',1)
    p=root/rel
    if not p.exists(): errors.append(f'missing: {rel}'); continue
    got=hashlib.sha256(p.read_bytes()).hexdigest()
    if got!=digest: errors.append(f'mismatch: {rel}')
print('PASS' if not errors else 'FAIL')
for e in errors: print(e)
sys.exit(1 if errors else 0)
