from pathlib import Path
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup
import json, subprocess, sys
ROOT = Path(__file__).resolve().parents[1]
issues = []
for page in sorted(ROOT.rglob('*.html')):
    soup = BeautifulSoup(page.read_text('utf-8', errors='replace'), 'html.parser')
    rel = page.relative_to(ROOT).as_posix()
    ids = [x.get('id') for x in soup.find_all(attrs={'id': True})]
    duplicate = sorted({x for x in ids if ids.count(x) > 1})
    if duplicate: issues.append(f'{rel}: duplicate ids {duplicate}')
    if len(soup.find_all('h1')) != 1: issues.append(f'{rel}: expected one h1')
    if not soup.html or not soup.html.get('lang'): issues.append(f'{rel}: missing html lang')
    for image in soup.find_all('img'):
        if image.get('alt') is None: issues.append(f'{rel}: image missing alt')
    for tag, attr in [('img','src'),('script','src'),('link','href'),('a','href'),('source','srcset')]:
        for element in soup.find_all(tag):
            value = element.get(attr)
            if not value: continue
            values = [value] if attr != 'srcset' else [x.strip().split()[0] for x in value.split(',') if x.strip()]
            for candidate in values:
                if candidate.startswith(('http:','https:','mailto:','tel:','data:','blob:','#','javascript:','//')): continue
                path = unquote(urlparse(candidate).path)
                if not path: continue
                target = (ROOT / path.lstrip('/')) if path.startswith('/') else (page.parent / path)
                try:
                    target = target.resolve(); target.relative_to(ROOT.resolve())
                except Exception:
                    continue
                if not target.exists(): issues.append(f'{rel}: missing local reference {candidate}')
for path in sorted(ROOT.rglob('*.json')):
    try: json.loads(path.read_text('utf-8'))
    except Exception as exc: issues.append(f'{path.relative_to(ROOT)}: invalid JSON: {exc}')
for path in sorted(ROOT.rglob('*.js')):
    result = subprocess.run(['node','--check',str(path)], capture_output=True, text=True)
    if result.returncode: issues.append(f'{path.relative_to(ROOT)}: invalid JavaScript')
for path in ROOT.rglob('*'):
    if not path.is_file(): continue
    name = path.name.lower()
    if path.suffix.lower() in {'.pem','.key','.p12','.pfx'} or any(x in name for x in ['publisher_vault','private-vault','private_vault','seed-phrase','mnemonic.txt']):
        issues.append(f'Forbidden public filename: {path.relative_to(ROOT)}')
if issues:
    print('\n'.join(issues)); sys.exit(1)
print(f'PASS: {len(list(ROOT.rglob("*.html")))} HTML surfaces validated.')
