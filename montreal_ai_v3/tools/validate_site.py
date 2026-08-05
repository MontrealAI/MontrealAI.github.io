from pathlib import Path
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup
from collections import Counter
import json, re, subprocess, sys

ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
issues=[]; htmls=sorted(ROOT.rglob('*.html')); release_only=set()
for page in htmls:
    rel=page.relative_to(ROOT).as_posix(); text=page.read_text('utf-8',errors='replace'); soup=BeautifulSoup(text,'html.parser')
    expected='fr-CA' if rel.startswith('fr/') else 'en-CA'
    if not soup.html or soup.html.get('lang')!=expected: issues.append(f'{rel}: expected lang={expected}')
    if len(soup.find_all('h1'))!=1: issues.append(f'{rel}: expected exactly one h1')
    if not soup.title or not soup.title.get_text(strip=True): issues.append(f'{rel}: missing title')
    desc=soup.find('meta',attrs={'name':re.compile('^description$',re.I)})
    if not desc or not desc.get('content','').strip(): issues.append(f'{rel}: missing description')
    if rel not in release_only:
        canonical=soup.find('link',rel=lambda x:x and 'canonical' in x)
        if not canonical or not canonical.get('href','').startswith('https://montrealai.github.io/'): issues.append(f'{rel}: invalid canonical')
        hlangs={x.get('hreflang') for x in soup.find_all('link',attrs={'hreflang':True})}
        if not {'en-CA','fr-CA','x-default'}.issubset(hlangs): issues.append(f'{rel}: incomplete hreflang')
    ids=[x.get('id') for x in soup.find_all(attrs={'id':True})]
    dup=[x for x,c in Counter(ids).items() if c>1]
    if dup: issues.append(f'{rel}: duplicate ids {dup[:6]}')
    for image in soup.find_all('img'):
        if image.get('alt') is None: issues.append(f'{rel}: image missing alt')
    for button in soup.find_all('button'):
        if not (button.get('aria-label') or button.get('title') or button.get_text(' ',strip=True)): issues.append(f'{rel}: unnamed button')
    for tag,attr in [('img','src'),('script','src'),('link','href'),('a','href'),('source','srcset')]:
        for el in soup.find_all(tag):
            value=el.get(attr)
            if not value: continue
            values=[value] if attr!='srcset' else [x.strip().split()[0] for x in value.split(',') if x.strip()]
            for candidate in values:
                if candidate.startswith(('http:','https:','mailto:','tel:','data:','blob:','#','javascript:','//')): continue
                raw=unquote(urlparse(candidate).path)
                if not raw: continue
                target=(ROOT/raw.lstrip('/')) if raw.startswith('/') else (page.parent/raw)
                try: target=target.resolve(); target.relative_to(ROOT)
                except Exception: continue
                if not target.exists(): issues.append(f'{rel}: missing local reference {candidate}')
    if ('info@'+'quebec.ai') in text: issues.append(f'{rel}: superseded contact address')
    if re.search(r'\b(?:lorem ipsum|placeholder text|coming soon|under construction|asi god)\b',text,re.I): issues.append(f'{rel}: placeholder/internal phrase')
public_htmls=[p for p in htmls if p.relative_to(ROOT).as_posix() not in release_only]
ens={p.relative_to(ROOT).as_posix() for p in public_htmls if not p.relative_to(ROOT).as_posix().startswith('fr/')}
frs={p.relative_to(ROOT).as_posix()[3:] for p in public_htmls if p.relative_to(ROOT).as_posix().startswith('fr/')}
if ens!=frs: issues.append(f'Bilingual path mismatch: missing_fr={sorted(ens-frs)}, missing_en={sorted(frs-ens)}')
if len(public_htmls)!=136 or len(ens)!=68 or len(htmls)!=136: issues.append(f'Unexpected surface count: public_html={len(public_htmls)}, total_html={len(htmls)}, pairs={len(ens)}')
for path in sorted(ROOT.rglob('*.json')):
    try: json.loads(path.read_text('utf-8'))
    except Exception as exc: issues.append(f'{path.relative_to(ROOT)}: invalid JSON: {exc}')
for path in sorted(ROOT.rglob('*.js')):
    result=subprocess.run(['node','--check',str(path)],capture_output=True,text=True)
    if result.returncode: issues.append(f'{path.relative_to(ROOT)}: invalid JavaScript: {result.stderr[:200]}')
for path in ROOT.rglob('*'):
    if not path.is_file(): continue
    name=path.name.lower()
    if path.suffix.lower() in {'.pem','.key','.p12','.pfx'} or any(x in name for x in ['publisher_vault','private-vault','private_vault','seed-phrase','mnemonic.txt']): issues.append(f'Forbidden public filename: {path.relative_to(ROOT)}')
    if path.suffix.lower() in {'.html','.js','.json','.md','.txt','.cff','.xml','.yml','.yaml','.css','.svg'}:
        try:
            if ('info@'+'quebec.ai') in path.read_text('utf-8'): issues.append(f'{path.relative_to(ROOT)}: superseded contact address')
        except Exception: pass
for required in ['index.html','fr/index.html','404.html','fr/404.html','robots.txt','sitemap.xml','manifest.webmanifest','VERSION.json','README.md','.nojekyll','ai-policy.txt','llms.txt','.github/workflows/website-qa.yml']:
    if not (ROOT/required).exists(): issues.append(f'Missing required release file: {required}')
if issues:
    print('\n'.join(issues)); sys.exit(1)
print(f'PASS: {len(htmls)} total HTML surfaces / {len(ens)} bilingual public route pairs validated.')
