#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import base64, datetime, hashlib, json, re, sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'qa'/'ACCESSIBILITY_AND_CONTENT_QA_REPORT.json'
EXCLUDE={'START_HERE.html'}
EXCLUDE_PREFIXES=('goalos-documents/','BUILD_SOURCE/','qa/')
PAGES=sorted(p for p in ROOT.rglob('*.html') if p.relative_to(ROOT).as_posix() not in EXCLUDE and not p.relative_to(ROOT).as_posix().startswith(EXCLUDE_PREFIXES))
errors=[]; warnings=[]; page_results=[]

def text_of(node): return ' '.join(node.stripped_strings) if node else ''

def expected_urls(rel:str):
    if rel.startswith('en/'):
        tail=rel[3:]
        en='https://montrealai.github.io/en/' if tail=='index.html' else 'https://montrealai.github.io/en/'+tail.removesuffix('index.html')
        fr='https://montrealai.github.io/' if tail=='index.html' else 'https://montrealai.github.io/'+tail.removesuffix('index.html')
    else:
        tail=rel
        fr='https://montrealai.github.io/' if tail=='index.html' else 'https://montrealai.github.io/'+tail.removesuffix('index.html')
        en='https://montrealai.github.io/en/' if tail=='index.html' else 'https://montrealai.github.io/en/'+tail.removesuffix('index.html')
    return fr,en

for p in PAGES:
    rel=p.relative_to(ROOT).as_posix(); html=p.read_text('utf-8'); soup=BeautifulSoup(html,'html.parser')
    local=[]
    def issue(msg, warn=False):
        (warnings if warn else errors).append(f'{rel}: {msg}'); local.append(('warning' if warn else 'error')+': '+msg)
    if not soup.html or soup.html.get('lang') not in {'fr-CA','en-CA'}: issue('invalid or missing language')
    if not soup.find('meta',attrs={'name':'viewport'}): issue('missing viewport meta')
    if not soup.title or not text_of(soup.title): issue('missing title')
    desc=soup.find('meta',attrs={'name':'description'})
    if not desc or len(desc.get('content','').strip())<60: issue('missing or short meta description')
    if len(soup.find_all('h1'))!=1: issue(f'expected one H1, found {len(soup.find_all("h1"))}')
    if not soup.find('main'): issue('missing main landmark')
    if not soup.find('footer'): issue('missing footer landmark')
    if not soup.find('a',class_='skip-link'): issue('missing skip link')
    ids=[x.get('id') for x in soup.find_all(id=True)]
    dup=sorted({x for x in ids if ids.count(x)>1})
    if dup: issue('duplicate IDs: '+', '.join(dup[:10]))
    for img in soup.find_all('img'):
        if not img.has_attr('alt'): issue(f'image missing alt: {img.get("src","")}')
    for button in soup.find_all('button'):
        if not text_of(button) and not button.get('aria-label') and not button.get('title'): issue('button has no accessible name')
    for a in soup.find_all('a',href=True):
        if not text_of(a) and not a.get('aria-label') and not a.find('img',alt=True): issue(f'link has no accessible name: {a.get("href")}')
        if a.get('target')=='_blank' and 'noopener' not in (a.get('rel') or []): issue(f'target blank lacks noopener: {a.get("href")}')
    for table in soup.find_all('table'):
        if not table.find('th'): issue('data table lacks header cells',warn=True)
    # Heading order (allow repeated same level, no jumps > 1)
    levels=[int(h.name[1]) for h in soup.find_all(re.compile('^h[1-6]$'))]
    for a,b in zip(levels,levels[1:]):
        if b>a+1: issue(f'heading level jumps from H{a} to H{b}',warn=True); break
    # CSP and structured-data hash
    csp=soup.find('meta',attrs={'http-equiv':lambda x:x and x.lower()=='content-security-policy'})
    if not csp: issue('missing CSP')
    else:
        cv=csp.get('content','')
        required=["default-src 'self'","connect-src 'none'","object-src 'none'","form-action 'none'","frame-ancestors 'none'","base-uri 'self'"]
        for item in required:
            if item not in cv: issue(f'CSP missing {item}')
        inline=[s for s in soup.find_all('script') if not s.get('src')]
        executable=[s for s in inline if s.get('type','').lower() not in {'application/ld+json','application/json'}]
        if executable: issue('unexpected inline executable script')
        for s in soup.find_all('script',attrs={'type':'application/ld+json'}):
            token="'sha256-"+base64.b64encode(hashlib.sha256((s.string or '').encode()).digest()).decode()+"'"
            if token not in cv: issue('CSP does not authorize exact JSON-LD hash')
    # Runtime dependency / privacy-minimization checks
    if soup.find('form'): issue('public static site contains a form')
    if soup.find('iframe'): issue('public static site contains an iframe')
    for tag,attr in [('script','src'),('link','href'),('img','src')]:
        for n in soup.find_all(tag):
            u=n.get(attr,'')
            if u.startswith(('http://','https://')) and urlparse(u).netloc not in {'montrealai.github.io','www.montrealai.github.io'}:
                if tag=='link' and n.get('rel') and not any(x in n.get('rel') for x in ['stylesheet','icon','manifest']): continue
                issue(f'external runtime asset: {u}')
    if re.search(r'\bhttp://',html,re.I): issue('mixed/insecure HTTP reference')
    for attr in ('onclick','onload','onerror','onmouseover'):
        if soup.find(attrs={attr:True}): issue(f'inline event handler present: {attr}')
    # Canonical / hreflang exactness
    fr,en=expected_urls(rel)
    expected=en if rel.startswith('en/') else fr
    canon=soup.find('link',rel='canonical')
    if not canon or canon.get('href')!=expected: issue(f'canonical mismatch; expected {expected}')
    alts={x.get('hreflang'):x.get('href') for x in soup.find_all('link',rel='alternate')}
    if alts.get('fr-CA')!=fr or alts.get('en-CA')!=en or alts.get('x-default')!=fr: issue('hreflang target mismatch')
    # No explicit achieved frontier claims in the public website. The claim-boundary pages
    # necessarily quote prohibited language in order to forbid it.
    visible=text_of(soup.body)
    forbidden=[] if 'claim-boundary/' in rel else [r'\bachieved\s+AGI\b',r'\bAGI\s+achieved\b',r'\bachieved\s+ASI\b',r'\bsuperintelligen(?:ce|t)\b']
    for pat in forbidden:
        m=re.search(pat,visible,re.I)
        if m:
            prefix=visible[max(0,m.start()-180):m.start()].lower()
            allowed_context=('do not publish','must not claim','does not claim','no claim of','ne pas publier','ne prétend pas','aucune prétention','pas atteint')
            if not any(x in prefix for x in allowed_context):
                issue(f'forbidden unsupported frontier claim matches {pat}')
    page_results.append({'page':rel,'lang':soup.html.get('lang','') if soup.html else '', 'errors':[x for x in local if x.startswith('error:')], 'warnings':[x for x in local if x.startswith('warning:')]})

# Sitemap coverage
sitemap=(ROOT/'sitemap.xml').read_text('utf-8')
for p in PAGES:
    rel=p.relative_to(ROOT).as_posix()
    if rel in {'404.html','en/404.html'}: continue
    _,en=expected_urls(rel); fr,_=expected_urls(rel)
    expected=en if rel.startswith('en/') else fr
    if expected not in sitemap: errors.append(f'sitemap missing {expected}')
# Source register quality
src=json.loads((ROOT/'source-register.json').read_text('utf-8'))
if src.get('version')!='3.0.0': errors.append('source-register.json version is not 3.0.0')
if len(src.get('sources',[]))<10: errors.append('source register has fewer than 10 sources')
for s in src.get('sources',[]):
    if not str(s.get('url','')).startswith('https://'): errors.append(f'source URL is not HTTPS: {s.get("title")}')
    if not (s.get('accessed') or s.get('date')): errors.append(f'source has no date: {s.get("title_en") or s.get("title")}')
# Favicon / manifest assets
manifest=json.loads((ROOT/'manifest.webmanifest').read_text('utf-8'))
for icon in manifest.get('icons',[]):
    q=ROOT/str(icon.get('src','')).lstrip('/')
    if not q.exists(): errors.append(f'manifest icon missing: {icon.get("src")}')

report={
 'schema':'montrealai.goalos.accessibility-content-qa.v3',
 'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'public_pages':len(PAGES),
 'errors':errors,
 'warnings':warnings,
 'checks':{
  'semantic_structure':not any('landmark' in x or 'H1' in x for x in errors),
  'accessible_names':not any('accessible name' in x or 'missing alt' in x for x in errors),
  'csp_hash_integrity':not any('CSP' in x for x in errors),
  'canonical_hreflang':not any('canonical' in x or 'hreflang' in x for x in errors),
  'privacy_minimized_surface':not any('form' in x or 'iframe' in x or 'runtime asset' in x for x in errors),
  'source_register':not any('source' in x.lower() for x in errors),
  'sitemap_coverage':not any('sitemap' in x for x in errors),
  'unsupported_frontier_claim_scan':not any('frontier claim' in x for x in errors),
 },
 'page_results':page_results
}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n','utf-8')
print(f'ACCESSIBILITY / CONTENT QA: {"PASS" if not errors else "FAIL"} · {len(PAGES)} pages · {len(errors)} errors · {len(warnings)} warnings')
if errors:
 print('\n'.join(errors)); sys.exit(1)
