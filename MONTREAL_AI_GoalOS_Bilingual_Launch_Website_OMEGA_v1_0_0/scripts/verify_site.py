#!/usr/bin/env python3
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse
import json, re, sys, datetime
ROOT=Path(__file__).resolve().parents[1]
PUBLIC_EXCLUDE={Path('goalos-documents/products/GoalOS_Autonomous_Money_Machine_OMEGA_Civilization_Value_Ceiling_v10_0_0.html')}
ORIGIN='https://montrealai.github.io'
class PageParser(HTMLParser):
 def __init__(self):
  super().__init__(); self.links=[]; self.scripts=[]; self.styles=[]; self.images=[]; self.forms=0; self.iframes=0; self.h1=0; self.lang=''; self.title=''; self.canonical=[]; self.alternates=[]; self.csp=[]; self._title=False
 def handle_starttag(self,t,a):
  d=dict(a)
  if t=='html': self.lang=d.get('lang','')
  if t=='h1': self.h1+=1
  if t=='form': self.forms+=1
  if t=='iframe': self.iframes+=1
  if t=='script' and d.get('src'): self.scripts.append(d['src'])
  if t=='img' and d.get('src'): self.images.append(d['src'])
  if t=='link':
   if d.get('rel')=='canonical': self.canonical.append(d.get('href',''))
   if d.get('rel')=='alternate': self.alternates.append((d.get('hreflang',''),d.get('href','')))
   if d.get('rel')=='stylesheet' and d.get('href'): self.styles.append(d['href'])
  if t=='meta' and d.get('http-equiv','').lower()=='content-security-policy': self.csp.append(d.get('content',''))
  for key in ('href','src'):
   u=d.get(key)
   if u:self.links.append(u)
  if t=='title': self._title=True
 def handle_endtag(self,t):
  if t=='title': self._title=False
 def handle_data(self,data):
  if self._title:self.title+=data

def is_local_runtime(u):
 if u.startswith(('data:','#','mailto:','tel:')): return True
 if u.startswith('/'):
  return True
 if u.startswith(('http://','https://')):
  p=urlparse(u); return p.netloc in {'montrealai.github.io','www.montrealai.github.io'}
 return True

def local_target(f,u):
 if u.startswith(('http:','https:','mailto:','tel:','#','data:','javascript:')): return None
 base=(ROOT/u.lstrip('/')) if u.startswith('/') else (f.parent/u)
 q=Path(str(base).split('#')[0].split('?')[0])
 if q.is_dir():q=q/'index.html'
 return q

errors=[]; warnings=[]; pages=[]
all_html=sorted(p for p in ROOT.rglob('*.html') if p.relative_to(ROOT) not in PUBLIC_EXCLUDE and '_Downloads' not in p.as_posix())
for f in all_html:
 rel=f.relative_to(ROOT)
 p=PageParser(); p.feed(f.read_text(encoding='utf-8'))
 pages.append(rel.as_posix())
 if not p.lang:errors.append(f'{rel}: missing html lang')
 if p.h1!=1:errors.append(f'{rel}: expected one h1, found {p.h1}')
 if not p.title.strip():errors.append(f'{rel}: missing title')
 if len(p.canonical)!=1:errors.append(f'{rel}: expected one canonical, found {len(p.canonical)}')
 langs={x[0] for x in p.alternates}
 if not {'fr-CA','en-CA','x-default'}.issubset(langs):errors.append(f'{rel}: incomplete hreflang set {sorted(langs)}')
 if len(p.csp)!=1:errors.append(f'{rel}: missing or duplicate CSP')
 elif "connect-src 'none'" not in p.csp[0] or "form-action 'none'" not in p.csp[0]:errors.append(f'{rel}: CSP is not fail-closed for connections/forms')
 if p.forms:errors.append(f'{rel}: public site must not contain server forms')
 if p.iframes:errors.append(f'{rel}: public site must not contain iframes')
 for u in p.scripts+p.styles+p.images:
  if not is_local_runtime(u):errors.append(f'{rel}: external runtime dependency {u}')
 for u in p.links:
  q=local_target(f,u)
  if q is not None and not q.exists():errors.append(f'{rel} -> missing {u}')
# Bilingual symmetry
fr=set()
en=set()
for rel in pages:
 if rel.startswith('en/'):
  en.add(rel[3:])
 else:
  fr.add(rel)
expected_exceptions=set()
missing_en=sorted(fr-en-expected_exceptions); missing_fr=sorted(en-fr-expected_exceptions)
if missing_en:errors.append('French pages without English mirror: '+', '.join(missing_en))
if missing_fr:errors.append('English pages without French mirror: '+', '.join(missing_fr))
# Required legal surfaces
legal={'index.html','terms/index.html','privacy/index.html','claim-boundary/index.html','regulatory/index.html','acceptable-use/index.html','web3-risk/index.html','ai-transparency/index.html','security/index.html','accessibility/index.html','enterprise-services/index.html'}
for prefix in ('goalos-legal','en/goalos-legal'):
 have={str(p.relative_to(ROOT/ prefix)) for p in (ROOT/prefix).rglob('*.html')}
 miss=sorted(legal-have)
 if miss:errors.append(f'{prefix}: missing legal pages '+', '.join(miss))
# JSON and release surface
for f in ROOT.rglob('*.json'):
 try:json.loads(f.read_text(encoding='utf-8'))
 except Exception as e:errors.append(f'{f.relative_to(ROOT)}: invalid JSON: {e}')
for needed in ['index.html','en/index.html','404.html','en/404.html','robots.txt','sitemap.xml','manifest.webmanifest','ai.txt','humans.txt','source-register.json','.nojekyll']:
 if not (ROOT/needed).exists():errors.append('missing required release file '+needed)
# Same-origin absolute canonical links should target GitHub Pages host
for f in all_html:
 text=f.read_text(encoding='utf-8')
 if 'https://montrealai.github.io' not in text:errors.append(f'{f.relative_to(ROOT)}: GitHub Pages canonical origin absent')
# Basic secret and placeholder scan on public source, excluding binary documents
secret_re=re.compile(r'(?i)(BEGIN [A-Z ]*PRIVATE KEY|aws_secret_access_key\s*=|api[_-]?key\s*[:=]\s*["\'][A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9]{20,})')
for f in ROOT.rglob('*'):
 if not f.is_file() or f.suffix.lower() not in {'.html','.js','.css','.json','.md','.txt','.yml','.yaml','.xml'}:continue
 try:text=f.read_text(encoding='utf-8')
 except Exception:continue
 if secret_re.search(text):errors.append(f'{f.relative_to(ROOT)}: potential credential material')
report={'schema':'montrealai.goalos.static-verification.v1','generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'public_pages':len(all_html),'french_pages':len(fr),'english_pages':len(en),'errors':errors,'warnings':warnings,'checks':{'bilingual_symmetry':not missing_en and not missing_fr,'legal_surface_complete':not any('missing legal pages' in x for x in errors),'no_external_runtime_dependencies':not any('external runtime dependency' in x for x in errors),'no_server_forms':not any('server forms' in x for x in errors),'json_parse':not any('invalid JSON' in x for x in errors),'internal_links':not any(' -> missing ' in x for x in errors)}}
(ROOT/'qa').mkdir(exist_ok=True)
(ROOT/'qa'/'STATIC_VERIFICATION_REPORT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
if errors:
 print('STATIC VERIFICATION: FAIL')
 print('\n'.join(errors));sys.exit(1)
print(f'STATIC VERIFICATION: PASS · {len(all_html)} pages · {len(fr)} FR · {len(en)} EN')
