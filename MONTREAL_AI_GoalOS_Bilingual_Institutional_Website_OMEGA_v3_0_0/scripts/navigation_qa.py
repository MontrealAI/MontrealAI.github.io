#!/usr/bin/env python3
from __future__ import annotations
import contextlib, datetime, http.server, json, pathlib, socket, socketserver, threading, urllib.error, urllib.parse, urllib.request
from bs4 import BeautifulSoup

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'qa'/'NAVIGATION_QA_REPORT.json'
EXCLUDE={'START_HERE.html'}
EXCLUDE_PREFIXES=('goalos-documents/','BUILD_SOURCE/','qa/')
PAGES=sorted(
    p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.html')
    if p.relative_to(ROOT).as_posix() not in EXCLUDE
    and not p.relative_to(ROOT).as_posix().startswith(EXCLUDE_PREFIXES)
)

def internal_target(page:pathlib.Path, href:str):
    if not href or href.startswith(('#','mailto:','tel:','javascript:','data:','http://','https://')):
        return None
    raw=urllib.parse.urlsplit(href).path
    if not raw: return None
    target=(ROOT/raw.lstrip('/')) if raw.startswith('/') else (page.parent/raw)
    target=target.resolve()
    try: target.relative_to(ROOT.resolve())
    except ValueError: return ('unsafe',target)
    if target.is_dir(): target=target/'index.html'
    return ('local',target)

link_results=[]; issues=[]; internal_links=0
for rel in PAGES:
    page=ROOT/rel
    soup=BeautifulSoup(page.read_text('utf-8'),'html.parser')
    for a in soup.find_all('a',href=True):
        href=a.get('href','')
        resolved=internal_target(page,href)
        if not resolved: continue
        internal_links+=1
        kind,target=resolved
        ok=kind=='local' and target.is_file()
        explicit_index=(not href.endswith('/'))
        no_root_absolute=not href.startswith('/')
        rec={'source':rel,'href':href,'target':target.relative_to(ROOT).as_posix() if target.exists() else str(target),'exists':ok,'explicit_index_or_file':explicit_index,'relative_local_safe':no_root_absolute}
        link_results.append(rec)
        if not ok: issues.append({**rec,'error':'target missing or unsafe'})
        if not explicit_index: issues.append({**rec,'error':'directory-only href is not file-preview safe'})
        if not no_root_absolute: issues.append({**rec,'error':'root-absolute href is not file-preview safe'})

# Verify every public page through a real local HTTP server using Python's HTTP stack.
class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args): pass
class ReuseServer(socketserver.ThreadingTCPServer):
    allow_reuse_address=True

http_results=[]; http_errors=[]
with contextlib.ExitStack() as stack:
    old_cwd=pathlib.Path.cwd()
    import os
    os.chdir(ROOT)
    stack.callback(os.chdir,old_cwd)
    server=ReuseServer(('127.0.0.1',0),QuietHandler)
    stack.callback(server.server_close)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    stack.callback(server.shutdown)
    port=server.server_address[1]
    for rel in PAGES:
        url=f'http://127.0.0.1:{port}/'+urllib.parse.quote(rel, safe='/')
        try:
            with urllib.request.urlopen(url,timeout=10) as resp:
                body=resp.read()
                ok=resp.status==200 and b'<html' in body[:1000].lower()
                rec={'page':rel,'status':resp.status,'bytes':len(body),'ok':ok}
        except Exception as exc:
            rec={'page':rel,'status':None,'bytes':0,'ok':False,'error':str(exc)}
        http_results.append(rec)
        if not rec['ok']: http_errors.append(rec)

# Representative user journeys: every edge must exist and be retrievable.
journeys={
 'fr_primary':['index.html','goalos/index.html','business-model/index.html','proof-missions/index.html','proof/index.html','contact/index.html','goalos-legal/index.html'],
 'en_primary':['en/index.html','en/goalos/index.html','en/business-model/index.html','en/proof-missions/index.html','en/proof/index.html','en/contact/index.html','en/goalos-legal/index.html'],
 'commercial':['index.html','autonomous-money-machine/index.html','frontier-masterclass/index.html','proof-missions/rfp-to-revenue/index.html','proof-offices/index.html','downloads/index.html'],
 'private_capability':['index.html','agent-identity/index.html','chronicle/index.html','validated-skill-graph/index.html','research/index.html'],
}
journey_results=[]
for name,steps in journeys.items():
    missing=[x for x in steps if not (ROOT/x).is_file()]
    served=[next((r for r in http_results if r['page']==x),None) for x in steps]
    ok=not missing and all(r and r['ok'] for r in served)
    journey_results.append({'journey':name,'steps':steps,'ok':ok,'missing':missing})

report={
 'schema':'montrealai.goalos.navigation-qa.v3',
 'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'method':{
   'file_preview':'static resolution of every internal anchor to an explicit local file; no root-absolute or directory-only internal hrefs permitted',
   'http_preview':'real local HTTP server with every public HTML page fetched and checked for HTTP 200 plus HTML payload',
   'browser_environment_note':'The controlled Chromium runtime blocks file:// and loopback navigation by administrator policy; full browser rendering is tested separately through network-free set_content with the actual release CSS, JavaScript, and images inlined.'
 },
 'public_pages':len(PAGES),
 'internal_links_checked':internal_links,
 'local_path_issues':issues,
 'http_pages_passed':sum(1 for r in http_results if r['ok']),
 'http_page_failures':http_errors,
 'journey_failures':[r for r in journey_results if not r['ok']],
 'status':'PASS' if not issues and not http_errors and all(r['ok'] for r in journey_results) else 'FAIL',
 'journeys':journey_results,
 'http_results':http_results,
 'link_results':link_results,
}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n','utf-8')
print(f"NAVIGATION QA: {report['status']} · {len(PAGES)} pages · {internal_links} internal links · {report['http_pages_passed']} HTTP pages passed")
if report['status']!='PASS': raise SystemExit(1)
