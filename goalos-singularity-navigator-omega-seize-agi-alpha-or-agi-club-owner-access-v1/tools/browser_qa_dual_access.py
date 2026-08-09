from __future__ import annotations
import contextlib, http.server, json, socket, threading, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*a): pass
@contextlib.contextmanager
def server():
    class H(Quiet):
        def __init__(self,*a,**kw): super().__init__(*a,directory=str(ROOT),**kw)
    s=http.server.ThreadingHTTPServer(('127.0.0.1',0),H);t=threading.Thread(target=s.serve_forever,daemon=True);t.start()
    try: yield f'http://127.0.0.1:{s.server_address[1]}'
    finally:s.shutdown();s.server_close()
MOCK="""({balance='0xd3c21bcecceda1000000',chain='0x1',reject=false}={})=>{const handlers={};window.__rpcCalls=[];window.ethereum={request:async({method,params})=>{window.__rpcCalls.push(method);if(method==='eth_chainId')return chain;if(method==='wallet_switchEthereumChain'){chain='0x1';return null;}if(method==='eth_requestAccounts'||method==='eth_accounts')return ['0x1111111111111111111111111111111111111111'];if(method==='eth_getCode')return '0x6001600055';if(method==='eth_call')return balance;if(method==='eth_blockNumber')return '0x123456';if(method==='eth_signTypedData_v4'){if(reject){const e=new Error('rejected');e.code=4001;throw e;}return '0x'+'11'.repeat(65);}if(method==='personal_sign')return '0x'+'22'.repeat(65);throw new Error('unexpected '+method);},on:(n,f)=>(handlers[n]??=[]).push(f),removeListener:()=>{},__emit:(n,v)=>(handlers[n]||[]).forEach(f=>f(v))};}"""
def run():
    report={'release':'v3.3.0-SN5-BI2','tests':[],'consoleErrors':[],'pageErrors':[]}
    def check(name,ok,detail=''):
        report['tests'].append({'name':name,'pass':bool(ok),'detail':detail})
        if not ok: raise AssertionError(name+(': '+detail if detail else ''))
    with server() as base, sync_playwright() as p:
      browser=p.chromium.launch(headless=True)
      for page_name,want_fr in [('index.html',False),('index-en.html',False),('index-fr.html',True)]:
        page=browser.new_page(viewport={'width':1440,'height':1100});page.on('console',lambda m: report['consoleErrors'].append(m.text) if m.type=='error' else None);page.on('pageerror',lambda e:report['pageErrors'].append(str(e)))
        page.add_init_script(f"({MOCK})()")
        page.goto(base+'/'+page_name,wait_until='domcontentloaded');page.wait_for_selector('#goalos-dual-access-card',timeout=15000)
        text=page.locator('#goalos-dual-access-card').inner_text()
        check(page_name+' exact threshold','1,000,000' in text or '1 000 000' in text)
        check(page_name+' exact contract','0xA61a3B3a130a9c20768EEBF97E21515A6046a1fA' in text)
        check(page_name+' language',('Accès' in text) if want_fr else ('Access' in text))
        page.check('#gda-accept');page.check('#gda-boundary');page.click('#gda-verify');page.wait_for_function("sessionStorage.getItem('goalos.sn5.bi2.agialpha.access.receipt')!==null",timeout=15000);page.wait_for_selector('#goalos-dual-access-status',timeout=10000)
        r=page.evaluate("JSON.parse(sessionStorage.getItem('goalos.sn5.bi2.agialpha.access.receipt'))")
        check(page_name+' access class',r['accessClass']=='AGIALPHA_BALANCE_QUALIFIED')
        check(page_name+' authority none',r['authorityCreated']=='None' and r['authority']=='NONE')
        check(page_name+' mainnet',r['chainId']==1)
        check(page_name+' threshold',r['minimumBalance']=='1000000000000000000000000')
        calls=page.evaluate('window.__rpcCalls')
        forbidden={'eth_sendTransaction','eth_sendRawTransaction','eth_signTransaction','wallet_watchAsset'}
        check(page_name+' no transaction methods',not (forbidden & set(calls)),str(calls))
        check(page_name+' app views present',page.locator('section.view,.view[id^="view-"]').count()>0)
        overflow=page.evaluate('document.documentElement.scrollWidth<=document.documentElement.clientWidth+2')
        check(page_name+' no horizontal overflow',overflow)
        page.close()
      # Below threshold.
      page=browser.new_page();page.add_init_script(f"({MOCK})({{balance:'0xd3c21bcecceda0ffffff'}})");page.goto(base+'/index-en.html',wait_until='domcontentloaded');page.wait_for_selector('#goalos-dual-access-card');page.check('#gda-accept');page.check('#gda-boundary');page.click('#gda-verify');page.wait_for_timeout(900);check('below threshold denied',page.evaluate("sessionStorage.getItem('goalos.sn5.bi2.agialpha.access.receipt')===null"));check('below threshold message','does not currently hold' in page.locator('#gda-status').inner_text());page.close()
      # Signature rejection.
      page=browser.new_page();page.add_init_script(f"({MOCK})({{reject:true}})");page.goto(base+'/index-en.html',wait_until='domcontentloaded');page.wait_for_selector('#goalos-dual-access-card');page.check('#gda-accept');page.check('#gda-boundary');page.click('#gda-verify');page.wait_for_timeout(700);check('signature rejection denied',page.evaluate("sessionStorage.getItem('goalos.sn5.bi2.agialpha.access.receipt')===null"));page.close()
      # Revalidation balance loss, without page reload by direct test API.
      page=browser.new_page();page.add_init_script(f"({MOCK})()");page.goto(base+'/index-en.html',wait_until='domcontentloaded');page.wait_for_selector('#goalos-dual-access-card');page.check('#gda-accept');page.check('#gda-boundary');page.click('#gda-verify');page.wait_for_selector('#goalos-dual-access-status');page.evaluate("window.ethereum.request=async({method,params})=>{if(method==='eth_chainId')return '0x1';if(method==='eth_accounts')return ['0x1111111111111111111111111111111111111111'];if(method==='eth_getCode')return '0x6001';if(method==='eth_call')return '0x0';if(method==='eth_blockNumber')return '0x123457';throw new Error(method)}")
      ok=page.evaluate('GoalOSDualAccess.revalidate()');check('balance loss revalidation fails',ok is False);check('balance loss clears receipt',page.evaluate("sessionStorage.getItem('goalos.sn5.bi2.agialpha.access.receipt')===null"));page.close()
      browser.close()
    check('no page errors',len(report['pageErrors'])==0,str(report['pageErrors']))
    # Existing app can emit harmless blocked-resource console messages; only JS exceptions are fatal above.
    report['passed']=sum(1 for t in report['tests'] if t['pass']);report['total']=len(report['tests']);return report
if __name__=='__main__':
    r=run();out=ROOT/'documentation'/'DUAL_ACCESS_BROWSER_QA.json';out.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2))
