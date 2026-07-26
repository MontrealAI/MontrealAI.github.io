import asyncio, json
from pathlib import Path
from playwright.async_api import async_playwright

HTML_PATH=Path('/mnt/data/GoalOS_Singularity_Navigator_Omega_AGI_CLUB_GITHUB_PAGES_FINAL_v2_0_0_SN3_2026-07-26/goalos-singularity-navigator-omega-agi-club-owner-access/index.html')
HTML=HTML_PATH.read_text(encoding='utf-8')
OUT=HTML_PATH.parent/'preview'; OUT.mkdir(exist_ok=True)
WALLET='0x1234567890abcdef1234567890abcdef12345678'; OTHER='0x9999999999999999999999999999999999999999'; WRAPPER='0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401'
POLY="""(()=>{const s=new Map();const ls={getItem:k=>s.has(String(k))?s.get(String(k)):null,setItem:(k,v)=>s.set(String(k),String(v)),removeItem:k=>s.delete(String(k)),clear:()=>s.clear(),key:i=>Array.from(s.keys())[i]??null,get length(){return s.size}};Object.defineProperty(window,'localStorage',{value:ls,configurable:true});})();"""

def word_addr(addr): return '0x'+'0'*24+addr[2:].lower()
def word_uint(n): return hex(n)[2:].rjust(64,'0')
def mock_script(mode='direct',expiry=None):
    if expiry is None: expiry=4102444800
    direct_owner=WALLET if mode=='direct' else (WRAPPER if mode in ('wrapped','expired') else OTHER)
    wrapped_owner=WALLET if mode in ('wrapped','expired') else OTHER
    exp=1 if mode=='expired' else expiry
    return f"""(()=>{{const listeners={{}};const wallet='{WALLET}';const ownerWord='{word_addr(direct_owner)}';const wrappedWord='{word_addr(wrapped_owner)}';const getData='0x'+wrappedWord.slice(2)+'{word_uint(0)}'+'{word_uint(exp)}';window.__walletListeners=listeners;window.ethereum={{request:async({{method,params}})=>{{if(method==='eth_requestAccounts'||method==='eth_accounts')return[wallet];if(method==='eth_chainId')return'0x1';if(method==='wallet_switchEthereumChain')return null;if(method==='eth_blockNumber')return'0x1312d00';if(method==='personal_sign')return'0x'+'ab'.repeat(65);if(method==='eth_call'){{const data=(params?.[0]?.data||'').toLowerCase();if(data.startsWith('0x02571be3'))return ownerWord;if(data.startsWith('0x6352211e'))return wrappedWord;if(data.startsWith('0x0178fe3f'))return getData;return'0x'+'0'.repeat(64)}}throw new Error('Unsupported mock method '+method)}},on:(event,cb)=>{{(listeners[event]||(listeners[event]=[])).push(cb)}},emit:(event,...args)=>{{(listeners[event]||[]).forEach(cb=>cb(...args))}}}}}})();"""

async def context_with(browser,mode='direct',viewport=None):
    c=await browser.new_context(viewport=viewport or {'width':1600,'height':1000},device_scale_factor=1,accept_downloads=True)
    await c.add_init_script(POLY); await c.add_init_script(mock_script(mode)); return c
async def load(page): await page.set_content(HTML,wait_until='load')
async def fill_gate(page,label):
    await page.fill('#agiClubLabelInput',label)
    for sel in ['#agiClubOwnerConfirm','#agiClubLicenseConfirm','#agiClubBoundaryConfirm']: await page.check(sel)
async def unlock(page,label):
    await fill_gate(page,label); await page.click('#agiClubVerifyBtn'); await page.wait_for_function("!document.body.classList.contains('agc-locked')")

def add(results,name,passed,detail=None): results.append({'name':name,'passed':bool(passed),'detail':detail})

async def run():
  results=[]
  async with async_playwright() as p:
    browser=await p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage'])
    # locked desktop
    c=await browser.new_context(viewport={'width':1600,'height':1000}); await c.add_init_script(POLY)
    page=await c.new_page(); errors=[]; requests=[]
    page.on('console',lambda m: errors.append('console:'+m.text) if m.type=='error' else None); page.on('pageerror',lambda e: errors.append('page:'+str(e))); page.on('request',lambda r: requests.append(r.url))
    await load(page)
    add(results,'locked body state',await page.locator('body').evaluate("e=>e.classList.contains('agc-locked')")); add(results,'locked gate visible',await page.locator('#agiClubAccessGate').is_visible()); add(results,'verify initially disabled',await page.locator('#agiClubVerifyBtn').is_disabled()); add(results,'locked page errors zero',not errors,errors); add(results,'locked external network requests zero',not requests,requests)
    await page.screenshot(path=str(OUT/'01_locked_preview.png'),full_page=True); await c.close()

    # direct owner full workflow
    c=await context_with(browser,'direct'); page=await c.new_page(); errors=[]; requests=[]
    page.on('console',lambda m: errors.append('console:'+m.text) if m.type=='error' else None); page.on('pageerror',lambda e: errors.append('page:'+str(e))); page.on('request',lambda r: requests.append(r.url))
    await load(page); await fill_gate(page,'founder'); add(results,'valid gate enables verification',not await page.locator('#agiClubVerifyBtn').is_disabled()); await page.click('#agiClubVerifyBtn'); await page.wait_for_function("!document.body.classList.contains('agc-locked')")
    add(results,'direct Registry owner unlocks',True); add(results,'gate hidden after unlock',await page.locator('#agiClubAccessGate').is_hidden()); add(results,'verified status names exact club', 'founder.club.agi.eth' in await page.locator('#releaseStatus').inner_text())
    await page.screenshot(path=str(OUT/'02_direct_owner_unlocked.png'),full_page=False)
    await page.click('#demoBtn'); await page.click('#runMission'); await page.wait_for_timeout(3900)
    add(results,'eight candidate architectures generated',await page.locator('#heroScenarios').inner_text()=='8',await page.locator('#heroScenarios').inner_text()); proof=await page.locator('#heroProof').inner_text(); add(results,'proof posture computed',proof not in ('0%','—',''),proof); rec=await page.locator('#recommendation').inner_text(); add(results,'recommendation generated','Verified Balanced Acceleration' in rec,rec[:180]); await page.click('button[data-route=brief]'); await page.wait_for_timeout(100); add(results,'Apex Brief decision state generated',await page.locator('#briefDecisionState').inner_text()!='CONSTITUTE',await page.locator('#briefDecisionState').inner_text()); add(results,'Apex Brief SER generated',await page.locator('#briefSER').inner_text() not in ('—',''),await page.locator('#briefSER').inner_text()); add(results,'Value of Proof queue generated',await page.locator('#briefProofQueue').inner_text()!=''); await page.click('button[data-route=command]'); jobs=await page.locator('#jobs').inner_text(); add(results,'fifteen bounded jobs generated',jobs.count('Budget index')==15,jobs.count('Budget index'))
    await page.screenshot(path=str(OUT/'03_navigation_command.png'),full_page=True)
    routes=[]
    for btn in await page.locator('button[data-route]').all():
        route=await btn.get_attribute('data-route')
        if route in routes: continue
        routes.append(route); await btn.click(); await page.wait_for_timeout(25); add(results,f'route {route} activates',await page.locator(f'#view-{route}').evaluate("e=>e.classList.contains('active')"))
    add(results,'all canonical views exercised',len(routes)==12,routes)
    await page.click('#clubAccessStatusBtn'); add(results,'owner access center opens',await page.locator('#ownerAccessDialog').evaluate('e=>e.open')); add(results,'owner center shows exact name','founder.club.agi.eth' in await page.locator('#ownerVaultName').inner_text()); add(results,'owner center reports direct mode','Registry direct owner' in await page.locator('#ownerVaultMode').inner_text()); await page.screenshot(path=str(OUT/'04_owner_access_center.png'),full_page=False)
    await page.click('#ownerVaultLock'); await page.wait_for_function("document.body.classList.contains('agc-locked')"); add(results,'manual lock relocks',await page.locator('#agiClubAccessGate').is_visible()); add(results,'direct workflow errors zero',not errors,errors); add(results,'direct workflow external network requests zero',not requests,requests); await c.close()

    # event relock tests
    for event,args,name in [('accountsChanged',[OTHER],'account change'),('chainChanged',['0xaa36a7'],'network change'),('disconnect',[],'wallet disconnect')]:
        c=await context_with(browser,'direct',{'width':1280,'height':820}); page=await c.new_page(); await load(page); await unlock(page,'eventtest'); await page.evaluate("([e,a])=>window.ethereum.emit(e,...a)",[event,args]); await page.wait_for_timeout(120); add(results,f'{name} relocks',await page.locator('body').evaluate("e=>e.classList.contains('agc-locked')")); await c.close()

    # wrapped owner
    c=await context_with(browser,'wrapped',{'width':1365,'height':900}); page=await c.new_page(); await load(page); await unlock(page,'wrapped'); add(results,'official wrapped direct owner unlocks',True); await page.click('#clubAccessStatusBtn'); add(results,'wrapped ownership mode recorded','Name Wrapper' in await page.locator('#ownerVaultMode').inner_text()); await c.close()

    # rejection cases
    for mode,label,needle,name in [('wrong','wrong','not the connected wallet','wrong owner'),('expired','expired','expired','expired wrapped name')]:
        c=await context_with(browser,mode,{'width':1280,'height':850}); page=await c.new_page(); await load(page); await fill_gate(page,label); await page.click('#agiClubVerifyBtn'); await page.wait_for_timeout(500); add(results,f'{name} denied',await page.locator('body').evaluate("e=>e.classList.contains('agc-locked')")); status=(await page.locator('#agiClubGateStatus').inner_text()).lower(); add(results,f'{name} reason shown',needle in status,status); await c.close()
    c=await context_with(browser,'direct',{'width':1280,'height':850}); page=await c.new_page(); await load(page); await fill_gate(page,'nested.alpha'); add(results,'nested label rejected before RPC',await page.locator('#agiClubVerifyBtn').is_disabled()); await c.close()

    # mobile
    c=await context_with(browser,'direct',{'width':390,'height':844}); page=await c.new_page(); errors=[]
    page.on('pageerror',lambda e: errors.append('page:'+str(e))); await load(page); ov=await page.evaluate('document.documentElement.scrollWidth-document.documentElement.clientWidth'); add(results,'mobile locked overflow zero',ov<=1,ov); await page.screenshot(path=str(OUT/'05_mobile_locked.png'),full_page=True); await unlock(page,'mobile'); ov=await page.evaluate('document.documentElement.scrollWidth-document.documentElement.clientWidth'); add(results,'mobile unlocked overflow zero',ov<=1,ov); await page.screenshot(path=str(OUT/'06_mobile_unlocked.png'),full_page=True); add(results,'mobile errors zero',not errors,errors); await c.close()
    await browser.close()
  out={'passed':all(x['passed'] for x in results),'assertions':len(results),'results':results}
  Path('/tmp/sn_browser_qa.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
  if not out['passed']: raise SystemExit(1)
asyncio.run(run())
