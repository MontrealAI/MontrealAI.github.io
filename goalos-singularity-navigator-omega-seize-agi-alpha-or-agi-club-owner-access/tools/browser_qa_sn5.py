import asyncio, json
from pathlib import Path
from playwright.async_api import async_playwright

SITE=Path('/mnt/data/GoalOS_Singularity_Navigator_Omega_AGI_CLUB_GITHUB_PAGES_FINAL_v3_0_0_SN5_2026-07-26/goalos-singularity-navigator-omega-agi-club-owner-access')
HTML_PATH=SITE/'index.html'; HTML=HTML_PATH.read_text(encoding='utf-8'); OUT=SITE/'preview'; OUT.mkdir(exist_ok=True)
WALLET='0x1234567890abcdef1234567890abcdef12345678'; OTHER='0x9999999999999999999999999999999999999999'; WRAPPER='0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401'; SEPOLIA_WRAPPER='0x0635513f179D50A207757E05759CbD106d7dFcE8'
POLY="""(()=>{const s=new Map();const ls={getItem:k=>s.has(String(k))?s.get(String(k)):null,setItem:(k,v)=>s.set(String(k),String(v)),removeItem:k=>s.delete(String(k)),clear:()=>s.clear(),key:i=>Array.from(s.keys())[i]??null,get length(){return s.size}};Object.defineProperty(window,'localStorage',{value:ls,configurable:true});})();"""
def word_addr(addr): return '0x'+'0'*24+addr[2:].lower()
def word_uint(n): return hex(n)[2:].rjust(64,'0')
def mock_script(mode='direct',expiry=None):
    if expiry is None: expiry=4102444800
    direct_owner=WALLET if mode=='direct' else (WRAPPER if mode in ('wrapped','expired') else (SEPOLIA_WRAPPER if mode=='sepolia' else OTHER))
    wrapped_owner=WALLET if mode in ('wrapped','expired','sepolia') else OTHER
    exp=1 if mode=='expired' else expiry
    return f"""(()=>{{const listeners={{}};const wallet='{WALLET}';const ownerWord='{word_addr(direct_owner)}';const wrappedWord='{word_addr(wrapped_owner)}';const getData='0x'+wrappedWord.slice(2)+'{word_uint(0)}'+'{word_uint(exp)}';window.__walletListeners=listeners;window.ethereum={{request:async({{method,params}})=>{{if(method==='eth_requestAccounts'||method==='eth_accounts')return[wallet];if(method==='eth_chainId')return'0x1';if(method==='wallet_switchEthereumChain')return null;if(method==='eth_blockNumber')return'0x1312d00';if(method==='eth_signTypedData_v4')return'0x'+'cd'.repeat(65);if(method==='personal_sign')return'0x'+'ab'.repeat(65);if(method==='eth_call'){{const data=(params?.[0]?.data||'').toLowerCase();if(data.startsWith('0x02571be3'))return ownerWord;if(data.startsWith('0x6352211e'))return wrappedWord;if(data.startsWith('0x0178fe3f'))return getData;return'0x'+'0'.repeat(64)}}throw new Error('Unsupported mock method '+method)}},on:(event,cb)=>{{(listeners[event]||(listeners[event]=[])).push(cb)}},emit:(event,...args)=>{{(listeners[event]||[]).forEach(cb=>cb(...args))}}}}}})();"""
async def context_with(browser,mode='direct',viewport=None):
    c=await browser.new_context(viewport=viewport or {'width':1600,'height':1000},device_scale_factor=1,accept_downloads=True)
    await c.add_init_script(POLY); await c.add_init_script(mock_script(mode)); return c
async def load(page): await page.set_content(HTML,wait_until='load')
async def fill_gate(page,label):
    await page.fill('#agiClubLabelInput',label)
    for sel in ['#agiClubOwnerConfirm','#agiClubLicenseConfirm','#agiClubBoundaryConfirm','#agiClubLanguageConfirm','#agiClubNoSecretsConfirm']:
        await page.check(sel)
async def unlock(page,label):
    await fill_gate(page,label); await page.click('#agiClubVerifyBtn'); await page.wait_for_function("!document.body.classList.contains('agc-locked')")
def add(results,name,passed,detail=None): results.append({'name':name,'passed':bool(passed),'detail':detail})
async def run():
  results=[]
  async with async_playwright() as p:
    browser=await p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage'])
    # Locked public preview
    c=await browser.new_context(viewport={'width':1600,'height':1000}); await c.add_init_script(POLY)
    page=await c.new_page(); errors=[]; external=[]
    page.on('console',lambda m: errors.append('console:'+m.text) if m.type=='error' else None); page.on('pageerror',lambda e: errors.append('page:'+str(e))); page.on('request',lambda r: external.append(r.url) if r.url.startswith(('http://','https://')) else None)
    await load(page)
    add(results,'locked body state',await page.locator('body').evaluate("e=>e.classList.contains('agc-locked')")); add(results,'locked gate visible',await page.locator('#agiClubAccessGate').is_visible()); add(results,'verify initially disabled',await page.locator('#agiClubVerifyBtn').is_disabled()); add(results,'bilingual confirmations present',await page.locator('#agiClubLanguageConfirm').count()==1 and await page.locator('#agiClubNoSecretsConfirm').count()==1); add(results,'locked page errors zero',not errors,errors); add(results,'locked external runtime requests zero',not external,external)
    await page.screenshot(path=str(OUT/'01_sn5_locked_preview.png'),full_page=False); await c.close()

    # Direct owner full workflow
    c=await context_with(browser,'direct'); page=await c.new_page(); errors=[]; external=[]
    page.on('console',lambda m: errors.append('console:'+m.text) if m.type=='error' else None); page.on('pageerror',lambda e: errors.append('page:'+str(e))); page.on('request',lambda r: external.append(r.url) if r.url.startswith(('http://','https://')) else None)
    await load(page); await fill_gate(page,'founder'); add(results,'all confirmations enable verification',not await page.locator('#agiClubVerifyBtn').is_disabled()); await page.click('#agiClubVerifyBtn'); await page.wait_for_function("!document.body.classList.contains('agc-locked')")
    add(results,'direct Registry owner unlocks',True); add(results,'gate hidden after unlock',await page.locator('#agiClubAccessGate').is_hidden()); add(results,'version SN5 visible','v3.0.0-SN5' in await page.locator('#releaseStatus').inner_text()); add(results,'exact club name visible','founder.club.agi.eth' in await page.locator('#releaseStatus').inner_text())
    add(results,'EIP-712 mode recorded','eip712' in (await page.locator('#ownerVaultSignatureMode').inner_text()).lower()); await page.evaluate("document.querySelector('#ownerAccessDialog')?.close()"); await page.wait_for_timeout(50); add(results,'owner access dialog closes',not await page.locator('#ownerAccessDialog').evaluate('e=>e.open'))
    await page.screenshot(path=str(OUT/'02_sn5_direct_owner_unlocked.png'),full_page=False)
    await page.evaluate("document.querySelector('#demoBtn').click();document.querySelector('#runMission').click()"); await page.wait_for_timeout(5200)
    add(results,'twelve candidate architectures generated',await page.locator('#heroScenarios').inner_text()=='12',await page.locator('#heroScenarios').inner_text()); proof=await page.locator('#heroProof').inner_text(); add(results,'proof posture computed',proof not in ('0%','—',''),proof); rec=await page.locator('#recommendation').inner_text(); add(results,'recommendation generated',len(rec)>100,rec[:180]); jobs=await page.locator('#jobs').inner_text(); add(results,'twenty-one bounded jobs generated',jobs.count('Budget index')==21,jobs.count('Budget index'))
    await page.evaluate("document.querySelector('#ownerAccessDialog')?.open&&document.querySelector('#ownerAccessDialog').close();document.querySelector('button[data-route=opportunity]').click()"); await page.wait_for_timeout(150); add(results,'Opportunity Engine renders champion',await page.locator('#sn5ProposalPortfolio article').count()==3,await page.locator('#sn5ProposalPortfolio article').count()); add(results,'Value-of-Proof queue renders',await page.locator('#sn5Vop .sn5-vop-row').count()>0); await page.screenshot(path=str(OUT/'03_sn5_opportunity_engine.png'),full_page=False)
    await page.evaluate("document.querySelector('#ownerAccessDialog')?.open&&document.querySelector('#ownerAccessDialog').close();document.querySelector('button[data-route=lab]').click()"); await page.evaluate("document.querySelector('#runCounterfactual').click()"); add(results,'Scenario Lab produces rankings',await page.locator('#sn5CounterRanking .proof-priority').count()>=8); add(results,'Scenario Lab stability generated',await page.locator('#sn5StabilityLabel').inner_text() not in ('—','0%')); await page.screenshot(path=str(OUT/'04_sn5_scenario_lab.png'),full_page=False)
    await page.evaluate("document.querySelector('#ownerAccessDialog')?.open&&document.querySelector('#ownerAccessDialog').close();document.querySelector('button[data-route=vault]').click()"); await page.evaluate("document.querySelector('#generateProofBundle').click()"); await page.wait_for_timeout(150); digest=await page.locator('#sn5BundleDigest').inner_text(); add(results,'Evidence Vault computes SHA-256','SHA-256' in digest and len(digest)>90,digest[:150]); await page.screenshot(path=str(OUT/'05_sn5_evidence_vault.png'),full_page=False)
    await page.evaluate("document.querySelector('#ownerAccessDialog')?.open&&document.querySelector('#ownerAccessDialog').close();document.querySelector('button[data-route=legal]').click()"); await page.wait_for_timeout(80); add(results,'French legal text first','Cadre opérationnel' in await page.locator('#view-legal').inner_text()); await page.evaluate("document.querySelector('[data-legal-lang=en]').click()"); add(results,'English legal choice works','Operating boundary' in await page.locator('#view-legal').inner_text()); await page.screenshot(path=str(OUT/'06_sn5_legal_center.png'),full_page=False)
    routes=[]
    for btn in await page.locator('nav button[data-route]').all():
        route=await btn.get_attribute('data-route')
        if route in routes: continue
        routes.append(route); await page.evaluate('(r)=>document.querySelector(`nav button[data-route=${r}]`).click()',route); await page.wait_for_timeout(35); add(results,f'route {route} activates',await page.locator(f'#view-{route}').evaluate("e=>e.classList.contains('active')"))
    add(results,'all canonical views exercised',len(routes)==16,routes)
    await page.evaluate("document.querySelector('#clubAccessStatusBtn').click()"); add(results,'owner access center opens',await page.locator('#ownerAccessDialog').evaluate('e=>e.open')); add(results,'owner center shows exact name','founder.club.agi.eth' in await page.locator('#ownerVaultName').inner_text()); await page.evaluate("document.querySelector('#ownerVaultLock').click()"); await page.wait_for_function("document.body.classList.contains('agc-locked')"); add(results,'manual lock relocks',await page.locator('#agiClubAccessGate').is_visible()); add(results,'direct workflow errors zero',not errors,errors); add(results,'direct workflow external runtime requests zero',not external,external); await c.close()

    # Event relock tests
    for event,args,name in [('accountsChanged',[OTHER],'account change'),('chainChanged',['0xaa36a7'],'network change'),('disconnect',[],'wallet disconnect')]:
        c=await context_with(browser,'direct',{'width':1280,'height':820}); page=await c.new_page(); await load(page); await unlock(page,'eventtest'); await page.evaluate("([e,a])=>window.ethereum.emit(e,...a)",[event,args]); await page.wait_for_timeout(150); add(results,f'{name} relocks',await page.locator('body').evaluate("e=>e.classList.contains('agc-locked')")); await c.close()
    # Wrapped owner — current official and authorized legacy deployments
    c=await context_with(browser,'wrapped',{'width':1365,'height':900}); page=await c.new_page(); await load(page); await unlock(page,'wrapped'); add(results,'current official wrapped direct owner unlocks',True); await page.evaluate("document.querySelector('#clubAccessStatusBtn').click()"); add(results,'current wrapper mode recorded','Name Wrapper' in await page.locator('#ownerVaultMode').inner_text()); await c.close()
    c=await context_with(browser,'legacy',{'width':1365,'height':900}); page=await c.new_page(); await load(page); await unlock(page,'legacy'); add(results,'legacy official wrapped direct owner unlocks',True); await page.evaluate("document.querySelector('#clubAccessStatusBtn').click()"); add(results,'legacy wrapper mode recorded','Name Wrapper' in await page.locator('#ownerVaultMode').inner_text()); await c.close()
    # Rejections
    for mode,label,needle,name in [('wrong','wrong','not the connected wallet','wrong owner'),('expired','expired','expired','expired wrapped name')]:
        c=await context_with(browser,mode,{'width':1280,'height':850}); page=await c.new_page(); await load(page); await fill_gate(page,label); await page.click('#agiClubVerifyBtn'); await page.wait_for_timeout(600); add(results,f'{name} denied',await page.locator('body').evaluate("e=>e.classList.contains('agc-locked')")); status=(await page.locator('#agiClubGateStatus').inner_text()).lower(); add(results,f'{name} reason shown',needle in status,status); await c.close()
    c=await context_with(browser,'direct',{'width':1280,'height':850}); page=await c.new_page(); await load(page); await fill_gate(page,'nested.alpha'); add(results,'nested label rejected before RPC',await page.locator('#agiClubVerifyBtn').is_disabled()); await c.close()
    # Mobile
    c=await context_with(browser,'direct',{'width':390,'height':844}); page=await c.new_page(); errors=[]
    page.on('pageerror',lambda e: errors.append('page:'+str(e))); await load(page); ov=await page.evaluate('document.documentElement.scrollWidth-document.documentElement.clientWidth'); add(results,'mobile locked overflow zero',ov<=1,ov); await page.screenshot(path=str(OUT/'07_sn5_mobile_locked.png'),full_page=False); await unlock(page,'mobile'); await page.evaluate("document.querySelector('#ownerAccessDialog')?.close();document.querySelector('#demoBtn').click();document.querySelector('#runMission').click()"); await page.wait_for_timeout(5200); await page.evaluate("document.querySelector('button[data-route=opportunity]').click()"); ov=await page.evaluate('document.documentElement.scrollWidth-document.documentElement.clientWidth'); add(results,'mobile unlocked overflow zero',ov<=1,ov); await page.screenshot(path=str(OUT/'08_sn5_mobile_opportunity.png'),full_page=False); add(results,'mobile errors zero',not errors,errors); await c.close()
    await browser.close()
  out={'passed':all(x['passed'] for x in results),'assertions':len(results),'results':results}
  (SITE/'documentation').mkdir(exist_ok=True); (SITE/'documentation'/'QA_REPORT.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
  if not out['passed']: raise SystemExit(1)
asyncio.run(run())
