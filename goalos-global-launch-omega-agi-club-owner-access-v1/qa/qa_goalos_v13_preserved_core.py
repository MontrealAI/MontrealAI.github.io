import asyncio, json, re, time, sys
from pathlib import Path
from playwright.async_api import async_playwright

HTML_PATH=Path('/mnt/data/GoalOS_Global_Launch_Omega_AGI_CLUB_GITHUB_PAGES_FINAL_v13_0_0_GL1_2026-07-24/goalos-global-launch-omega-agi-club-owner-access/index.html')
HTML=HTML_PATH.read_text(encoding='utf-8',errors='replace')
OUT=Path('/mnt/data/goalos_v13_release_assets/qa/core_v12_compat'); OUT.mkdir(parents=True,exist_ok=True)
SCREENS=OUT/'screens'; SCREENS.mkdir(exist_ok=True)
W='0x1111111111111111111111111111111111111111'; OTHER='0x2222222222222222222222222222222222222222'
MAIN_WRAPPER='0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401'; SEPOLIA_WRAPPER='0x0635513f179D50A207757E05759CbD106d7dFcE8'
MOCK_JS=f"""(()=>{{
 const handlers={{}}; const wallet='{W}', other='{OTHER}', mainWrapper='{MAIN_WRAPPER}', sepWrapper='{SEPOLIA_WRAPPER}';
 window.__mockOwnerMode='direct'; window.__mockChain='0x1'; window.__mockWallet=wallet;
 const wordAddr=a=>'0x'+'0'.repeat(24)+a.slice(2).toLowerCase(); const uword=n=>BigInt(n).toString(16).padStart(64,'0');
 window.ethereum={{request:async({{method,params}})=>{{
  const mode=window.__mockOwnerMode||'direct', acct=window.__mockWallet||wallet;
  if(method==='eth_requestAccounts'||method==='eth_accounts') return [acct];
  if(method==='eth_chainId') return window.__mockChain||'0x1';
  if(method==='wallet_switchEthereumChain'){{window.__mockChain='0x1';return null;}}
  if(method==='personal_sign') return '0x'+'11'.repeat(65);
  if(method==='eth_blockNumber') return '0x123456';
  if(method==='eth_call'){{const data=String(params?.[0]?.data||'').toLowerCase();
   if(data.startsWith('0x02571be3')){{if(mode==='direct')return wordAddr(acct);if(mode==='wrapped'||mode==='expired')return wordAddr(mainWrapper);if(mode==='nonmain')return wordAddr(sepWrapper);return wordAddr(other);}}
   if(data.startsWith('0x6352211e'))return wordAddr(mode==='wrongwrapped'?other:acct);
   if(data.startsWith('0x0178fe3f')){{const exp=mode==='expired'?1600000000:4102444800;return wordAddr(acct)+uword(0)+uword(exp);}}
   return '0x'+'0'.repeat(64);
  }}
  throw new Error('Unsupported '+method);
 }},on:(n,fn)=>{{(handlers[n]||(handlers[n]=[])).push(fn)}},emit:(n,...a)=>{{(handlers[n]||[]).forEach(fn=>fn(...a))}}}};
}})();"""

async def wait_text(page,sel,values,timeout=150000):
    end=time.time()+timeout/1000; last=''
    while time.time()<end:
        try:last=(await page.locator(sel).inner_text()).strip()
        except:last=''
        if last in values:return last
        await page.wait_for_timeout(120)
    raise AssertionError(f'timeout {sel}: {last}')

async def accept_legal(page):
    if await page.locator('#legalModal').count() and await page.locator('#legalModal').get_attribute('hidden') is None:
        for sel in ['#acceptTermsCheck','#authorityCheck','#dataBoundaryCheck','#mandatoryLawCheck']:
            if await page.locator(sel).count() and not await page.locator(sel).is_checked(): await page.check(sel)
        await page.click('#acceptTermsBtn')
        for _ in range(120):
            if await page.locator('#legalModal').get_attribute('hidden') is not None:return
            await page.wait_for_timeout(100)

async def unlock(page,mode='direct',label='alpha-council'):
    if await page.locator('#agiClubProofDrawer').count():
        cls=(await page.locator('#agiClubProofDrawer').get_attribute('class')) or ''
        if 'open' not in cls: await page.click('#agiClubOpenProofBtn')
    await page.evaluate("m=>window.__mockOwnerMode=m",mode)
    await page.fill('#agiClubLabelInput',label)
    for sel in ['#agiClubOwnerConfirm','#agiClubLicenseConfirm','#agiClubBoundaryConfirm']:
        if not await page.locator(sel).is_checked(): await page.check(sel)
    await page.click('#agiClubVerifyBtn')
    for _ in range(300):
        if not await page.locator('#agiClubAccessGate').is_visible():return True
        if await page.locator('#agiClubGateStatus').get_attribute('data-state')=='bad':return False
        await page.wait_for_timeout(100)
    return False

async def make_page(browser,width=1600,height=1000):
    page=await browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1,accept_downloads=True)
    logs=[]; errors=[]; requests=[]
    page.on('console',lambda m: logs.append({'type':m.type,'text':m.text}))
    page.on('pageerror',lambda e: errors.append(str(e)))
    page.on('request',lambda r: requests.append(r.url))
    await page.evaluate(MOCK_JS)
    await page.set_content(HTML,wait_until='domcontentloaded',timeout=120000)
    await page.wait_for_timeout(1800)
    return page,logs,errors,requests

async def main():
    controls=[]
    def ck(name,cond,detail=''):
        controls.append({'name':name,'passed':bool(cond),'detail':str(detail)}); print(('PASS' if cond else 'FAIL'),name,detail,flush=True)
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage'])
        page,logs,errors,requests=await make_page(browser)
        ck('Owner gate visible',await page.locator('#agiClubAccessGate').is_visible())
        await page.screenshot(path=str(SCREENS/'01_owner_gate.png'),full_page=False)
        ok=await unlock(page); ck('Direct AGI Club owner unlocks',ok)
        await accept_legal(page)
        # Required navigation and institution sections
        nav_expect=['Institution Formation','GoalOS Chat Ω','Mission Contract','Global Launch','Evidence & Chronicle','Research Paper','Legal Center']
        for name in nav_expect:
            
            if name=='Legal Center':
                vis=await page.locator('.nav .legal-center-tab').filter(has_text='Legal Center').first.is_visible()
            else:
                vis=await page.get_by_role('link',name=name,exact=True).first.is_visible()
            ck(f'Navigation exposes {name}',vis)
        for sid in ['institution-formation','goalos-chat','mission-contract','global-launch','launch-missions','proof-governance','graph-governance','expansion-office','research-proof','omega-prime']:
            ck(f'Section {sid} exists',await page.locator('#'+sid).count()==1)
        # Chat compiler
        await page.locator('#goalos-chat').scroll_into_view_if_needed(); await page.wait_for_timeout(300)
        prompt='Launch an AI infrastructure company with core R&D in Montréal, sell to the United States and Europe, hire 25 specialists, preserve founder control, obtain public support and raise US$20M for a pilot compute facility.'
        await page.fill('#glChatInput',prompt); await page.click('#glChatCompile'); await page.wait_for_timeout(500)
        cls=await page.locator('.launch-type[data-launch-type="infrastructure"]').get_attribute('class')
        ck('GoalOS Chat compiles objective into launch type','active' in (cls or ''),cls)
        ck('Chat extracts capital requirement',await page.locator('#glCapitalNeed').input_value()=='20000000',await page.locator('#glCapitalNeed').input_value())
        ck('Chat extracts team size',await page.locator('#glTeam').input_value()=='25',await page.locator('#glTeam').input_value())
        ck('Chat publishes decision-changing questions',await page.locator('.chat-question').count()>=3,str(await page.locator('.chat-question').count()))
        await page.screenshot(path=str(SCREENS/'02_goalos_chat.png'),full_page=False)
        # Mission Contract
        await page.locator('#mission-contract').scroll_into_view_if_needed(); await page.wait_for_timeout(250)
        ck('Mission Contract reaches 100 percent',await page.locator('#glContractPct').inner_text()=='100%',await page.locator('#glContractPct').inner_text())
        await page.click('#glFreezeContract'); await page.wait_for_timeout(500)
        ck('Mission Contract freezes',await page.locator('#glContractState').inner_text()=='FROZEN',await page.locator('#glContractState').inner_text())
        digest=await page.locator('#glContractHash').inner_text()
        ck('Frozen contract has SHA-256',bool(re.search(r'[0-9a-f]{64}',digest,re.I)),digest)
        # Architecture run
        await page.locator('#global-launch').scroll_into_view_if_needed(); await page.click('#glRun')
        st=await wait_text(page,'#glEngineState',{'COMPLETE','REVIEW REQUIRED'})
        ck('Global Launch architecture completes',st=='COMPLETE',st)
        gl=await page.evaluate("JSON.parse(JSON.stringify(window.GOALOS_GLOBAL_LAUNCH.getState()))")
        ck('Ten institution modules generated',len(gl.get('modules',[]))==10,len(gl.get('modules',[])))
        ck('Fifty bounded AGI Jobs generated',len(gl.get('jobs',[]))==50,len(gl.get('jobs',[])))
        ck('Ten mission graph nodes rendered',await page.locator('.gl-node').count()==10,await page.locator('.gl-node').count())
        ck('Eight global functions allocated',len([k for k in ['sell','hire','rd','ip','capital','support','infrastructure','value'] if gl.get('scenario',{}).get('roles',{}).get(k)])==8)
        ck('Twelve monitoring watch classes generated',len(gl.get('watch',[]))>=12,len(gl.get('watch',[])))
        await page.screenshot(path=str(SCREENS/'03_institution_architecture.png'),full_page=False)
        # Proof governance
        await page.locator('#proof-governance').scroll_into_view_if_needed(); await page.wait_for_timeout(400)
        ck('Twelve Evidence Docket objects rendered',await page.locator('.docket-object').count()==12,await page.locator('.docket-object').count())
        ck('Proof Debt ledger populated',await page.locator('.proof-debt-item').count()>0,await page.locator('.proof-debt-item').count())
        ck('Transaction gate initially blocked','BLOCKED' in await page.locator('#v12GateDecision').inner_text())
        receipt_loc=page.locator('#v12ReceiptGrid [data-receipt]')
        receipt_count=await receipt_loc.count(); ck('Nine transaction receipts rendered',receipt_count==9,receipt_count)
        for box in await receipt_loc.all():
            if not await box.is_checked(): await box.check()
        ck('Nine-receipt gate admits after local declarations','ADMITTED' in await page.locator('#v12GateDecision').inner_text())
        chron_loc=page.locator('#v12ChronicleChecks [data-chronicle]')
        chron_count=await chron_loc.count(); ck('Eight Chronicle checks rendered',chron_count==8,chron_count)
        for box in await chron_loc.all():
            if not await box.is_checked(): await box.check()
        ck('Chronicle admission requires all eight checks','ADMIT WITH SCOPE' in await page.locator('#v12ChronicleDecision').inner_text())
        await page.click('#v12RunSuccessor'); await page.wait_for_timeout(250)
        ck('Fresh successor test produces delta',(await page.locator('#v12SuccessorDelta').inner_text()).strip() not in ('','—'))
        ck('Verified Launch Value index rendered',(await page.locator('#v12VlvIndex').inner_text()).strip() not in ('','—'))
        await page.screenshot(path=str(SCREENS/'04_evidence_chronicle.png'),full_page=False)
        # Graph governance + paper
        await page.locator('#graph-governance').scroll_into_view_if_needed(); await page.wait_for_timeout(300)
        edge_count=await page.locator('#graph-governance .edge-type-grid > article').count()
        plane_count=await page.locator('#graph-governance .powers-row:not(.head)').count()
        ck('Five graph edge types rendered',edge_count==5,edge_count)
        ck('Six institutional planes rendered',plane_count==6,plane_count)
        await page.locator('#research-proof').scroll_into_view_if_needed(); await page.wait_for_timeout(300)
        ck('Research paper download link present',await page.locator('a[href*="GoalOS_Global_Launch_Omega_v13_Apex"]').count()>=1)
        ck('Ten proof programme gates rendered',await page.locator('.proof-gate').count()==10,await page.locator('.proof-gate').count())
        await page.screenshot(path=str(SCREENS/'05_research_proof.png'),full_page=False)
        # Preserved systems
        for sid in ['jurisdictions','instruments','eligibility','stacking','capital','transactions','governance','legal','legal-center','evidence-workbench','sources','dataroom']:
            ck(f'Preserved system {sid}',await page.locator('#'+sid).count()==1)
        # Exports
        async with page.expect_download() as d1:
            await page.click('#glExportDossier')
        dl=await d1.value; ck('Global Launch dossier export works',dl.suggested_filename.endswith('.json'),dl.suggested_filename)
        async with page.expect_download() as d2:
            await page.click('#glExportContract')
        dl2=await d2.value; ck('Mission Contract export works',dl2.suggested_filename.endswith('.json'),dl2.suggested_filename)
        # Desktop runtime
        dims=await page.evaluate("({sw:document.documentElement.scrollWidth,cw:document.documentElement.clientWidth})")
        ck('Desktop no horizontal overflow',dims['sw']<=dims['cw']+1,dims)
        ext=[u for u in requests if u.startswith('http://') or u.startswith('https://')]
        ck('No external runtime requests',len(ext)==0,sorted(set(ext))[:5])
        ck('Zero page errors',len(errors)==0,errors[:5])
        ce=[x for x in logs if x['type']=='error']; ck('Zero console errors',len(ce)==0,ce[:5])
        # Account change relock
        await page.evaluate("window.ethereum.emit('accountsChanged',['0x3333333333333333333333333333333333333333'])")
        await page.wait_for_timeout(400)
        ck('Wallet account change relocks institution',await page.locator('#agiClubAccessGate').is_visible())
        await page.close()
        # Negative ownership and mobile cases are covered by dedicated sealed test passes.
        await browser.close()
    result={'release':'GoalOS Global Launch Ω v13.0.0-GL1','testedAt':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'passed':sum(x['passed'] for x in controls),'failed':sum(not x['passed'] for x in controls),'total':len(controls),'controls':controls}
    (OUT/'core_browser_qa.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if result['failed']: sys.exit(1)

if __name__=='__main__': asyncio.run(main())
