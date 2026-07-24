import asyncio, json, re, sys, time
from pathlib import Path
from playwright.async_api import async_playwright

SITE=Path('/mnt/data/GoalOS_Global_Launch_Omega_AGI_CLUB_GITHUB_PAGES_FINAL_v13_0_0_GL1_2026-07-24/goalos-global-launch-omega-agi-club-owner-access')
HTML=(SITE/'index.html').read_text('utf-8',errors='replace')
OUT=Path('/mnt/data/goalos_v13_release_assets/qa');OUT.mkdir(parents=True,exist_ok=True)
SCREENS=OUT/'screens';SCREENS.mkdir(exist_ok=True)
W='0x1111111111111111111111111111111111111111';OTHER='0x2222222222222222222222222222222222222222';MAIN='0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401';SEP='0x0635513f179D50A207757E05759CbD106d7dFcE8'
MOCK=f"""(()=>{{const handlers={{}},wallet='{W}',other='{OTHER}',main='{MAIN}',sep='{SEP}';window.__mockOwnerMode='direct';window.__mockWallet=wallet;window.__mockChain='0x1';const wa=a=>'0x'+'0'.repeat(24)+a.slice(2).toLowerCase(),uw=n=>BigInt(n).toString(16).padStart(64,'0');window.ethereum={{request:async({{method,params}})=>{{const mode=window.__mockOwnerMode||'direct',acct=window.__mockWallet||wallet;if(method==='eth_requestAccounts'||method==='eth_accounts')return[acct];if(method==='eth_chainId')return window.__mockChain;if(method==='wallet_switchEthereumChain'){{window.__mockChain='0x1';return null}}if(method==='personal_sign')return'0x'+'11'.repeat(65);if(method==='eth_blockNumber')return'0x123456';if(method==='eth_call'){{const d=String(params?.[0]?.data||'').toLowerCase();if(d.startsWith('0x02571be3')){{if(mode==='direct')return wa(acct);if(mode==='wrapped'||mode==='expired')return wa(main);if(mode==='nonmain')return wa(sep);return wa(other)}}if(d.startsWith('0x6352211e'))return wa(mode==='wrongwrapped'?other:acct);if(d.startsWith('0x0178fe3f'))return wa(acct)+uw(0)+uw(mode==='expired'?1600000000:4102444800);return'0x'+'0'.repeat(64)}}throw Error('Unsupported mock method '+method)}},on:(n,f)=>(handlers[n]||(handlers[n]=[])).push(f),emit:(n,...a)=>(handlers[n]||[]).forEach(f=>f(...a))}}}})();"""

async def page_new(browser,w=1600,h=1000):
    ctx=await browser.new_context(viewport={'width':w,'height':h},accept_downloads=True)
    p=await ctx.new_page();p.set_default_timeout(30000)
    logs=[];errs=[];req=[]
    p.on('console',lambda m:logs.append({'type':m.type,'text':m.text}))
    p.on('pageerror',lambda e:errs.append(getattr(e,'stack',str(e))))
    p.on('request',lambda r:req.append(r.url))
    await p.evaluate(MOCK);await p.set_content(HTML,wait_until='domcontentloaded',timeout=120000);await p.wait_for_timeout(1200)
    return ctx,p,logs,errs,req

async def unlock(p,mode='direct',label='alpha-council'):
    if 'open' not in ((await p.get_attribute('#agiClubProofDrawer','class')) or ''):
        await p.click('#agiClubOpenProofBtn')
    await p.evaluate('m=>window.__mockOwnerMode=m',mode)
    await p.fill('#agiClubLabelInput',label)
    for s in ['#agiClubOwnerConfirm','#agiClubLicenseConfirm','#agiClubBoundaryConfirm']:
        if not await p.locator(s).is_checked():await p.check(s)
    await p.click('#agiClubVerifyBtn')
    for _ in range(250):
        if not await p.locator('#agiClubAccessGate').is_visible():return True
        if await p.locator('#agiClubGateStatus').get_attribute('data-state')=='bad':return False
        await p.wait_for_timeout(80)
    return False

async def legal(p):
    if await p.locator('#legalModal').count() and await p.locator('#legalModal').get_attribute('hidden') is None:
        for s in ['#acceptTermsCheck','#authorityCheck','#dataBoundaryCheck','#mandatoryLawCheck']:
            if await p.locator(s).count() and not await p.locator(s).is_checked():await p.check(s)
        await p.click('#acceptTermsBtn');await p.wait_for_timeout(250)

async def wait_state(p,sel,vals,timeout=150000):
    end=time.time()+timeout/1000;t=''
    while time.time()<end:
        try:t=(await p.locator(sel).inner_text()).strip()
        except:t=''
        if t in vals:return t
        await p.wait_for_timeout(100)
    raise AssertionError(f'{sel} timeout: {t}')

async def main():
    tests=[]
    def ck(n,c,d=''):
        tests.append({'name':n,'passed':bool(c),'detail':d});print(('PASS' if c else 'FAIL'),n,d,flush=True)
    async with async_playwright() as pw:
        b=await pw.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage'])
        ctx,p,logs,errs,req=await page_new(b)
        ck('Premium owner preview visible',await p.locator('#agiClubAccessGate').is_visible())
        ck('Premium preview title visible',await p.locator('.v13-preview-copy h1').is_visible())
        ck('Six preview metrics',await p.locator('.v13-preview-stats>div').count()==6,str(await p.locator('.v13-preview-stats>div').count()))
        ck('Four institutional preview cards',await p.locator('.v13-preview-capabilities article').count()==4,str(await p.locator('.v13-preview-capabilities article').count()))
        await p.screenshot(path=str(SCREENS/'01_locked_preview.png'),full_page=False)
        await p.click('#agiClubOpenProofBtn');ck('Owner access drawer opens','open' in ((await p.get_attribute('#agiClubProofDrawer','class')) or ''))
        await p.screenshot(path=str(SCREENS/'02_owner_access_drawer.png'),full_page=False)
        ck('Direct owner unlocks',await unlock(p,'direct'))
        await legal(p)
        ck('Main flagship hero visible',await p.locator('.v13-main-hero').is_visible())
        ck('Main four live panels',await p.locator('.v13-live-card').count()==4,str(await p.locator('.v13-live-card').count()))
        await p.screenshot(path=str(SCREENS/'03_main_dashboard.png'),full_page=False)
        # Existing v12 core workflow.
        prompt='Launch an AI infrastructure company with core R&D in Montréal, customers in the United States and Europe, 25 specialists, protected capability, public support and a US$20M capital plan.'
        await p.locator('#goalos-chat').scroll_into_view_if_needed();await p.fill('#glChatInput',prompt);await p.click('#glChatCompile');await p.wait_for_timeout(250)
        ck('GoalOS Chat compiles capital',await p.input_value('#glCapitalNeed')=='20000000',await p.input_value('#glCapitalNeed'))
        await p.locator('#mission-contract').scroll_into_view_if_needed();ck('Mission Contract complete',await p.inner_text('#glContractPct')=='100%',await p.inner_text('#glContractPct'));await p.click('#glFreezeContract');
        for _ in range(100):
            if await p.inner_text('#glContractState')=='FROZEN':break
            await p.wait_for_timeout(50)
        ck('Mission Contract frozen',await p.inner_text('#glContractState')=='FROZEN')
        await p.locator('#global-launch').scroll_into_view_if_needed();await p.click('#glRun');state=await wait_state(p,'#glEngineState',{'COMPLETE','REVIEW REQUIRED'});ck('Architecture engine completes',state=='COMPLETE',state)
        gl=await p.evaluate('JSON.parse(JSON.stringify(window.GOALOS_GLOBAL_LAUNCH.getState()))')
        ck('Ten mission systems',len(gl.get('modules',[]))==10,str(len(gl.get('modules',[]))))
        ck('Fifty bounded jobs',len(gl.get('jobs',[]))>=50,str(len(gl.get('jobs',[]))))
        ck('Eight scenarios',await p.evaluate('window.GOALOS_OMEGA_PRIME_API.getState().scenarios.length')==8)
        # Deep deliberation.
        await p.locator('#apex-council').scroll_into_view_if_needed();await p.select_option('#v13SearchBudget','10000');await p.select_option('#v13ObjectiveLens','verified-upside');await p.click('#v13RunDeliberation');await p.wait_for_timeout(1500)
        dstate=(await p.inner_text('#v13DeliberationState')).strip();ck('Apex deliberation completes',dstate in ['COMPLETE','REVIEW REQUIRED'],dstate)
        ck('Deliberation champion generated',(await p.inner_text('#v13DeliberationChampion')).strip()!='Awaiting frontier',await p.inner_text('#v13DeliberationChampion'))
        ck('Tail metric generated','/' in await p.inner_text('#v13DeliberationTail'),await p.inner_text('#v13DeliberationTail'))
        ck('Deliberation audit log',await p.locator('#v13DeliberationLog>div').count()>=6,str(await p.locator('#v13DeliberationLog>div').count()))
        await p.screenshot(path=str(SCREENS/'04_deep_deliberation.png'),full_page=False)
        await p.click('#v13FalsifyChampion');await p.wait_for_timeout(1000);ck('Champion falsification runs',(await p.inner_text('#v13DeliberationState')).strip() in ['COMPLETE','REVIEW REQUIRED'])
        # Core evidence controls.
        await p.locator('#proof-governance').scroll_into_view_if_needed();ck('Twelve docket objects',await p.locator('.docket-object').count()==12);ck('Nine transaction receipts',await p.locator('#v12ReceiptGrid [data-receipt]').count()==9)
        # Live hero sync after scenario.
        await p.locator('#top').scroll_into_view_if_needed();await p.wait_for_timeout(300)
        ck('Live hero scenario sync','Awaiting' not in await p.inner_text('#v13LiveInstitution'),await p.inner_text('#v13LiveInstitution'))
        ck('Live hero capital sync','Awaiting' not in await p.inner_text('#v13LiveCapital'),await p.inner_text('#v13LiveCapital'))
        # layout/errors
        dims=await p.evaluate('({sw:document.documentElement.scrollWidth,cw:document.documentElement.clientWidth})');ck('Desktop no overflow',dims['sw']<=dims['cw']+1,str(dims))
        ck('No external runtime requests',len([u for u in req if u.startswith('http')])==0,str([u for u in req if u.startswith('http')][:5]))
        ck('No page errors',len(errs)==0,json.dumps(errs[:3]))
        ck('No console errors',len([x for x in logs if x['type']=='error'])==0,json.dumps([x for x in logs if x['type']=='error'][:3]))
        # Account change relock.
        await p.evaluate("window.ethereum.emit('accountsChanged',['0x3333333333333333333333333333333333333333'])");await p.wait_for_timeout(300);ck('Account change relocks',await p.locator('#agiClubAccessGate').is_visible())
        await ctx.close()
        # mobile
        c,m,ml,me,mr=await page_new(b,390,844);ck('Mobile preview visible',await m.locator('.v13-preview-copy h1').is_visible());await m.screenshot(path=str(SCREENS/'05_mobile_locked.png'),full_page=False);ck('Mobile direct owner unlocks',await unlock(m));await legal(m);await m.locator('#top').scroll_into_view_if_needed();await m.wait_for_timeout(250);md=await m.evaluate('({sw:document.documentElement.scrollWidth,cw:document.documentElement.clientWidth})');ck('Mobile no overflow',md['sw']<=md['cw']+1,str(md));ck('Mobile main dashboard visible',await m.locator('.v13-main-hero').is_visible());ck('Mobile no page errors',len(me)==0,json.dumps(me[:3]));ck('Mobile no console errors',len([x for x in ml if x['type']=='error'])==0,json.dumps([x for x in ml if x['type']=='error'][:3]));await m.screenshot(path=str(SCREENS/'06_mobile_main.png'),full_page=False);await c.close();await b.close()
    result={'release':'GoalOS Global Launch Ω v13.0.0-GL1','testedAt':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'passed':sum(t['passed'] for t in tests),'failed':sum(not t['passed'] for t in tests),'total':len(tests),'controls':tests}
    (OUT/'browser_qa.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result,indent=2));sys.exit(1 if result['failed'] else 0)

asyncio.run(main())
