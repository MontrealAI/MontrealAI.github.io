import asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright
HTML=Path('/mnt/data/GoalOS_Global_Launch_Omega_AGI_CLUB_GITHUB_PAGES_FINAL_v13_0_0_GL1_2026-07-24/goalos-global-launch-omega-agi-club-owner-access/index.html').read_text('utf-8')
W='0x1111111111111111111111111111111111111111';O='0x2222222222222222222222222222222222222222';MW='0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401';SW='0x0635513f179D50A207757E05759CbD106d7dFcE8'
MOCK=f"""(()=>{{const h={{}},w='{W}',o='{O}',mw='{MW}',sw='{SW}';window.__mockOwnerMode='direct';const wa=a=>'0x'+'0'.repeat(24)+a.slice(2).toLowerCase(),uw=n=>BigInt(n).toString(16).padStart(64,'0');window.ethereum={{request:async({{method,params}})=>{{const m=window.__mockOwnerMode||'direct';if(method==='eth_requestAccounts'||method==='eth_accounts')return[w];if(method==='eth_chainId')return'0x1';if(method==='wallet_switchEthereumChain')return null;if(method==='personal_sign')return'0x'+'11'.repeat(65);if(method==='eth_blockNumber')return'0x123456';if(method==='eth_call'){{const d=String(params?.[0]?.data||'').toLowerCase();if(d.startsWith('0x02571be3')){{if(m==='direct')return wa(w);if(['wrapped','expired','wrongwrapped'].includes(m))return wa(mw);if(m==='nonmain')return wa(sw);return wa(o)}}if(d.startsWith('0x6352211e'))return wa(m==='wrongwrapped'?o:w);if(d.startsWith('0x0178fe3f'))return wa(m==='wrongwrapped'?o:w)+uw(0)+uw(m==='expired'?1600000000:4102444800);return'0x'+'0'.repeat(64)}}throw Error('Unsupported mock '+method)}},on:(n,f)=>(h[n]||(h[n]=[])).push(f)}}}})();"""

async def run_case(browser,mode,label,expected):
    ctx=await browser.new_context(viewport={'width':1600,'height':1000})
    p=await ctx.new_page(); p.set_default_timeout(30000)
    errs=[];console=[]
    p.on('pageerror',lambda e:errs.append(str(e)))
    p.on('console',lambda m: console.append(m.text) if m.type=='error' else None)
    await p.evaluate(MOCK)
    await p.set_content(HTML,wait_until='domcontentloaded',timeout=120000)
    await p.wait_for_timeout(1000)
    await p.click('#agiClubOpenProofBtn')
    await p.evaluate('(m)=>window.__mockOwnerMode=m',mode)
    await p.fill('#agiClubLabelInput',label)
    for s in ['#agiClubOwnerConfirm','#agiClubLicenseConfirm','#agiClubBoundaryConfirm']:
        if not await p.locator(s).is_checked(): await p.check(s)
    if await p.locator('#agiClubVerifyBtn').is_disabled():
        status=await p.locator('#agiClubGateStatus').inner_text()
        await ctx.close()
        ok=(False==expected and not errs and not console)
        return {'name':f"Owner mode {mode}"+(' rejects nested label' if '.' in label else ''),'passed':ok,'detail':f'unlocked=False; expected={expected}; status={status or "Verify disabled by validation"}; pageErrors={errs}; consoleErrors={console}'}
    await p.click('#agiClubVerifyBtn')
    unlocked=False
    for _ in range(250):
        if not await p.locator('#agiClubAccessGate').is_visible(): unlocked=True; break
        if await p.locator('#agiClubGateStatus').get_attribute('data-state')=='bad': break
        await p.wait_for_timeout(80)
    status=await p.locator('#agiClubGateStatus').inner_text()
    await ctx.close()
    ok=(unlocked==expected and not errs and not console)
    return {'name':f"Owner mode {mode}"+(' rejects nested label' if '.' in label else ''),'passed':ok,'detail':f'unlocked={unlocked}; expected={expected}; status={status}; pageErrors={errs}; consoleErrors={console}'}

async def main():
    cases=[('direct','alpha-council',True),('wrapped','alpha-council',True),('wrong','alpha-council',False),('wrongwrapped','alpha-council',False),('expired','alpha-council',False),('nonmain','alpha-council',False)]
    controls=[]
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
        for case in cases:
            r=await run_case(browser,*case); controls.append(r); print(('PASS' if r['passed'] else 'FAIL'),r['name'],r['detail'],flush=True)
        await browser.close()
    result={'release':'GoalOS Global Launch Ω v13.0.0-GL1','testedAt':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'passed':sum(c['passed'] for c in controls),'failed':sum(not c['passed'] for c in controls),'total':len(controls),'controls':controls}
    out=Path('/mnt/data/goalos_v13_release_assets/qa/owner_modes_qa.json'); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,ensure_ascii=False),'utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    raise SystemExit(1 if result['failed'] else 0)
asyncio.run(main())
