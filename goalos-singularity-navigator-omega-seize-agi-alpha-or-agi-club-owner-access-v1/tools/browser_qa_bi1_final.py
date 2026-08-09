import asyncio, json
from pathlib import Path
from playwright.async_api import async_playwright
SITE=Path(__file__).resolve().parents[1]
HTML=(SITE/'index.html').read_text(encoding='utf-8')
OUT=SITE/'preview'; OUT.mkdir(exist_ok=True)
WALLET='0x1234567890abcdef1234567890abcdef12345678'; OTHER='0x9999999999999999999999999999999999999999'; WRAPPER='0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401'
POLY="""(()=>{const s=new Map();const ls={getItem:k=>s.has(String(k))?s.get(String(k)):null,setItem:(k,v)=>s.set(String(k),String(v)),removeItem:k=>s.delete(String(k)),clear:()=>s.clear(),key:i=>Array.from(s.keys())[i]??null,get length(){return s.size}};Object.defineProperty(window,'localStorage',{value:ls,configurable:true});})();"""
def wa(a): return '0x'+'0'*24+a[2:].lower()
def wu(n): return hex(n)[2:].rjust(64,'0')
def mock(mode='direct'):
    registry_owner = WALLET if mode=='direct' else (WRAPPER if mode in ('wrapped','expired') else OTHER)
    wrapped_owner = WALLET if mode in ('wrapped','expired') else OTHER
    exp=1 if mode=='expired' else 4102444800
    return f"""(()=>{{const l={{}},wallet='{WALLET}',ownerWord='{wa(registry_owner)}',wrappedWord='{wa(wrapped_owner)}',getData='0x'+wrappedWord.slice(2)+'{wu(0)}'+'{wu(exp)}';window.ethereum={{request:async({{method,params}})=>{{if(method==='eth_requestAccounts'||method==='eth_accounts')return[wallet];if(method==='eth_chainId')return'0x1';if(method==='wallet_switchEthereumChain')return null;if(method==='eth_blockNumber')return'0x1312d00';if(method==='eth_signTypedData_v4')return'0x'+'cd'.repeat(65);if(method==='personal_sign')return'0x'+'ab'.repeat(65);if(method==='eth_call'){{const data=(params?.[0]?.data||'').toLowerCase();if(data.startsWith('0x02571be3'))return ownerWord;if(data.startsWith('0x6352211e'))return wrappedWord;if(data.startsWith('0x0178fe3f'))return getData;return'0x'+'0'.repeat(64)}}throw new Error('Unsupported '+method)}},on:(e,cb)=>{{(l[e]||(l[e]=[])).push(cb)}},emit:(e,...a)=>{{(l[e]||[]).forEach(cb=>cb(...a))}}}}}})();"""
async def new_context(browser, mode='direct', language='en-CA', viewport=None):
    c=await browser.new_context(viewport=viewport or {'width':1600,'height':1000}, locale=language, accept_downloads=True)
    await c.add_init_script(POLY); await c.add_init_script(mock(mode)); return c
async def load(page): await page.set_content(HTML,wait_until='load')
async def gate(page,label='founder'):
    await page.fill('#agiClubLabelInput',label)
    for sel in ['#agiClubOwnerConfirm','#agiClubLicenseConfirm','#agiClubBoundaryConfirm','#agiClubLanguageConfirm','#agiClubNoSecretsConfirm']:
        await page.check(sel)
async def unlock(page,label='founder'):
    await gate(page,label); await page.click('#agiClubVerifyBtn'); await page.wait_for_function("!document.body.classList.contains('agc-locked')")
def add(results,name,passed,detail=''): results.append({'name':name,'passed':bool(passed),'detail':detail})
async def run():
    r=[]
    async with async_playwright() as pw:
        b=await pw.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage'])
        # English locked
        c=await new_context(b,'direct','en-CA'); p=await c.new_page(); errs=[]; ext=[]
        p.on('pageerror',lambda e:errs.append(str(e))); p.on('console',lambda m:errs.append(m.text) if m.type=='error' else None); p.on('request',lambda q:ext.append(q.url) if q.url.startswith(('http://','https://')) else None)
        await load(p)
        add(r,'English browser defaults to English',(await p.locator('html').get_attribute('data-lang'))=='en')
        add(r,'pre-access bilingual selector visible',await p.locator('.bi-language-dock').is_visible())
        add(r,'English AGI Club heading visible','AGI Club Owner Access' in await p.locator('#agiClubAccessGate').inner_text())
        await p.fill('#agiClubLabelInput','founder'); await p.check('#agiClubOwnerConfirm'); await p.click('.bi-language-dock [data-bi-lang="fr"]')
        add(r,'locked switch to French',(await p.locator('html').get_attribute('data-lang'))=='fr')
        add(r,'locked label preserved',(await p.locator('#agiClubLabelInput').input_value())=='founder')
        add(r,'locked checkbox preserved',await p.locator('#agiClubOwnerConfirm').is_checked())
        await p.click('.bi-language-dock [data-bi-lang="en"]'); add(r,'locked switch back to English',(await p.locator('html').get_attribute('data-lang'))=='en')
        add(r,'locked page errors zero',not errs,str(errs)); add(r,'locked external requests zero',not ext,str(ext))
        await c.close()
        # Direct owner and all routes in both languages
        c=await new_context(b,'direct','en-CA'); p=await c.new_page(); errs=[]; ext=[]
        p.on('pageerror',lambda e:errs.append(str(e))); p.on('console',lambda m:errs.append(m.text) if m.type=='error' else None); p.on('request',lambda q:ext.append(q.url) if q.url.startswith(('http://','https://')) else None)
        await load(p); await unlock(p); add(r,'direct owner unlock',True)
        await p.evaluate("document.querySelector('#ownerAccessDialog')?.close(); document.querySelector('#demoBtn').click(); document.querySelector('#runMission').click()")
        await p.wait_for_timeout(5200)
        add(r,'12 architectures',await p.locator('#heroScenarios').inner_text()=='12')
        add(r,'21 bounded AGI Jobs',(await p.locator('#jobs').inner_text()).count('Budget index')==21)
        add(r,'top language selector visible',await p.locator('.bi-top-language').is_visible())
        objective=await p.locator('#missionText').input_value()
        await p.click('.bi-top-language [data-bi-lang=fr]'); await p.wait_for_timeout(100)
        add(r,'unlocked switch to French',(await p.locator('html').get_attribute('data-lang'))=='fr')
        add(r,'mission objective preserved',(await p.locator('#missionText').input_value())==objective)
        routes=[]
        for btn in await p.locator('nav button[data-route]').all():
            route=await btn.get_attribute('data-route')
            if route in routes: continue
            routes.append(route); await p.evaluate('(x)=>document.querySelector(`nav button[data-route=${x}]`).click()',route); await p.wait_for_timeout(30)
            add(r,f'French route {route}',await p.locator(f'#view-{route}').evaluate("e=>e.classList.contains('active')"))
        add(r,'16 French routes',len(routes)==16,str(routes))
        await p.evaluate("document.querySelector('button[data-route=masterclass]').click()")
        mc=await p.locator('#view-masterclass').inner_text()
        add(r,'222-page bilingual MasterClass advertised','222' in mc and ('bilingue' in mc.lower() or 'bilingual' in mc.lower()))
        href=await p.locator('#view-masterclass a[href*="v3_2_0_SN5_BI1.pdf"]').get_attribute('href')
        add(r,'current MasterClass link',bool(href),str(href))
        await p.click('.bi-top-language [data-bi-lang=en]'); await p.wait_for_timeout(100)
        add(r,'unlocked switch back to English',(await p.locator('html').get_attribute('data-lang'))=='en')
        add(r,'unlocked errors zero',not errs,str(errs)); add(r,'unlocked external requests zero',not ext,str(ext))
        await c.close()
        # wrapped and denials
        for mode,label,expected in [('wrapped','wrapped',True),('wrong','wrong',False),('expired','expired',False)]:
            c=await new_context(b,mode,'en-CA',{'width':1280,'height':850}); p=await c.new_page(); await load(p); await gate(p,label); await p.click('#agiClubVerifyBtn'); await p.wait_for_timeout(600)
            unlocked=not await p.locator('body').evaluate("e=>e.classList.contains('agc-locked')")
            add(r,f'{mode} ownership expected state',unlocked==expected,f'unlocked={unlocked}')
            await c.close()
        # mobile
        c=await new_context(b,'direct','fr-CA',{'width':390,'height':844}); p=await c.new_page(); errs=[]; p.on('pageerror',lambda e:errs.append(str(e))); await load(p)
        add(r,'mobile defaults French',(await p.locator('html').get_attribute('data-lang'))=='fr')
        add(r,'mobile selector visible',await p.locator('.bi-language-dock').is_visible())
        add(r,'mobile locked overflow',(await p.evaluate('document.documentElement.scrollWidth-document.documentElement.clientWidth'))<=1)
        await unlock(p,'mobile'); await p.evaluate("document.querySelector('#ownerAccessDialog')?.close()")
        add(r,'mobile unlocked overflow',(await p.evaluate('document.documentElement.scrollWidth-document.documentElement.clientWidth'))<=1)
        add(r,'mobile errors zero',not errs,str(errs)); await c.close(); await b.close()
    # static artifacts
    pdf=SITE/'research'/'GoalOS_Singularity_Navigator_Omega_MasterClass_Tools_to_Ride_the_Singularity_by_Vincent_Boucher_v3_2_0_SN5_BI1.pdf'
    add(r,'current 222-page MasterClass file exists',pdf.exists(),str(pdf))
    out={'release':'v3.2.0-SN5-BI1','passed':all(x['passed'] for x in r),'assertions':len(r),'results':r}
    (SITE/'documentation'/'FINAL_BILINGUAL_BROWSER_QA.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Final Bilingual Browser QA','',f"**Release:** {out['release']}",f"**Result:** {'PASS' if out['passed'] else 'FAIL'}",f"**Assertions:** {len(r)}",'', '| Control | Result | Detail |','|---|---:|---|']
    for x in r: lines.append(f"| {x['name'].replace('|','/')} | {'PASS' if x['passed'] else 'FAIL'} | {str(x['detail']).replace('|','/')} |")
    (SITE/'documentation'/'FINAL_BILINGUAL_BROWSER_QA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2,ensure_ascii=False))
    raise SystemExit(0 if out['passed'] else 1)
asyncio.run(run())
