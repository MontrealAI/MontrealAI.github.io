#!/usr/bin/env python3
import asyncio, json, pathlib, datetime, sys
from playwright.async_api import async_playwright
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:8765/'
OUT=ROOT/'qa'/'BROWSER_QA_REPORT.json'
EXCLUDE={'goalos-documents/products/GoalOS_Autonomous_Money_Machine_OMEGA_Civilization_Value_Ceiling_v10_0_0.html'}
PAGES=sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.html') if p.relative_to(ROOT).as_posix() not in EXCLUDE and '_Downloads' not in p.as_posix())
VIEWPORTS={'desktop':{'width':1440,'height':1000},'mobile':{'width':390,'height':844}}
SHOTS={
 ('index.html','desktop'):'home_desktop.png',('index.html','mobile'):'home_mobile.png',
 ('en/index.html','desktop'):'home_english_desktop.png',('goalos/index.html','desktop'):'goalos_desktop.png',
 ('proof/index.html','desktop'):'proof_desktop.png',('goalos-legal/index.html','mobile'):'legal_mobile.png',
 ('contact/index.html','mobile'):'contact_mobile.png',('en/contact/index.html','mobile'):'contact_english_mobile.png'
}
async def main():
 results=[]
 async with async_playwright() as pw:
  browser=await pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
  for vp_name,vp in VIEWPORTS.items():
   context=await browser.new_context(viewport=vp,device_scale_factor=1)
   page=await context.new_page()
   state={'errors':[]}
   page.on('pageerror',lambda exc: state['errors'].append('pageerror: '+str(exc)))
   page.on('console',lambda msg: state['errors'].append(f'console {msg.type}: {msg.text}') if msg.type=='error' else None)
   for i,page_path in enumerate(PAGES,1):
    state['errors']=[]; response=None
    try:
     response=await page.goto(BASE+page_path,wait_until='domcontentloaded',timeout=5000)
     await page.wait_for_timeout(25)
     data=await page.evaluate('''() => ({
       title: document.title,
       lang: document.documentElement.lang,
       h1Count: document.querySelectorAll('h1').length,
       overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
       brokenImages: [...document.images].filter(i=>i.complete && i.naturalWidth===0).map(i=>i.getAttribute('src')),
       menuButton: !!document.querySelector('.menu-btn'),
       menuNav: !!document.querySelector('.nav')
     })''')
     menu_ok=True
     if vp_name=='mobile' and data['menuButton'] and data['menuNav']:
      try:
       await page.locator('.menu-btn').click(timeout=1000)
       menu_ok=await page.locator('.nav').evaluate("e=>e.classList.contains('open')")
       if menu_ok:
        await page.locator('.menu-btn').click(timeout=1000)
      except Exception as e:
       menu_ok=False; state['errors'].append('menu: '+str(e))
     shot=SHOTS.get((page_path,vp_name))
     if shot:
      await page.screenshot(path=str(ROOT/'qa'/shot),full_page=True,timeout=10000)
     result={'page':page_path,'viewport':vp_name,'status':response.status if response else None,'title':data['title'],'lang':data['lang'],'h1_count':data['h1Count'],'overflow_px':data['overflow'],'broken_images':data['brokenImages'],'menu_ok':menu_ok,'errors':list(state['errors'])}
    except Exception as e:
     result={'page':page_path,'viewport':vp_name,'status':response.status if response else None,'title':'','lang':'','h1_count':0,'overflow_px':0,'broken_images':[],'menu_ok':False,'errors':list(state['errors'])+[str(e)]}
    results.append(result)
    if i%10==0 or i==len(PAGES): print(f'{vp_name}: {i}/{len(PAGES)}',flush=True)
   await page.close(); await context.close()
  await browser.close()
 report={'schema':'montrealai.goalos.browser-qa.v1','generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pages_tested':len(PAGES),'scenarios':len(results),'status_failures':[r for r in results if r['status']!=200],'runtime_error_scenarios':[r for r in results if r['errors']],'overflow_failures':[r for r in results if r['overflow_px']>0],'broken_image_scenarios':[r for r in results if r['broken_images']],'language_failures':[r for r in results if not r['lang']],'h1_failures':[r for r in results if r['h1_count']!=1],'mobile_menu_failures':[r for r in results if r['viewport']=='mobile' and not r['menu_ok']],'results':results}
 OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps({k:(len(v) if isinstance(v,list) else v) for k,v in report.items() if k!='results'},indent=2,ensure_ascii=False))
if __name__=='__main__': asyncio.run(main())
