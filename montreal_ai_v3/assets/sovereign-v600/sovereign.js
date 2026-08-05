(()=>{
  const q=(s,c=document)=>c.querySelector(s),qa=(s,c=document)=>[...c.querySelectorAll(s)];
  const root=document.documentElement,progress=q('[data-progress]'),header=q('.site-header');
  const scroll=()=>{const max=root.scrollHeight-innerHeight; if(progress) progress.style.width=(max?scrollY/max*100:0)+'%'; if(header) header.classList.toggle('scrolled',scrollY>16)};
  addEventListener('scroll',scroll,{passive:true});scroll();

  const menu=q('[data-v6-menu]'),mobile=q('[data-v6-mobile]');
  if(menu&&mobile){menu.onclick=()=>{const o=mobile.classList.toggle('open');menu.setAttribute('aria-expanded',String(o));menu.textContent=o?'×':'☰'};qa('a',mobile).forEach(a=>a.onclick=()=>{mobile.classList.remove('open');menu.setAttribute('aria-expanded','false');menu.textContent='☰'})}

  const cmd=q('[data-v6-command]'),openers=qa('[data-v6-command-open]'),closer=q('[data-v6-command-close]'),cmdInput=q('[data-v6-command-input]');
  const openCmd=()=>{if(!cmd)return;cmd.classList.add('open');cmd.setAttribute('aria-hidden','false');setTimeout(()=>cmdInput?.focus(),30)};
  const closeCmd=()=>{if(!cmd)return;cmd.classList.remove('open');cmd.setAttribute('aria-hidden','true')};
  openers.forEach(b=>b.addEventListener('click',openCmd));closer?.addEventListener('click',closeCmd);cmd?.addEventListener('click',e=>{if(e.target===cmd)closeCmd()});
  addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();cmd?.classList.contains('open')?closeCmd():openCmd()}if(e.key==='Escape')closeCmd()});
  if(cmdInput){cmdInput.addEventListener('input',()=>{const v=cmdInput.value.trim().toLowerCase();qa('a',q('.v6-command-links')).forEach(a=>a.hidden=!!v&&!a.textContent.toLowerCase().includes(v))})}

  const observer='IntersectionObserver'in window?new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');observer.unobserve(e.target)}}),{threshold:.08,rootMargin:'0px 0px -25px'}):null;
  qa('.reveal').forEach(el=>observer?observer.observe(el):el.classList.add('visible'));

  const steps=qa('[data-v6-step]'),stage=q('[data-v6-stage]');
  const render=i=>{if(!steps.length||!stage)return;steps.forEach((s,j)=>{const a=i===j;s.classList.toggle('active',a);s.setAttribute('aria-selected',String(a));s.tabIndex=a?0:-1});const d=steps[i].dataset;stage.innerHTML=`<div class="num">${d.num||''}</div><h3>${d.title||''}</h3><p>${d.body||''}</p><div class="law">${d.law||''}</div><div class="microgrid"><div><b>Authority boundary</b><p>${d.authority||'Explicitly assigned before consequential action.'}</p></div><div><b>Failure mode</b><p>${d.failure||'Unsupported output cannot progress by confidence alone.'}</p></div></div>`};
  steps.forEach((s,i)=>{s.onclick=()=>render(i);s.onkeydown=e=>{if(!['ArrowRight','ArrowDown','ArrowLeft','ArrowUp','Home','End'].includes(e.key))return;e.preventDefault();let n=i;if(e.key==='Home')n=0;else if(e.key==='End')n=steps.length-1;else n=(i+(e.key==='ArrowRight'||e.key==='ArrowDown'?1:-1)+steps.length)%steps.length;render(n);steps[n].focus()}});if(steps.length)render(0);

  qa('[data-count]').forEach(el=>{const target=Number(el.dataset.count||0),suffix=el.dataset.suffix||'';let done=false;const go=()=>{if(done)return;done=true;const st=performance.now(),dur=1000;const tick=t=>{const p=Math.min(1,(t-st)/dur),v=Math.round(target*(1-Math.pow(1-p,3)));el.textContent=v.toLocaleString()+suffix;if(p<1)requestAnimationFrame(tick)};requestAnimationFrame(tick)};if(observer){const o=new IntersectionObserver(es=>{if(es[0].isIntersecting){go();o.disconnect()}},{threshold:.35});o.observe(el)}else go()});

  const search=q('[data-corpus-search]'),filters=qa('[data-corpus-filter]'),records=qa('[data-record]');let active='all';
  const filter=()=>{const term=(search?.value||'').trim().toLowerCase();records.forEach(r=>{const tags=(r.dataset.tags||'').split(',');const okTag=active==='all'||tags.includes(active);const okText=!term||r.textContent.toLowerCase().includes(term);r.hidden=!(okTag&&okText)});const count=records.filter(r=>!r.hidden).length;const out=q('[data-result-count]');if(out)out.textContent=String(count)};
  search?.addEventListener('input',filter);filters.forEach(b=>b.onclick=()=>{active=b.dataset.corpusFilter||'all';filters.forEach(x=>x.classList.toggle('active',x===b));filter()});filter();

  qa('[data-copy]').forEach(b=>b.onclick=async()=>{const text=b.dataset.copy||'';try{await navigator.clipboard.writeText(text);const old=b.textContent;b.textContent='Copied';setTimeout(()=>b.textContent=old,1200)}catch{}});
  qa('[data-year]').forEach(y=>y.textContent=String(new Date().getFullYear()));

  if(matchMedia('(pointer:fine) and (prefers-reduced-motion:no-preference)').matches){qa('.v6-card,.v6-engine').forEach(card=>{card.addEventListener('pointermove',e=>{const r=card.getBoundingClientRect(),x=(e.clientX-r.left)/r.width-.5,y=(e.clientY-r.top)/r.height-.5;card.style.transform=`perspective(1000px) rotateX(${(-y*3).toFixed(2)}deg) rotateY(${(x*4).toFixed(2)}deg) translateY(-4px)`});card.addEventListener('pointerleave',()=>card.style.transform='')})}
})();
