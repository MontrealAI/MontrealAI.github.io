(()=>{
  const q=(s,c=document)=>c.querySelector(s),qa=(s,c=document)=>[...c.querySelectorAll(s)];
  const root=document.documentElement,header=q('.site-header'),progress=q('[data-progress]');
  const onScroll=()=>{const max=root.scrollHeight-innerHeight;const pct=max>0?scrollY/max*100:0;if(progress)progress.style.width=pct+'%';if(header)header.classList.toggle('scrolled',scrollY>18)};
  addEventListener('scroll',onScroll,{passive:true});onScroll();

  const menu=q('[data-menu]'),panel=q('[data-mobile-nav]');
  if(menu&&panel){menu.addEventListener('click',()=>{const open=panel.classList.toggle('open');menu.setAttribute('aria-expanded',String(open));menu.textContent=open?'×':'☰'});qa('a',panel).forEach(a=>a.addEventListener('click',()=>{panel.classList.remove('open');menu.setAttribute('aria-expanded','false');menu.textContent='☰'}));}

  const observer='IntersectionObserver'in window?new IntersectionObserver(entries=>{for(const e of entries){if(e.isIntersecting){e.target.classList.add('visible');observer.unobserve(e.target)}}},{threshold:.13,rootMargin:'0px 0px -30px'}):null;
  qa('.reveal').forEach(el=>observer?observer.observe(el):el.classList.add('visible'));

  const steps=qa('[data-proof-step]'),detail=q('[data-proof-detail]');
  const selectStep=i=>{if(!steps.length||!detail)return;steps.forEach((s,j)=>{const active=i===j;s.classList.toggle('active',active);s.setAttribute('aria-selected',String(active));s.tabIndex=active?0:-1});const s=steps[i],d=s.dataset;detail.innerHTML=`<div class="num">${d.num}</div><h3>${d.title}</h3><p>${d.body}</p><div class="law">${d.law}</div>`;};
  steps.forEach((s,i)=>{s.addEventListener('click',()=>selectStep(i));s.addEventListener('keydown',e=>{if(!['ArrowRight','ArrowDown','ArrowLeft','ArrowUp','Home','End'].includes(e.key))return;e.preventDefault();let n=i;if(e.key==='Home')n=0;else if(e.key==='End')n=steps.length-1;else n=(i+(e.key==='ArrowRight'||e.key==='ArrowDown'?1:-1)+steps.length)%steps.length;selectStep(n);steps[n].focus()})});if(steps.length)selectStep(0);

  if(matchMedia('(pointer:fine) and (prefers-reduced-motion:no-preference)').matches){qa('.tilt').forEach(card=>{card.addEventListener('pointermove',e=>{const r=card.getBoundingClientRect(),x=(e.clientX-r.left)/r.width-.5,y=(e.clientY-r.top)/r.height-.5;card.style.transform=`perspective(900px) rotateX(${(-y*4).toFixed(2)}deg) rotateY(${(x*5).toFixed(2)}deg) translateY(-2px)`});card.addEventListener('pointerleave',()=>card.style.transform='')});
    const hero=q('.hero');if(hero)hero.addEventListener('pointermove',e=>{const x=(e.clientX/innerWidth-.5)*12,y=(e.clientY/innerHeight-.5)*8;hero.style.backgroundPosition=`calc(50% + ${x}px) calc(50% + ${y}px)`});
  }

  qa('[data-count]').forEach(el=>{const target=Number(el.dataset.count||0),suffix=el.dataset.suffix||'';let done=false;const animate=()=>{if(done)return;done=true;const start=performance.now(),dur=1100;const tick=t=>{const p=Math.min(1,(t-start)/dur),v=Math.round(target*(1-Math.pow(1-p,3)));el.textContent=v.toLocaleString() + suffix;if(p<1)requestAnimationFrame(tick)};requestAnimationFrame(tick)};if(observer){const o=new IntersectionObserver(es=>{if(es[0].isIntersecting){animate();o.disconnect()}},{threshold:.5});o.observe(el)}else animate()});

  const year=q('[data-year]');if(year)year.textContent=new Date().getFullYear();
})();
