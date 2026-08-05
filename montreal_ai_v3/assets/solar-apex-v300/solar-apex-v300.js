(()=>{
  const d=document,root=d.documentElement;
  root.classList.add('solar-apex-v300');
  d.querySelectorAll('[data-year]').forEach(el=>el.textContent=String(new Date().getFullYear()));
  d.querySelectorAll('.apex-reveal').forEach(el=>{el.classList.add('is-visible');el.style.opacity='1';el.style.transform='none'});
})();
