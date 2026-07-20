(()=>{
 const btn=document.querySelector('.menu-btn'),nav=document.querySelector('.nav');
 if(btn&&nav){btn.addEventListener('click',()=>{const open=nav.classList.toggle('open');btn.setAttribute('aria-expanded',String(open));});nav.querySelectorAll('a').forEach(a=>a.addEventListener('click',()=>{nav.classList.remove('open');btn.setAttribute('aria-expanded','false');}));}
 const y=document.querySelector('[data-year]'); if(y)y.textContent=new Date().getFullYear();
 document.querySelectorAll('[data-copy]').forEach(b=>b.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(b.dataset.copy);const old=b.textContent;b.textContent=b.dataset.copied||'Copié';setTimeout(()=>b.textContent=old,1400);}catch(e){}}));
})();
