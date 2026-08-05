/* MONTREAL.AI v11.0.0 — final interaction and accessibility guard.
   Loaded last on every public HTML surface. It does not replace page-specific
   behavior; it normalizes menu semantics and supplies a universal Escape close. */
(()=>{
  const q=(s,c=document)=>c.querySelector(s);
  const menu=q('.site-header .menu-btn,.site-header .menu-button,.site-header [data-menu],.site-header [data-v6-menu]');
  if(!menu)return;
  const panel=q('.mobile-panel,.mobile-nav,[data-mobile-nav],[data-mobile-panel],[data-v6-mobile]')||(menu.hasAttribute('data-menu')?q('[data-nav]'):null);
  if(!panel)return;
  if(!panel.id)panel.id='institutional-mobile-navigation';
  menu.setAttribute('aria-controls',panel.id);
  const isOpen=()=>panel.classList.contains('open');
  const sync=()=>{const open=isOpen();menu.setAttribute('aria-expanded',String(open));panel.setAttribute('aria-hidden',String(!open));};
  const close=(restore=true)=>{
    if(!isOpen())return;
    panel.classList.remove('open');
    document.body.classList.remove('menu-open');
    menu.setAttribute('aria-expanded','false');
    panel.setAttribute('aria-hidden','true');
    if((menu.textContent||'').trim()==='×')menu.textContent='☰';
    if(restore)menu.focus({preventScroll:true});
  };
  sync();
  menu.addEventListener('click',()=>setTimeout(sync,0));
  panel.addEventListener('click',e=>{if(e.target.closest('a[href]'))setTimeout(()=>close(false),0)});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&isOpen()){e.preventDefault();e.stopPropagation();close(true)}},true);
  addEventListener('resize',()=>{if(innerWidth>1180)close(false)},{passive:true});
})();
