(function(){
  'use strict';
  const M=window.GoalOSMembership;
  async function run(){
    try{
      const session=await M.validateSession({interactive:false});
      const style=document.getElementById('goalos-guard-style');if(style)style.remove(); document.documentElement.classList.remove('goalos-guarded'); document.documentElement.style.visibility='visible';
      window.dispatchEvent(new CustomEvent('goalos:member-verified',{detail:session}));
      try{window.parent!==window&&window.parent.postMessage({type:'goalos:member-verified',session:{name:session.name,address:session.address,tier:session.tier,capabilities:session.capabilities}},location.origin);}catch{}
      const badge=document.getElementById('goalos-member-badge');
      if(badge)badge.textContent=`${session.name} · ${session.tier}`;
      const banner=document.createElement('div');banner.id='goalos-static-access-banner';banner.textContent=`AGI Club verified: ${session.name} · ${session.tier}`;banner.style.cssText='position:fixed;right:12px;bottom:12px;z-index:99999;background:rgba(5,11,21,.92);color:#f5d98f;border:1px solid rgba(245,217,143,.35);padding:8px 11px;border-radius:999px;font:700 12px system-ui;box-shadow:0 12px 30px rgba(0,0,0,.25)';
      if(document.body&&!document.getElementById(banner.id))document.body.appendChild(banner);
    }catch(err){
      M.clearSession();
      const candidate=String(window.GOALOS_GUARD_HOME||'../index.html#access'); const home=/^(?:\.\.\/|\.\/|\/)[^:]*$/.test(candidate)?candidate:'../index.html#access';
      document.documentElement.classList.remove('goalos-guarded'); document.documentElement.style.visibility='visible';
      document.body.innerHTML=`<main style="min-height:100vh;display:grid;place-items:center;padding:24px;background:#06101d;color:#edf5ff;font-family:system-ui"><section style="max-width:620px;border:1px solid rgba(245,217,143,.25);border-radius:24px;background:#0b1a2f;padding:28px;box-shadow:0 30px 90px rgba(0,0,0,.35)"><div style="font:800 13px system-ui;letter-spacing:.16em;text-transform:uppercase;color:#efcf83">AGI Club verification required</div><h1 style="font:800 38px Georgia;margin:12px 0">Reconnect MetaMask</h1><p style="line-height:1.65;color:#afbed0">${String(err.message||err).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]))}</p><a href="${home}" style="display:inline-block;margin-top:12px;background:#b9801d;color:white;text-decoration:none;font-weight:800;padding:12px 16px;border-radius:13px">Verify membership</a></section></main>`;
    }
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',run);else run();
})();
