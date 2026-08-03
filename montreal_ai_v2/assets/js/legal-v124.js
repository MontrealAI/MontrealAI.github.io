
/* MONTREAL.AI public legal acceptance gate v124.0.0 */
(()=>{
'use strict';
const VERSION='MONTREAL-AI-PUBLIC-TERMS-2026-08-01-v124';
const KEY='montrealai.publicTerms.acceptance.v124';
let pending=null,opener=null,bypass=false;
const $=s=>document.querySelector(s);
function record(){try{return JSON.parse(localStorage.getItem(KEY)||'null')}catch{return null}}
function accepted(){const r=record();return !!(r&&r.version===VERSION&&r.accepted===true)}
function modal(){return $('[data-legal-gate-modal]')}
function checks(){return [...document.querySelectorAll('[data-legal-check]')]}
function update(){const b=$('[data-legal-accept]');if(b)b.disabled=!checks().every(x=>x.checked)}
function focusables(){const m=modal();return m?[...m.querySelectorAll('a[href],button:not([disabled]),input:not([disabled])')]:[]}
function open(el){const m=modal();if(!m)return;pending=el;opener=document.activeElement;m.classList.add('open');m.removeAttribute('hidden');document.body.style.overflow='hidden';checks().forEach(x=>x.checked=false);update();setTimeout(()=>{const f=focusables();(f[0]||m).focus()},0)}
function close(){const m=modal();if(!m)return;m.classList.remove('open');m.setAttribute('hidden','');document.body.style.overflow='';pending=null;if(opener&&opener.focus)opener.focus()}
function accept(){if(!checks().every(x=>x.checked))return;const lang=document.documentElement.lang||'en-CA';try{localStorage.setItem(KEY,JSON.stringify({version:VERSION,accepted:true,language:lang,acceptedAt:new Date().toISOString(),scope:'public interactive demonstrations and downloads only'}))}catch{}const el=pending;close();if(el){bypass=true;setTimeout(()=>{try{el.click()}finally{bypass=false}},0)}}
function requires(el){if(!el)return false;if(el.closest('[data-legal-gate-modal]'))return false;if(el.matches('[data-legal-gate],a[download]'))return true;const a=el.closest('a[href]');if(a){const h=(a.getAttribute('href')||'').toLowerCase();const t=(a.textContent||'').toLowerCase();if(/\.zip(?:$|[?#])/.test(h))return true;if(/goalos.*(?:demo|theatre|member)|(?:demo|theatre|member).*goalos|app\.ens\.domains|metamask|wallet/.test(h))return true;if(/launch demo|open demo|enter theatre|activate|connect wallet|member access/.test(t))return true}return false}
document.addEventListener('click',e=>{if(bypass||accepted())return;const target=e.target.closest('[data-legal-gate],a[download],a[href],button');if(!requires(target))return;e.preventDefault();e.stopImmediatePropagation();open(target)},true);
document.addEventListener('change',e=>{if(e.target.matches('[data-legal-check]'))update()});
document.addEventListener('click',e=>{const a=e.target.closest('[data-legal-accept]');if(a){e.preventDefault();accept()}const c=e.target.closest('[data-legal-close]');if(c){e.preventDefault();close()}const r=e.target.closest('[data-legal-reset]');if(r){e.preventDefault();try{localStorage.removeItem(KEY)}catch{}open(r)}});
document.addEventListener('keydown',e=>{const m=modal();if(!m||!m.classList.contains('open'))return;if(e.key==='Escape'){e.preventDefault();close();return}if(e.key==='Tab'){const fs=focusables();if(!fs.length)return;const first=fs[0],last=fs[fs.length-1];if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus()}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus()}}});
window.MONTREALAI_TERMS={version:VERSION,isAccepted:accepted,reset:()=>{try{localStorage.removeItem(KEY)}catch{}}};
})();
