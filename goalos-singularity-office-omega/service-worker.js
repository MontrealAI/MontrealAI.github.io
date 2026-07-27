'use strict';
const CACHE='goalos-singularity-office-v2-0-0-so2';
const ASSETS=["./", "index.html", "index-en.html", "index-fr.html", "styles.css", "i18n.js", "engine.js", "access.js", "app.js", "app.webmanifest", "offline.html", "assets/icon.svg", "assets/frontier_map.webp", "data/frontier-source-pack.json", "data/model-catalog.json", "data/publisher-public-key.json", "governance/LEGAL_INDEX.html"];
self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS)).then(()=>self.skipWaiting())));
self.addEventListener('activate',e=>e.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',e=>{if(e.request.method!=='GET')return;const u=new URL(e.request.url);if(u.origin!==location.origin)return;e.respondWith(fetch(e.request).then(r=>{if(r.ok){const c=r.clone();caches.open(CACHE).then(x=>x.put(e.request,c));}return r;}).catch(()=>caches.match(e.request).then(r=>r||caches.match('./offline.html'))));});
