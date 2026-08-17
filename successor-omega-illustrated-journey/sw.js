const CACHE='successor-omega-direct-gate-v1-0-2';
const SHELL=['./','index.html','styles.css','config.js','keccak.js','delivery.js','app.js','manifest.webmanifest','assets/icon.svg','assets/successor-omega-cover.webp','assets/social-card.jpg','protected/manifest.json'];
const NETWORK_FIRST=['/index.html','/config.js','/keccak.js','/delivery.js','/app.js','/protected/manifest.json'];
self.addEventListener('install',event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(SHELL)).then(()=>self.skipWaiting())));
self.addEventListener('activate',event=>event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(key=>key!==CACHE).map(key=>caches.delete(key)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',event=>{
  if(event.request.method!=='GET')return;
  const url=new URL(event.request.url); if(url.origin!==self.location.origin)return;
  if(url.pathname.includes('/protected/')&&!url.pathname.endsWith('/manifest.json')){event.respondWith(fetch(event.request,{cache:'no-store'}));return;}
  if(url.pathname.endsWith('/')||NETWORK_FIRST.some(path=>url.pathname.endsWith(path))){event.respondWith(fetch(event.request,{cache:'no-store'}).then(response=>{if(response.ok){const copy=response.clone();caches.open(CACHE).then(cache=>cache.put(event.request,copy));}return response;}).catch(()=>caches.match(event.request)));return;}
  event.respondWith(caches.match(event.request).then(hit=>hit||fetch(event.request).then(response=>{if(response.ok){const copy=response.clone();caches.open(CACHE).then(cache=>cache.put(event.request,copy));}return response;})));
});
