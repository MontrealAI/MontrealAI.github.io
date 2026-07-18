const CACHE='goalos-agi-club-web3-v7.0.0';
const IMMUTABLE=['./assets/app.css','./assets/app.bundle.js','./assets/vault-provider.js','./assets/flagship.webp','./assets/proof-to-value.webp','./manifest.webmanifest'];
self.addEventListener('install',event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(IMMUTABLE)).then(()=>self.skipWaiting())));
self.addEventListener('activate',event=>event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(key=>key!==CACHE).map(key=>caches.delete(key)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',event=>{
  if(event.request.method!=='GET')return;
  const url=new URL(event.request.url);
  if(url.origin!==self.location.origin)return;
  const networkFirst=url.pathname.endsWith('/config/runtime.json')||url.pathname.endsWith('/index.html')||url.pathname.endsWith('/404.html')||url.pathname.endsWith('/');
  if(networkFirst){
    event.respondWith(fetch(event.request,{cache:'no-store'}).then(response=>{const copy=response.clone();caches.open(CACHE).then(cache=>cache.put(event.request,copy));return response}).catch(()=>caches.match(event.request)));
    return;
  }
  event.respondWith(caches.match(event.request).then(hit=>hit||fetch(event.request).then(response=>{if(response.ok){const copy=response.clone();caches.open(CACHE).then(cache=>cache.put(event.request,copy))}return response})));
});
