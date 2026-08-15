#!/usr/bin/env python3
import json, urllib.request, urllib.error, time

CONTRACT='0x495f947276749ce646f68ac8c248420045cb7b5e'
OWNER='0x049ffe5432fa375049c2ed19faccd9b31bb4523e'
TOKEN='2392630434290240917728431095880785304289144848761899072947382440480049463297'
EPS=[
 'https://ethereum-rpc.publicnode.com',
 'https://1rpc.io/eth/',
 'https://eth.drpc.org',
 'https://rpc.ankr.com/eth',
 'https://cloudflare-eth.com',
 'https://eth-mainnet.public.blastapi.io',
 'https://eth.merkle.io',
 'https://ethereum.blockpi.network/v1/rpc/public',
 'https://api.zan.top/node/v1/eth/mainnet/public',
 'https://rpc.payload.de',
 'https://rpc.mevblocker.io',
 'https://eth-mainnet.g.alchemy.com/v2/demo'
]

def post(ep,payload):
    req=urllib.request.Request(ep,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','User-Agent':'MONTREAL.AI-rpc-diagnostic/2.0'},method='POST')
    try:
        with urllib.request.urlopen(req,timeout=25) as r:
            raw=r.read().decode();
            try: body=json.loads(raw)
            except: body=raw[:2000]
            return {'status':r.status,'body':body}
    except urllib.error.HTTPError as e:return {'status':e.code,'body':e.read().decode(errors='replace')[:2000]}
    except Exception as e:return {'error':repr(e)}

def data(owner):return '0x00fdd58e'+owner[2:].zfill(64)+hex(int(TOKEN))[2:].zfill(64)

def rpc(i,method,params):return {'jsonrpc':'2.0','id':i,'method':method,'params':params}

for ep in EPS:
    print('\nENDPOINT',ep,flush=True)
    before=post(ep,rpc(1,'eth_getBlockByNumber',['finalized',False]))
    batch=post(ep,[rpc(2,'eth_call',[{'to':CONTRACT,'data':data(OWNER)},'finalized']),rpc(3,'eth_call',[{'to':CONTRACT,'data':data(OWNER)},'latest'])])
    after=post(ep,rpc(4,'eth_getBlockByNumber',['finalized',False]))
    def slim(x):
        s=json.dumps(x,ensure_ascii=False)
        return s[:3000]
    print('before',slim(before),flush=True)
    print('batch',slim(batch),flush=True)
    print('after',slim(after),flush=True)
    time.sleep(.25)
