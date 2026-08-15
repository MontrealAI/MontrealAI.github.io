#!/usr/bin/env python3
import json, urllib.request, urllib.error

CONTRACT='0x495f947276749ce646f68ac8c248420045cb7b5e'
CREATOR='0x054a2e4b3b5ea2c62372e92358fdf7fb74b4f34a'
OWNER='0x049ffe5432fa375049c2ed19faccd9b31bb4523e'
TOKEN='2392630434290240917728431095880785304289144848761899072947382440480049463297'
BLOCK='0x1891d45'
EPS=['https://ethereum-rpc.publicnode.com','https://rpc.flashbots.net','https://eth.drpc.org','https://1rpc.io/eth']

def post(ep,payload):
    req=urllib.request.Request(ep,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','User-Agent':'MONTREAL.AI-rpc-diagnostic/1.0'},method='POST')
    try:
        with urllib.request.urlopen(req,timeout=45) as r:return {'status':r.status,'body':json.loads(r.read())}
    except urllib.error.HTTPError as e:return {'status':e.code,'body':e.read().decode(errors='replace')}
    except Exception as e:return {'error':repr(e)}

def data(owner):return '0x00fdd58e'+owner[2:].zfill(64)+hex(int(TOKEN))[2:].zfill(64)

for ep in EPS:
    print('\nENDPOINT',ep)
    for name,method,params in [
      ('chainId','eth_chainId',[]),
      ('code_at_block','eth_getCode',[CONTRACT,BLOCK]),
      ('creator_at_block','eth_call',[{'to':CONTRACT,'data':data(CREATOR)},BLOCK]),
      ('owner_at_block','eth_call',[{'to':CONTRACT,'data':data(OWNER)},BLOCK]),
      ('owner_latest','eth_call',[{'to':CONTRACT,'data':data(OWNER)},'latest']),
      ('owner_finalized','eth_call',[{'to':CONTRACT,'data':data(OWNER)},'finalized']),
    ]:
        print(name,json.dumps(post(ep,{'jsonrpc':'2.0','id':1,'method':method,'params':params}),ensure_ascii=False)[:4000])
