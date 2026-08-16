#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, os, re, sys, time, zipfile
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests

CONTRACT='0x495f947276749ce646f68ac8c248420045cb7b5e'
CREATOR='0x054a2e4b3b5ea2c62372e92358fdf7fb74b4f34a'
ZERO='0x0000000000000000000000000000000000000000'
DEAD='0x000000000000000000000000000000000000dead'
CHAIN_ID=1
EXCLUDED={159,495,523}
BASE=Path(os.environ.get('BASE_SNAPSHOT','base_snapshot'))
OUT=Path(os.environ.get('SNAPSHOT_OUT','snapshot_output'))
RPC_URLS=[x.strip() for x in os.environ.get('ETH_RPC_URLS','https://eth.llamarpc.com,https://rpc.flashbots.net,https://eth.drpc.org,https://ethereum-rpc.publicnode.com').split(',') if x.strip()]
BLOCKSCOUT='https://eth.blockscout.com/api'
TIMEOUT=float(os.environ.get('HTTP_TIMEOUT','45'))
RETRIES=int(os.environ.get('HTTP_RETRIES','6'))
SESSION=requests.Session(); SESSION.headers.update({'Accept':'application/json','User-Agent':'MONTREALAI-Genesis-556-Focused-Resolver/1.0'})

def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')
def norm(v:Any)->str|None:
    if isinstance(v,dict):
        for k in ('hash','address','address_hash'):
            z=norm(v.get(k))
            if z:return z
        return None
    if not isinstance(v,str):return None
    v=v.strip().lower()
    if not re.fullmatch(r'0x[0-9a-f]{40}',v) or v==ZERO:return None
    return v

def http_get(url:str,params:dict[str,Any]|None=None)->dict[str,Any]:
    err=None
    for n in range(RETRIES+1):
        try:
            r=SESSION.get(url,params=params,timeout=TIMEOUT)
            if r.status_code==429 or r.status_code>=500:
                raise RuntimeError(f'HTTP {r.status_code}: {r.text[:200]}')
            r.raise_for_status(); x=r.json()
            if not isinstance(x,dict):raise ValueError('JSON object expected')
            return x
        except Exception as e:
            err=e
            if n==RETRIES:break
            time.sleep(min(20,.7*(2**n)))
    raise RuntimeError(f'GET failed {url}: {err}')

def rpc_one(url:str,method:str,params:list[Any],tries:int=4)->Any:
    err=None
    for n in range(tries+1):
        try:
            r=SESSION.post(url,json={'jsonrpc':'2.0','id':1,'method':method,'params':params},timeout=TIMEOUT)
            if r.status_code in (403,429) or r.status_code>=500:raise RuntimeError(f'HTTP {r.status_code}')
            r.raise_for_status(); x=r.json()
            if x.get('error'):raise RuntimeError(str(x['error']))
            return x['result']
        except Exception as e:
            err=e
            if n==tries:break
            time.sleep(min(10,.5*(2**n)))
    raise RuntimeError(f'{url}: {method}: {err}')

def rpc_any(method:str,params:list[Any],preferred:str|None=None)->tuple[Any,str,list[dict[str,Any]]]:
    urls=([preferred] if preferred else [])+[x for x in RPC_URLS if x!=preferred]
    attempts=[]
    for u in urls:
        if not u:continue
        try:
            result=rpc_one(u,method,params,2); attempts.append({'url':u,'status':'PASS'}); return result,u,attempts
        except Exception as e: attempts.append({'url':u,'status':'FAIL','error':str(e)})
    raise RuntimeError(json.dumps(attempts))

def manifest()->list[dict[str,Any]]:
    nonces=[n for n in range(5,564) if n not in EXCLUDED]
    assert len(nonces)==556
    nonces[257],nonces[258]=nonces[258],nonces[257]
    c=int(CREATOR,16); out=[]
    for number,nonce in enumerate(nonces,1):
        tid=(c<<96)|(nonce<<40)|1; dec=str(tid)
        out.append({'canonical_number':number,'title':f'Crypto AI Art #{number:03d}','chain_id':1,'contract':CONTRACT,'standard':'ERC-1155','token_id_decimal':dec,'token_id_hex':f'0x{tid:064x}','creator_encoded':CREATOR,'internal_nonce':nonce,'edition_supply':1,'opensea_item_url':f'https://opensea.io/item/ethereum/{CONTRACT}/{dec}','id_basis':'corrected legacy sequence; nonces 5-563 excluding 159, 495, 523; 258/259 creation-order swap'})
    return out

def read_csv(p:Path)->list[dict[str,str]]:
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write_csv(p:Path,rows:list[dict[str,Any]],fields:list[str]):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def write_json(p:Path,x:Any):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def address_transfers(address:str,end_block:int)->list[dict[str,Any]]:
    rows=[]; page=1
    while True:
        x=http_get(BLOCKSCOUT,{'module':'account','action':'token1155tx','address':address,'contractaddress':CONTRACT,'startblock':0,'endblock':end_block,'page':page,'offset':10000,'sort':'asc'})
        result=x.get('result')
        if not isinstance(result,list):
            if str(x.get('status'))=='0' and isinstance(result,str) and 'No transactions' in result:return rows
            raise RuntimeError(f'legacy response for {address}: {str(x)[:500]}')
        rows.extend(r for r in result if isinstance(r,dict))
        if len(result)<10000:return rows
        page+=1
        if page>50:raise RuntimeError(f'pagination limit for {address}')

def discover(target:set[str],end_block:int)->tuple[dict[str,set[str]],list[dict[str,Any]],list[dict[str,Any]]]:
    cand={t:{CREATOR} for t in target}; queue=deque([CREATOR]); seen=set(); evidence=[]; errors=[]
    while queue:
        address=queue.popleft()
        if address in seen:continue
        seen.add(address)
        try: txs=address_transfers(address,end_block)
        except Exception as e: errors.append({'address':address,'error':str(e)});continue
        relevant=0
        for tx in txs:
            tid=str(tx.get('tokenID') or tx.get('tokenId') or tx.get('token_id') or '')
            if tid not in target:continue
            try:
                if int(str(tx.get('blockNumber') or '0'))>end_block:continue
            except:continue
            relevant+=1; fr=norm(tx.get('from')); to=norm(tx.get('to'))
            for a in (fr,to):
                if a:
                    cand[tid].add(a)
                    if a not in seen:queue.append(a)
            evidence.append({'token_id_decimal':tid,'block_number':str(tx.get('blockNumber') or ''),'transaction_hash':str(tx.get('hash') or tx.get('transactionHash') or ''),'log_index':str(tx.get('logIndex') or ''),'from_address':fr or ZERO,'to_address':to or ZERO,'token_value':str(tx.get('tokenValue') or tx.get('value') or '1'),'queried_address':address})
        print(f'Graph address {len(seen)}: {address}, relevant={relevant}, queue={len(queue)}',flush=True)
        if len(seen)>500:raise RuntimeError('focused graph exceeded 500 addresses')
    unique={(r['transaction_hash'],r['log_index'],r['token_id_decimal'],r['from_address'],r['to_address']):r for r in evidence}
    return cand,sorted(unique.values(),key=lambda r:(int(r['block_number'] or 0),int(r['log_index'] or 0),r['token_id_decimal'])),errors

def abi_balance(accounts:list[str],ids:list[int])->str:
    sel=bytes.fromhex('4e1273f4');n=len(accounts);a=n.to_bytes(32,'big')+b''.join(bytes.fromhex(x[2:]).rjust(32,b'\0') for x in accounts);i=n.to_bytes(32,'big')+b''.join(x.to_bytes(32,'big') for x in ids);head=(64).to_bytes(32,'big')+(64+len(a)).to_bytes(32,'big');return '0x'+(sel+head+a+i).hex()
def decode(raw:str)->list[int]:
    b=bytes.fromhex(raw[2:]);o=int.from_bytes(b[:32],'big');n=int.from_bytes(b[o:o+32],'big');s=o+32;return [int.from_bytes(b[s+32*k:s+32*(k+1)],'big') for k in range(n)]
def query(cand:dict[str,set[str]],block_hex:str)->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    pairs=[(tid,a) for tid in sorted(cand,key=int) for a in sorted(cand[tid])]; out=[]; attempts=[]
    for start in range(0,len(pairs),40):
        q=pairs[start:start+40]; data=abi_balance([a for _,a in q],[int(t) for t,_ in q]); raw,u,att=rpc_any('eth_call',[{'to':CONTRACT,'data':data},block_hex]); attempts.extend({'batch_start':start,**x} for x in att);vals=decode(raw)
        if len(vals)!=len(q):raise RuntimeError('balance length mismatch')
        out.extend({'token_id_decimal':t,'holder_address':a,'balance':v,'rpc_endpoint':u} for (t,a),v in zip(q,vals))
        print(f'Balances {min(start+40,len(pairs))}/{len(pairs)} via {u}',flush=True)
    return out,attempts

def main()->int:
    started=now(); OUT.mkdir(parents=True,exist_ok=True)
    base_status=json.loads((BASE/'snapshot/snapshot_status.json').read_text())
    block=json.loads((BASE/'snapshot/snapshot_block.json').read_text())
    block_num=int(block['block_number']); block_hex=hex(block_num); block_hash=str(block['block_hash']).lower()
    cross=[]
    for u in RPC_URLS:
        try:
            b=rpc_one(u,'eth_getBlockByNumber',[block_hex,False],2); h=str((b or {}).get('hash','')).lower(); cross.append({'url':u,'status':'PASS' if h==block_hash else 'MISMATCH','hash':h})
        except Exception as e:cross.append({'url':u,'status':'ERROR','error':str(e)})
    mani=manifest(); by_id={r['token_id_decimal']:r for r in mani}
    old=read_csv(BASE/'snapshot/token_holdings.csv')
    kept=[]
    for r in old:
        tid=r['token_id_decimal']
        if tid not in by_id:continue
        m=by_id[tid]
        kept.append({**r,'canonical_number':m['canonical_number'],'title':m['title'],'token_id_hex':m['token_id_hex'],'opensea_item_url':m['opensea_item_url'],'verification_method':'ERC1155.balanceOfBatch at frozen block (preserved from independently audited base run)'})
    held_ids={r['token_id_decimal'] for r in kept}; target=set(by_id)-held_ids
    print('Preserved verified rows:',len(kept),'Focused target IDs:',len(target),sorted((by_id[t]['internal_nonce'],t) for t in target),flush=True)
    cand,evidence,graph_errors=discover(target,block_num)
    write_json(OUT/'checkpoint/focused_candidates.json',{t:sorted(v) for t,v in cand.items()});write_csv(OUT/'checkpoint/focused_transfer_evidence.csv',evidence,['token_id_decimal','block_number','transaction_hash','log_index','from_address','to_address','token_value','queried_address']);write_json(OUT/'checkpoint/graph_errors.json',graph_errors)
    queried,rpc_attempts=query(cand,block_hex)
    pos=defaultdict(list)
    for r in queried:
        if int(r['balance'])>0:pos[r['token_id_decimal']].append(r)
    new=[]; audit=[]
    for tid in sorted(target,key=lambda x:by_id[x]['canonical_number']):
        m=by_id[tid]; rows=pos.get(tid,[]); total=sum(int(x['balance']) for x in rows)
        audit.append({'canonical_number':m['canonical_number'],'internal_nonce':m['internal_nonce'],'token_id_decimal':tid,'candidate_addresses':len(cand[tid]),'positive_holder_rows':len(rows),'verified_balance_sum':total,'status':'PASS' if total==1 else 'FAIL'})
        for x in rows:
            new.append({'chain_id':1,'block_number':block_num,'block_hash':block_hash,'block_timestamp_utc':block['block_timestamp_utc'],'block_timestamp_montreal':block['block_timestamp_montreal'],'contract':CONTRACT,'canonical_number':m['canonical_number'],'title':m['title'],'token_id_decimal':tid,'token_id_hex':m['token_id_hex'],'holder_address':x['holder_address'],'balance':x['balance'],'verification_method':f"ERC1155.balanceOfBatch at frozen block via {x['rpc_endpoint']}",'opensea_item_url':m['opensea_item_url']})
    holdings=kept+new; holdings.sort(key=lambda r:(int(r['canonical_number']),r['holder_address']))
    final_by=defaultdict(list)
    for r in holdings:final_by[r['token_id_decimal']].append(r)
    full_audit=[]
    for m in mani:
        rs=final_by.get(m['token_id_decimal'],[]);total=sum(int(x['balance']) for x in rs)
        full_audit.append({'canonical_number':m['canonical_number'],'internal_nonce':m['internal_nonce'],'token_id_decimal':m['token_id_decimal'],'expected_supply':1,'positive_holder_rows':len(rs),'verified_balance_sum':total,'status':'PASS' if total==1 else 'FAIL'})
    wallets=defaultdict(lambda:{'tokens':[],'units':0})
    for r in holdings:
        w=wallets[r['holder_address']];w['tokens'].append(int(r['canonical_number']));w['units']+=int(r['balance'])
    wallet_rows=[]
    for a,v in wallets.items():wallet_rows.append({'holder_address':a,'holder_category':'Burn address' if a==DEAD else ('MONTREAL.AI / Creator' if a==CREATOR else 'External holder'),'distinct_genesis_tokens':len(v['tokens']),'total_genesis_units':v['units'],'share_of_556':v['units']/556,'canonical_numbers':' '.join(map(str,sorted(v['tokens'])))})
    wallet_rows.sort(key=lambda r:(-r['total_genesis_units'],r['holder_address']))
    complete=len(holdings)==556 and sum(int(r['balance']) for r in holdings)==556 and all(r['status']=='PASS' for r in full_audit) and len(target)==8 and all(r['status']=='PASS' for r in audit)
    write_json(OUT/'manifest/manifest.json',{'schema':'montrealai.becoming.genesis-manifest.v5','records':mani});write_csv(OUT/'manifest/manifest.csv',mani,list(mani[0]));(OUT/'manifest/token_ids.txt').write_text('\n'.join(r['token_id_decimal'] for r in mani)+'\n')
    hf=['chain_id','block_number','block_hash','block_timestamp_utc','block_timestamp_montreal','contract','canonical_number','title','token_id_decimal','token_id_hex','holder_address','balance','verification_method','opensea_item_url']
    write_csv(OUT/'snapshot/token_holdings.csv',holdings,hf);write_csv(OUT/'snapshot/wallet_summary.csv',wallet_rows,['holder_address','holder_category','distinct_genesis_tokens','total_genesis_units','share_of_556','canonical_numbers']);write_csv(OUT/'audit/token_supply_audit.csv',full_audit,list(full_audit[0]));write_csv(OUT/'audit/focused_target_audit.csv',audit,list(audit[0]));write_csv(OUT/'audit/focused_queried_balances.csv',queried,['token_id_decimal','holder_address','balance','rpc_endpoint'])
    write_json(OUT/'snapshot/snapshot_block.json',{**block,'source_base_run_status':base_status,'independent_rpc_crosschecks':cross})
    status={'schema':'montrealai.becoming.atomic-snapshot.v5','capture_started_utc':started,'capture_finished_utc':now(),'chain_id':1,'block_tag':'frozen finalized block from base run','block_number':block_num,'block_hash':block_hash,'block_timestamp_utc':block['block_timestamp_utc'],'block_timestamp_montreal':block['block_timestamp_montreal'],'contract':CONTRACT,'manifest_records':len(mani),'preserved_verified_rows':len(kept),'focused_target_ids':len(target),'focused_target_passed':sum(r['status']=='PASS' for r in audit),'positive_holder_rows':len(holdings),'wallets':len(wallet_rows),'tokens_passed':sum(r['status']=='PASS' for r in full_audit),'tokens_failed':sum(r['status']!='PASS' for r in full_audit),'verified_units':sum(int(r['balance']) for r in holdings),'expected_units':556,'atomic_snapshot_complete':complete,'block_crosscheck_passes':sum(r['status']=='PASS' for r in cross),'graph_errors':graph_errors,'rpc_attempts':rpc_attempts,'unresolved_tokens':[r for r in full_audit if r['status']!='PASS']}
    write_json(OUT/'snapshot/snapshot_status.json',status)
    readme=f'''# MONTREAL.AI — BECOMING: GENESIS 556\n\nComplete fixed-block ownership snapshot.\n\n- Ethereum block: `{block_num}`\n- Block hash: `{block_hash}`\n- UTC: `{block['block_timestamp_utc']}`\n- Montréal: `{block['block_timestamp_montreal']}`\n- Token IDs: `{len(mani)}`\n- Verified units: `{status['verified_units']}`\n- Holder wallets: `{status['wallets']}`\n- Complete: `{str(complete).lower()}`\n\nThe base run had already verified 548 token balances at this exact block. This focused resolver corrected the legacy sequence gap (nonce 159 is excluded; nonce 161 is canonical), traced only the eight missing IDs through public ERC-1155 transfer history, and verified their balances by direct `balanceOfBatch` calls at the same block.\n'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in {'SHA256SUMS','MONTREALAI_BECOMING_GENESIS_556_COMPLETE_SNAPSHOT.zip'})
    (OUT/'SHA256SUMS').write_text('\n'.join(f'{sha(p)}  {p.relative_to(OUT).as_posix()}' for p in files)+'\n')
    z=OUT/'MONTREALAI_BECOMING_GENESIS_556_COMPLETE_SNAPSHOT.zip'
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as a:
        for p in sorted(x for x in OUT.rglob('*') if x.is_file() and x!=z):a.write(p,p.relative_to(OUT))
    (OUT/(z.name+'.sha256')).write_text(f'{sha(z)}  {z.name}\n')
    print(json.dumps(status,indent=2),flush=True)
    return 0 if complete else 2
if __name__=='__main__':raise SystemExit(main())
