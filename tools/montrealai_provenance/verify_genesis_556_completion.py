#!/usr/bin/env python3
import csv, hashlib, json, os, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests
from eth_abi import encode as abi_encode, decode as abi_decode
from eth_utils import keccak

CONTRACT='0x495f947276749ce646f68ac8c248420045cb7b5e'.lower()
CREATOR='0x054a2e4b3b5ea2c62372e92358fdf7fb74b4f34a'.lower()
ZERO='0x0000000000000000000000000000000000000000'
BASELINE_BLOCK=25768301
BASELINE_HASH='0xa5c03cfebd865fc8c7214954c1fa6fa1ab91badf0a227784d381c081a1500b23'
TARGET_ARTWORKS={'495','497','499','517','531','532'}
RPC_URLS=[x.strip() for x in os.getenv('ETH_RPC_URLS','https://rpc.mevblocker.io,https://ethereum.publicnode.com,https://1rpc.io/eth,https://eth.llamarpc.com,https://eth.drpc.org').split(',') if x.strip()]
OUT=Path(os.getenv('OUTPUT_DIR','genesis_556_completion_verification'))
BASELINE=Path(os.getenv('BASELINE_CSV','baseline/crosscheck_rows.csv'))
TIMEOUT=int(os.getenv('HTTP_TIMEOUT','45'))
LOG_CHUNK=int(os.getenv('LOG_CHUNK','250'))
BALANCE_CHUNK=int(os.getenv('BALANCE_CHUNK','25'))
SESSION=requests.Session(); SESSION.headers.update({'User-Agent':'MONTREAL.AI-Genesis-556-Completion-Verifier/1.0'})
TRANSFER_SINGLE='0x'+keccak(text='TransferSingle(address,address,address,uint256,uint256)').hex()
TRANSFER_BATCH='0x'+keccak(text='TransferBatch(address,address,address,uint256[],uint256[])').hex()


def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write_json(path:Path,obj:Any): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def write_csv(path:Path,rows:list[dict[str,Any]],fields:list[str]|None=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or (list(rows[0]) if rows else [])
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def rpc(url:str,method:str,params:list[Any],retries:int=3)->Any:
    last=None
    for attempt in range(retries+1):
        try:
            r=SESSION.post(url,json={'jsonrpc':'2.0','id':1,'method':method,'params':params},timeout=TIMEOUT)
            if r.status_code in (403,408,425,429) or r.status_code>=500: raise RuntimeError(f'HTTP {r.status_code}: {r.text[:250]}')
            r.raise_for_status(); data=r.json()
            if data.get('error'): raise RuntimeError(json.dumps(data['error']))
            return data.get('result')
        except Exception as e:
            last=e
            if attempt<retries: time.sleep(min(8,0.5*(2**attempt)))
    raise RuntimeError(f'{url} {method} failed: {last}')

def rpc_any(method:str,params:list[Any],preferred:str|None=None,retries:int=2):
    urls=([preferred] if preferred else [])+[u for u in RPC_URLS if u!=preferred]
    attempts=[]
    for url in urls:
        if not url: continue
        try:
            val=rpc(url,method,params,retries); attempts.append({'url':url,'status':'PASS'}); return val,url,attempts
        except Exception as e: attempts.append({'url':url,'status':'FAIL','error':str(e)})
    raise RuntimeError(f'All RPCs failed for {method}: {attempts}')

def addr(topic:str)->str: return '0x'+topic[-40:].lower()
def parse_logs(raw:list[dict[str,Any]],token_ids:set[str])->list[dict[str,Any]]:
    rows=[]
    for log in raw:
        topics=[str(x).lower() for x in log.get('topics') or []]
        if len(topics)<4: continue
        t0=topics[0]
        block=int(log['blockNumber'],16); txi=int(log['transactionIndex'],16); li=int(log['logIndex'],16)
        base={'transaction_hash':str(log['transactionHash']).lower(),'block_number':block,'transaction_index':txi,'log_index':li,'operator':addr(topics[1]),'from_address':addr(topics[2]),'to_address':addr(topics[3])}
        data=bytes.fromhex(str(log.get('data') or '0x')[2:])
        if t0==TRANSFER_SINGLE.lower():
            tid,val=abi_decode(['uint256','uint256'],data); tid=str(tid)
            if tid in token_ids: rows.append({**base,'event_signature':'TransferSingle','batch_position':0,'token_id_decimal':tid,'quantity':int(val)})
        elif t0==TRANSFER_BATCH.lower():
            ids,vals=abi_decode(['uint256[]','uint256[]'],data)
            for pos,(tid,val) in enumerate(zip(ids,vals)):
                tid=str(tid)
                if tid in token_ids: rows.append({**base,'event_signature':'TransferBatch','batch_position':pos,'token_id_decimal':tid,'quantity':int(val)})
    rows.sort(key=lambda r:(r['block_number'],r['transaction_index'],r['log_index'],r['batch_position']))
    return rows

def encode_balance_batch(accounts:list[str],ids:list[str])->str:
    selector=keccak(text='balanceOfBatch(address[],uint256[])')[:4]
    return '0x'+(selector+abi_encode(['address[]','uint256[]'],[accounts,[int(x) for x in ids]])).hex()
def query_pairs(url:str,block_hex:str,pairs:list[tuple[str,str]],label:str):
    out=[]; preferred=url
    for start in range(0,len(pairs),BALANCE_CHUNK):
        chunk=pairs[start:start+BALANCE_CHUNK]
        call={'to':CONTRACT,'data':encode_balance_batch([a for _,a in chunk],[t for t,_ in chunk])}
        raw,used,attempts=rpc_any('eth_call',[call,block_hex],preferred=preferred,retries=2); preferred=used
        vals=list(abi_decode(['uint256[]'],bytes.fromhex(raw[2:]))[0])
        if len(vals)!=len(chunk): raise RuntimeError('balanceOfBatch length mismatch')
        out.extend({'token_id_decimal':t,'address':a,'balance':int(v),'rpc':used,'attempts':json.dumps(attempts,separators=(',',':'))} for (t,a),v in zip(chunk,vals))
        print(f'{label}: {min(start+BALANCE_CHUNK,len(pairs))}/{len(pairs)} via {used}',flush=True)
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True); started=now()
    with BASELINE.open(encoding='utf-8-sig',newline='') as f: raw_baseline=list(csv.DictReader(f))
    # Accept either the compact baseline schema or the preserved two-endpoint crosscheck schema.
    if raw_baseline and 'endpoint' in raw_baseline[0]:
        unique={}
        for x in raw_baseline:
            n=str(x['artwork_number']).zfill(3)
            if n in unique: continue
            state=x.get('state') or ''
            address=(x.get('address') or '').lower()
            tid=x['token_id_decimal']
            unique[n]={
                'artwork_number':n,'token_id_decimal':tid,'token_id_hex':f'0x{int(tid):064x}',
                'ownership_state':state,
                'onchain_holder_address':address if state=='ONCHAIN_HELD' and int(x.get('expected_balance') or 0)==1 else '',
                'controller_address':address if state!='ONCHAIN_HELD' else CREATOR,
            }
        baseline=list(unique.values())
    else:
        baseline=raw_baseline
    if len(baseline)!=556: raise RuntimeError(f'Expected 556 baseline rows, got {len(baseline)}')
    baseline.sort(key=lambda r:int(r['artwork_number']))
    token_ids={r['token_id_decimal'] for r in baseline}; by_tid={r['token_id_decimal']:r for r in baseline}
    if len(token_ids)!=556 or {r['artwork_number'] for r in baseline}!={f'{i:03d}' for i in range(556)}: raise RuntimeError('Baseline identity validation failed')
    # Confirm baseline boundary.
    baseline_checks=[]
    for url in RPC_URLS:
        try:
            b=rpc(url,'eth_getBlockByNumber',[hex(BASELINE_BLOCK),False],2); actual=str((b or {}).get('hash') or '').lower()
            baseline_checks.append({'url':url,'status':'PASS' if actual==BASELINE_HASH else 'MISMATCH','actual_hash':actual})
        except Exception as e: baseline_checks.append({'url':url,'status':'ERROR','error':str(e)})
    if not any(x['status']=='PASS' for x in baseline_checks): raise RuntimeError('Baseline block could not be independently confirmed')
    # Choose finalized block and two independent endpoints with the same hash.
    block_candidates=[]
    for url in RPC_URLS:
        try:
            chain=int(rpc(url,'eth_chainId',[],2),16)
            b=rpc(url,'eth_getBlockByNumber',['finalized',False],2)
            block_candidates.append({'url':url,'chain_id':chain,'number':int(b['number'],16),'hash':str(b['hash']).lower(),'timestamp':int(b['timestamp'],16),'status':'PASS'})
        except Exception as e: block_candidates.append({'url':url,'status':'ERROR','error':str(e)})
    passes=[x for x in block_candidates if x.get('status')=='PASS' and x.get('chain_id')==1 and x['number']>BASELINE_BLOCK]
    if not passes: raise RuntimeError(f'No usable finalized block: {block_candidates}')
    # Find a block candidate reproduced by at least two endpoints.
    selected=None; endpoints=[]
    for cand in sorted(passes,key=lambda x:x['number'],reverse=True):
        same=[]
        for url in RPC_URLS:
            try:
                b=rpc(url,'eth_getBlockByNumber',[hex(cand['number']),False],2)
                if str((b or {}).get('hash') or '').lower()==cand['hash']: same.append(url)
            except Exception: pass
        if len(same)>=2: selected=cand; endpoints=same[:2]; break
    if not selected: raise RuntimeError(f'No finalized block reproduced by two endpoints: {block_candidates}')
    block_num=selected['number']; block_hex=hex(block_num); block_hash=selected['hash']; block_time=datetime.fromtimestamp(selected['timestamp'],timezone.utc).isoformat().replace('+00:00','Z')
    print(f'Selected block {block_num} {block_hash} endpoints={endpoints}',flush=True)
    # Fetch all shared-storefront ERC-1155 logs after the previous definitive block.
    raw_logs=[]; log_attempts=[]; preferred=endpoints[0]
    for start in range(BASELINE_BLOCK+1,block_num+1,LOG_CHUNK):
        end=min(start+LOG_CHUNK-1,block_num)
        params=[{'address':CONTRACT,'fromBlock':hex(start),'toBlock':hex(end),'topics':[[TRANSFER_SINGLE,TRANSFER_BATCH]]}]
        val,used,attempts=rpc_any('eth_getLogs',params,preferred=preferred,retries=3); preferred=used
        raw_logs.extend(val or []); log_attempts.append({'from_block':start,'to_block':end,'rpc':used,'raw_logs':len(val or []),'attempts':attempts})
        print(f'Logs {start}-{end}: {len(val or [])} via {used}',flush=True)
    collection_logs=parse_logs(raw_logs,token_ids)
    write_json(OUT/'raw_logs.json',raw_logs); write_csv(OUT/'POST_BASELINE_COLLECTION_TRANSFER_LOGS.csv',collection_logs)
    write_json(OUT/'rpc_log_attempts.json',log_attempts)
    logs_by_tid=defaultdict(list)
    for x in collection_logs: logs_by_tid[x['token_id_decimal']].append(x)
    # Discover candidate addresses and expected current recipient from the complete post-baseline log sequence.
    candidates=defaultdict(set); expected={}; changed={}
    for r in baseline:
        tid=r['token_id_decimal']; holder=(r.get('onchain_holder_address') or '').lower(); controller=(r.get('controller_address') or '').lower()
        if holder: candidates[tid].add(holder); expected[tid]=holder
        if controller: candidates[tid].add(controller)
        evs=logs_by_tid.get(tid,[])
        for e in evs:
            if e['from_address']!=ZERO: candidates[tid].add(e['from_address'])
            if e['to_address']!=ZERO: candidates[tid].add(e['to_address'])
        if evs:
            last=evs[-1]; expected[tid]=last['to_address'] if last['to_address']!=ZERO else ''
            changed[tid]=True
        else: changed[tid]=False
    pairs=[(tid,a) for tid in sorted(token_ids,key=int) for a in sorted(candidates[tid])]
    primary=query_pairs(endpoints[0],block_hex,pairs,'Primary balances')
    secondary=query_pairs(endpoints[1],block_hex,pairs,'Secondary balances')
    write_csv(OUT/'CANDIDATE_BALANCES_PRIMARY.csv',primary); write_csv(OUT/'CANDIDATE_BALANCES_SECONDARY.csv',secondary)
    pmap={(x['token_id_decimal'],x['address']):x['balance'] for x in primary}; smap={(x['token_id_decimal'],x['address']):x['balance'] for x in secondary}
    completion_rows=[]; all_rows=[]; failures=[]; unique_holders=set()
    block_cache={}
    def block_info(n:int):
        if n not in block_cache:
            b=rpc(endpoints[0],'eth_getBlockByNumber',[hex(n),False],3)
            # Crosscheck exact block hash on second endpoint.
            b2=rpc(endpoints[1],'eth_getBlockByNumber',[hex(n),False],3)
            if str(b['hash']).lower()!=str(b2['hash']).lower(): raise RuntimeError(f'Block hash mismatch at {n}')
            block_cache[n]={'block_number':n,'block_hash':str(b['hash']).lower(),'timestamp_unix':int(b['timestamp'],16),'timestamp_utc':datetime.fromtimestamp(int(b['timestamp'],16),timezone.utc).isoformat().replace('+00:00','Z')}
        return block_cache[n]
    for r in baseline:
        tid=r['token_id_decimal']; evs=logs_by_tid.get(tid,[]); positives=[]
        for a in sorted(candidates[tid]):
            pv=pmap.get((tid,a),0); sv=smap.get((tid,a),0)
            if pv!=sv: failures.append({'artwork_number':r['artwork_number'],'token_id_decimal':tid,'failure':'RPC_BALANCE_MISMATCH','address':a,'primary':pv,'secondary':sv})
            if pv>0: positives.append((a,pv))
        total=sum(v for _,v in positives); holder=positives[0][0] if len(positives)==1 and positives[0][1]==1 else ''
        state='ONCHAIN_HELD' if total==1 and len(positives)==1 else ('ZERO_BALANCE' if total==0 else 'INVALID_SUPPLY')
        if state!='ONCHAIN_HELD': failures.append({'artwork_number':r['artwork_number'],'token_id_decimal':tid,'failure':state,'positive_balances':positives})
        if holder: unique_holders.add(holder)
        first=evs[0] if evs else None; last=evs[-1] if evs else None
        first_info=block_info(first['block_number']) if first else None; last_info=block_info(last['block_number']) if last else None
        row={
            'artwork_number':r['artwork_number'],'token_id_decimal':tid,'token_id_hex':r['token_id_hex'],'original_identity_preserved':True,
            'baseline_state':r['ownership_state'],'baseline_holder_address':r.get('onchain_holder_address') or '',
            'current_state':state,'current_holder_address':holder,'current_balance':total,
            'changed_after_baseline':bool(evs),'post_baseline_transfer_count':len(evs),
            'first_post_baseline_transaction':first['transaction_hash'] if first else '',
            'first_post_baseline_block':first['block_number'] if first else '',
            'first_post_baseline_block_hash':first_info['block_hash'] if first_info else '',
            'first_post_baseline_timestamp_utc':first_info['timestamp_utc'] if first_info else '',
            'first_post_baseline_from':first['from_address'] if first else '',
            'first_post_baseline_to':first['to_address'] if first else '',
            'last_post_baseline_transaction':last['transaction_hash'] if last else '',
            'last_post_baseline_block':last['block_number'] if last else '',
            'last_post_baseline_timestamp_utc':last_info['timestamp_utc'] if last_info else '',
            'verification_block_number':block_num,'verification_block_hash':block_hash,'verification_timestamp_utc':block_time,
            'primary_rpc':endpoints[0],'secondary_rpc':endpoints[1],'verification_status':'PASS' if state=='ONCHAIN_HELD' else 'FAIL'
        }
        all_rows.append(row)
        if r['artwork_number'] in TARGET_ARTWORKS: completion_rows.append(row)
    completion_rows.sort(key=lambda x:int(x['artwork_number'])); all_rows.sort(key=lambda x:int(x['artwork_number']))
    write_csv(OUT/'GENESIS_556_CURRENT_STATE.csv',all_rows); write_csv(OUT/'LEGACY_SIX_ONCHAIN_TRANSITIONS.csv',completion_rows)
    write_json(OUT/'block_cache.json',{str(k):v for k,v in block_cache.items()})
    target_pass=len(completion_rows)==6 and all(x['current_state']=='ONCHAIN_HELD' and x['current_balance']==1 and x['changed_after_baseline'] for x in completion_rows)
    all_pass=len(all_rows)==556 and all(x['current_state']=='ONCHAIN_HELD' and x['current_balance']==1 for x in all_rows) and not failures
    completion_times=[x['first_post_baseline_timestamp_utc'] for x in completion_rows if x['first_post_baseline_timestamp_utc']]
    completion_time=max(completion_times) if len(completion_times)==6 else None
    completion_date=completion_time[:10] if completion_time else None
    status={
        'schema':'montrealai.genesis-556.onchain-completion-verification.v1','capture_started_utc':started,'capture_finished_utc':now(),
        'baseline_block_number':BASELINE_BLOCK,'baseline_block_hash':BASELINE_HASH,'baseline_block_crosschecks':baseline_checks,
        'verification_block_number':block_num,'verification_block_hash':block_hash,'verification_timestamp_utc':block_time,'verification_endpoints':endpoints,
        'canonical_records':556,'target_legacy_records':sorted(TARGET_ARTWORKS),'post_baseline_collection_transfer_logs':len(collection_logs),
        'target_records_positive_onchain':sum(x['current_state']=='ONCHAIN_HELD' for x in completion_rows),'total_positive_onchain_assets':sum(x['current_state']=='ONCHAIN_HELD' for x in all_rows),
        'total_onchain_units':sum(x['current_balance'] for x in all_rows),'latent_records':sum(x['current_state']!='ONCHAIN_HELD' for x in all_rows),
        'unique_current_holder_addresses':len(unique_holders),'completion_timestamp_utc':completion_time,'completion_date_utc':completion_date,
        'original_contract_preserved':True,'original_token_ids_preserved':True,'edition_supply_expected':1,
        'claim_verified':bool(target_pass and all_pass and completion_date=='2026-08-16'),'target_acceptance_pass':target_pass,'corpus_acceptance_pass':all_pass,
        'failures':failures,'statement':('VERIFIED' if target_pass and all_pass and completion_date=='2026-08-16' else 'NOT VERIFIED')
    }
    write_json(OUT/'VERIFICATION_STATUS.json',status)
    # Human-readable certificate.
    lines=['# MONTREAL.AI Genesis 556 - Onchain Completion Verification','',f"Status: **{status['statement']}**",'',f"Verification block: `{block_num}`",f"Block hash: `{block_hash}`",f"UTC: `{block_time}`",'',f"Positive onchain assets: **{status['total_positive_onchain_assets']} / 556**",f"Latent records: **{status['latent_records']}**",'', '## Six legacy transitions']
    for x in completion_rows: lines += ['',f"### Crypto AI Art #{x['artwork_number']}",f"- Original token ID: `{x['token_id_decimal']}`",f"- First confirmed onchain transition: `{x['first_post_baseline_timestamp_utc']}`",f"- Transaction: `{x['first_post_baseline_transaction']}`",f"- Current holder: `{x['current_holder_address']}`",f"- Balance at verification block: `{x['current_balance']}`"]
    (OUT/'VERIFICATION_CERTIFICATE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Checksums.
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (OUT/'SHA256SUMS').write_text('\n'.join(f'{sha256(p)}  {p.relative_to(OUT).as_posix()}' for p in files)+'\n',encoding='utf-8')
    print(json.dumps(status,indent=2),flush=True)
    return 0 if status['claim_verified'] else 2

if __name__=='__main__': raise SystemExit(main())
