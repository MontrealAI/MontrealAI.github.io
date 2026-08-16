#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, os, re, sys, time, zipfile
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import requests

CONTRACT="0x495f947276749ce646f68ac8c248420045cb7b5e"
CREATOR="0x054a2e4b3b5ea2c62372e92358fdf7fb74b4f34a"
ZERO="0x0000000000000000000000000000000000000000"
BASE="https://eth.blockscout.com/api/v2"
LEGACY="https://eth.blockscout.com/api"
OUT=Path(os.environ.get("SNAPSHOT_OUT","snapshot_output"))
RPCS=[x.strip() for x in os.environ.get("ETH_RPC_URLS","https://ethereum-rpc.publicnode.com,https://eth.llamarpc.com,https://rpc.flashbots.net,https://eth.drpc.org").split(",") if x.strip()]
WORKERS=int(os.environ.get("SNAPSHOT_WORKERS","4")); RETRIES=int(os.environ.get("HTTP_RETRIES","8")); TIMEOUT=int(os.environ.get("HTTP_TIMEOUT","40")); BATCH=int(os.environ.get("RPC_BATCH_SIZE","80"))
S=requests.Session(); S.headers.update({"Accept":"application/json","User-Agent":"MONTREALAI-Becoming-Genesis-556-Snapshot/3.0"})

def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
def nap(i): time.sleep(min(20, .6*(2**i)))
def norm(v):
    if isinstance(v,dict):
        for k in ("hash","address_hash","address"):
            z=norm(v.get(k))
            if z: return z
        return None
    if not isinstance(v,str): return None
    v=v.lower().strip()
    return v if re.fullmatch(r"0x[0-9a-f]{40}",v) and v!=ZERO else None

def get(url,params=None,allow404=False):
    err=None
    for i in range(RETRIES+1):
        try:
            r=S.get(url,params=params,timeout=TIMEOUT)
            if allow404 and r.status_code==404: return None
            if r.status_code==429 or r.status_code>=500: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:180]}")
            r.raise_for_status(); x=r.json()
            if not isinstance(x,dict): raise RuntimeError("non-object JSON")
            return x
        except Exception as e:
            err=e
            if i==RETRIES: break
            nap(i)
    raise RuntimeError(f"GET {url} failed: {err}")

def post_rpc(url,method,params,retries=5):
    err=None
    for i in range(retries+1):
        try:
            r=S.post(url,json={"jsonrpc":"2.0","id":1,"method":method,"params":params},timeout=TIMEOUT)
            if r.status_code==429 or r.status_code>=500: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:180]}")
            r.raise_for_status(); x=r.json()
            if "error" in x: raise RuntimeError(json.dumps(x["error"]))
            return x["result"]
        except Exception as e:
            err=e
            if i==retries: break
            nap(i)
    raise RuntimeError(f"RPC {method} at {url} failed: {err}")

def pages(url,params=None):
    p=dict(params or {}); seen=set()
    for _ in range(300):
        key=json.dumps(p,sort_keys=True)
        if key in seen: raise RuntimeError("pagination loop")
        seen.add(key); x=get(url,p)
        for item in x.get("items") or []:
            if isinstance(item,dict): yield item
        p=x.get("next_page_params")
        if not isinstance(p,dict) or not p: return
        p={k:v for k,v in p.items() if v is not None}
    raise RuntimeError("pagination limit exceeded")

def manifest():
    nonces=[n for n in range(5,564) if n not in {161,495,523}]
    assert len(nonces)==556
    nonces[257],nonces[258]=nonces[258],nonces[257]
    c=int(CREATOR,16); rows=[]
    for no,n in enumerate(nonces,1):
        tid=(c<<96)|(n<<40)|1
        rows.append({"canonical_number":no,"title":f"Crypto AI Art #{no:03d}","chain_id":1,"contract":CONTRACT,"standard":"ERC-1155","token_id_decimal":str(tid),"token_id_hex":f"0x{tid:064x}","creator_encoded":CREATOR,"internal_nonce":n,"edition_supply":1,"opensea_item_url":f"https://opensea.io/item/ethereum/{CONTRACT}/{tid}"})
    assert len({r["token_id_decimal"] for r in rows})==556
    return rows

def add_addr(item,key,out):
    a=norm(item.get(key))
    if a: out.add(a)

def discover(row):
    tid=row["token_id_decimal"]; root=f"{BASE}/tokens/{CONTRACT}/instances/{tid}"; cand={CREATOR}; errors=[]; meta=""; instance=False; holders=0; transfers=0
    try:
        x=get(root,allow404=True)
        if x:
            instance=True; md=x.get("metadata") if isinstance(x.get("metadata"),dict) else {}; meta=str(md.get("name") or "")
            for k in ("holder_address_hash","owner","address"): add_addr(x,k,cand)
    except Exception as e: errors.append("instance: "+str(e))
    try:
        for x in pages(root+"/holders"):
            before=len(cand)
            for k in ("address_hash","address","holder_address_hash","holder"): add_addr(x,k,cand)
            if len(cand)>before: holders+=1
    except Exception as e: errors.append("holders: "+str(e))
    try:
        for x in pages(root+"/transfers"):
            transfers+=1
            for k in ("from","to","from_address_hash","to_address_hash"): add_addr(x,k,cand)
    except Exception as e: errors.append("transfers: "+str(e))
    return {"canonical_number":row["canonical_number"],"token_id_decimal":tid,"instance_found":instance,"metadata_name":meta,"candidate_count":len(cand),"holder_addresses_found":holders,"transfer_records_found":transfers,"errors":" | ".join(errors),"candidates":sorted(cand)}

def choose_rpc():
    attempts=[]
    for u in RPCS:
        try:
            cid=int(post_rpc(u,"eth_chainId",[],2),16)
            if cid!=1: raise RuntimeError(f"chain {cid}")
            b=post_rpc(u,"eth_getBlockByNumber",["finalized",False],2)
            if not isinstance(b,dict) or not b.get("number") or not b.get("hash"): raise RuntimeError("missing finalized block")
            if post_rpc(u,"eth_getCode",[CONTRACT,b["number"]],2)=="0x": raise RuntimeError("contract code missing")
            attempts.append({"url":u,"status":"PASS","block_number":int(b["number"],16),"block_hash":b["hash"]}); return u,b,attempts
        except Exception as e: attempts.append({"url":u,"status":"FAIL","error":str(e)})
    raise RuntimeError(json.dumps(attempts))

def crosscheck(primary,b):
    out=[]
    for u in RPCS:
        if u==primary: continue
        try:
            x=post_rpc(u,"eth_getBlockByNumber",[b["number"],False],2); h=str((x or {}).get("hash","")).lower(); want=b["hash"].lower()
            out.append({"url":u,"status":"PASS" if h==want else "MISMATCH","actual_hash":h,"expected_hash":want})
        except Exception as e: out.append({"url":u,"status":"ERROR","error":str(e)})
    return out

def abi_batch(accounts,ids):
    n=len(accounts); aa=n.to_bytes(32,"big")+b"".join(bytes.fromhex(a[2:]).rjust(32,b"\0") for a in accounts); ii=n.to_bytes(32,"big")+b"".join(i.to_bytes(32,"big") for i in ids)
    head=(64).to_bytes(32,"big")+(64+len(aa)).to_bytes(32,"big")
    return "0x"+(bytes.fromhex("4e1273f4")+head+aa+ii).hex()

def decode_array(v):
    d=bytes.fromhex(v[2:]); off=int.from_bytes(d[:32],"big"); n=int.from_bytes(d[off:off+32],"big"); start=off+32
    return [int.from_bytes(d[start+32*i:start+32*(i+1)],"big") for i in range(n)]

def balances(rpc,block,pairs):
    out=[]
    for i in range(0,len(pairs),BATCH):
        q=pairs[i:i+BATCH]; data=abi_batch([a for _,a in q],[int(t) for t,_ in q]); raw=post_rpc(rpc,"eth_call",[{"to":CONTRACT,"data":data},block],6); vals=decode_array(raw)
        if len(vals)!=len(q): raise RuntimeError("balanceOfBatch length mismatch")
        out += [{"token_id_decimal":t,"holder_address":a,"balance":v} for (t,a),v in zip(q,vals)]
        print(f"Balances {min(i+BATCH,len(pairs))}/{len(pairs)}",flush=True)
    return out

def legacy_bfs(target_ids,endblock,max_addresses=5000):
    q=deque([CREATOR]); seen=set(); by={t:{CREATOR} for t in target_ids}; errors=[]
    while q and len(seen)<max_addresses:
        address=q.popleft()
        if address in seen: continue
        seen.add(address); page=1
        while page<=200:
            try:
                x=get(LEGACY,{"module":"account","action":"token1155tx","address":address,"contractaddress":CONTRACT,"startblock":0,"endblock":endblock,"page":page,"offset":10000,"sort":"asc"})
            except Exception as e:
                errors.append(f"{address} page {page}: {e}"); break
            items=x.get("result") if isinstance(x.get("result"),list) else []
            for z in items:
                tid=str(z.get("tokenID") or z.get("tokenId") or z.get("token_id") or "")
                if tid not in by: continue
                for k in ("from","to"):
                    a=norm(z.get(k))
                    if a:
                        by[tid].add(a)
                        if a not in seen: q.append(a)
            if len(items)<10000: break
            page+=1
    return by,{"addresses_scanned":len(seen),"queue_remaining":len(q),"errors":errors,"limit_hit":bool(q)}

def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def write_json(path,x): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True); started=now(); man=manifest(); byid={r["token_id_decimal"]:r for r in man}; print("Manifest 556 built",flush=True)
    disc=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fs={ex.submit(discover,r):r for r in man}
        for n,f in enumerate(as_completed(fs),1):
            try: disc.append(f.result())
            except Exception as e:
                r=fs[f]; disc.append({"canonical_number":r["canonical_number"],"token_id_decimal":r["token_id_decimal"],"instance_found":False,"metadata_name":"","candidate_count":1,"holder_addresses_found":0,"transfer_records_found":0,"errors":str(e),"candidates":[CREATOR]})
            if n%25==0 or n==556: print(f"Discovery {n}/556",flush=True)
    disc.sort(key=lambda x:x["canonical_number"])
    rpc,block,rpc_attempts=choose_rpc(); bn=int(block["number"],16); bh=block["hash"]; bt=datetime.fromtimestamp(int(block["timestamp"],16),timezone.utc); butc=bt.isoformat().replace("+00:00","Z"); bmtl=bt.astimezone(ZoneInfo("America/Montreal")).isoformat(); checks=crosscheck(rpc,block)
    candidates={d["token_id_decimal"]:set(d["candidates"]) for d in disc}; pairs=sorted((t,a) for t,s in candidates.items() for a in s); queried=balances(rpc,block["number"],pairs)
    positive=defaultdict(list)
    for x in queried:
        if x["balance"]>0: positive[x["token_id_decimal"]].append(x)
    unresolved=[t for t in byid if sum(x["balance"] for x in positive[t])!=1]
    fallback={"used":False,"addresses_scanned":0,"errors":[]}
    if unresolved:
        print(f"Fallback needed for {len(unresolved)} tokens",flush=True); extra,fallback=legacy_bfs(set(unresolved),bn); fallback["used"]=True
        new=[]
        done={(x["token_id_decimal"],x["holder_address"]) for x in queried}
        for t,addrs in extra.items():
            for a in addrs:
                if (t,a) not in done: new.append((t,a))
        if new:
            more=balances(rpc,block["number"],sorted(new)); queried+=more
            for x in more:
                if x["balance"]>0: positive[x["token_id_decimal"]].append(x)
    holdings=[]; audit=[]
    for r in man:
        t=r["token_id_decimal"]; hs=sorted(positive[t],key=lambda x:x["holder_address"]); total=sum(x["balance"] for x in hs); ok=total==1 and len(hs)==1
        audit.append({"canonical_number":r["canonical_number"],"token_id_decimal":t,"expected_supply":1,"positive_holder_rows":len(hs),"verified_balance_sum":total,"status":"PASS" if ok else "FAIL","candidate_addresses_queried":sum(1 for x in queried if x["token_id_decimal"]==t)})
        for x in hs:
            holdings.append({"chain_id":1,"block_number":bn,"block_hash":bh,"block_timestamp_utc":butc,"block_timestamp_montreal":bmtl,"contract":CONTRACT,"canonical_number":r["canonical_number"],"title":r["title"],"token_id_decimal":t,"token_id_hex":r["token_id_hex"],"holder_address":x["holder_address"],"balance":x["balance"],"verification_method":"ERC-1155 balanceOfBatch at finalized block","opensea_item_url":r["opensea_item_url"]})
    wh=defaultdict(list)
    for h in holdings: wh[h["holder_address"]].append(h)
    wallets=[{"holder_address":a,"distinct_genesis_tokens":len(v),"total_genesis_units":sum(x["balance"] for x in v),"canonical_numbers":" ".join(str(x["canonical_number"]) for x in sorted(v,key=lambda z:z["canonical_number"]))} for a,v in wh.items()]
    wallets.sort(key=lambda x:(-x["total_genesis_units"],x["holder_address"])); holdings.sort(key=lambda x:x["canonical_number"])
    passed=sum(x["status"]=="PASS" for x in audit); units=sum(x["balance"] for x in holdings); complete=len(man)==556 and passed==556 and units==556 and len(holdings)==556
    write_json(OUT/"manifest"/"manifest.json",man); write_csv(OUT/"manifest"/"manifest.csv",man,list(man[0])); (OUT/"manifest"/"token_ids.txt").write_text("\n".join(r["token_id_decimal"] for r in man)+"\n",encoding="utf-8")
    drows=[{**d,"candidates":" ".join(d["candidates"])} for d in disc]; write_csv(OUT/"audit"/"discovery.csv",drows,["canonical_number","token_id_decimal","instance_found","metadata_name","candidate_count","holder_addresses_found","transfer_records_found","errors","candidates"])
    write_csv(OUT/"snapshot"/"token_holdings.csv",holdings,list(holdings[0]) if holdings else ["chain_id","block_number","block_hash","canonical_number","token_id_decimal","holder_address","balance"]); write_csv(OUT/"snapshot"/"wallet_summary.csv",wallets,["holder_address","distinct_genesis_tokens","total_genesis_units","canonical_numbers"]); write_csv(OUT/"audit"/"token_supply_audit.csv",audit,list(audit[0])); write_csv(OUT/"audit"/"all_candidate_balances.csv",sorted(queried,key=lambda x:(int(byid[x["token_id_decimal"]]["canonical_number"]),x["holder_address"])),["token_id_decimal","holder_address","balance"])
    blockinfo={"chain_id":1,"block_tag":"finalized","block_number":bn,"block_number_hex":block["number"],"block_hash":bh,"block_timestamp_utc":butc,"block_timestamp_montreal":bmtl,"contract":CONTRACT,"primary_rpc":rpc,"rpc_attempts":rpc_attempts,"independent_rpc_crosschecks":checks}; write_json(OUT/"snapshot"/"snapshot_block.json",blockinfo)
    status={"schema":"montrealai.becoming.atomic-snapshot.v3","capture_started_utc":started,"capture_finished_utc":now(),"manifest_records":556,"unique_token_ids":556,"block_tag":"finalized","block_number":bn,"block_hash":bh,"block_timestamp_utc":butc,"block_timestamp_montreal":bmtl,"tokens_passed":passed,"tokens_failed":556-passed,"positive_holder_rows":len(holdings),"verified_units":units,"expected_units":556,"distinct_holder_wallets":len(wallets),"fallback":fallback,"atomic_snapshot_complete":complete,"unresolved_tokens":[x for x in audit if x["status"]!="PASS"]}; write_json(OUT/"snapshot"/"snapshot_status.json",status)
    (OUT/"README.md").write_text(f"# MONTREAL.AI — BECOMING: GENESIS 556\n\nFinalized-block ownership snapshot.\n\n- Block: `{bn}`\n- Hash: `{bh}`\n- UTC: `{butc}`\n- Montréal: `{bmtl}`\n- Tokens reconciled: `{passed}/556`\n- Units reconciled: `{units}/556`\n- Complete: `{str(complete).lower()}`\n\nMarketplace/explorer data was used only for candidate discovery. Every published owner balance was read from the ERC-1155 contract at the same finalized block.\n",encoding="utf-8")
    src=OUT/"source"; src.mkdir(exist_ok=True); (src/"snapshot.py").write_text(Path(__file__).read_text(encoding="utf-8"),encoding="utf-8")
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS","MONTREALAI_BECOMING_GENESIS_556_ATOMIC_SNAPSHOT.zip","MONTREALAI_BECOMING_GENESIS_556_ATOMIC_SNAPSHOT.zip.sha256"}); (OUT/"SHA256SUMS").write_text("\n".join(f"{digest(p)}  {p.relative_to(OUT).as_posix()}" for p in files)+"\n",encoding="utf-8")
    z=OUT/"MONTREALAI_BECOMING_GENESIS_556_ATOMIC_SNAPSHOT.zip"
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as a:
        for p in sorted(x for x in OUT.rglob("*") if x.is_file() and x!=z): a.write(p,p.relative_to(OUT))
    (OUT/(z.name+".sha256")).write_text(f"{digest(z)}  {z.name}\n",encoding="utf-8")
    print(json.dumps(status,indent=2),flush=True); return 0 if complete else 2
if __name__=="__main__": raise SystemExit(main())
