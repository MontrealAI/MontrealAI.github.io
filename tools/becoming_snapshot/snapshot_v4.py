#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, os, re, sys, time, zipfile
from collections import defaultdict
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
OUT=Path(os.environ.get("SNAPSHOT_OUT","snapshot_output"))
RPCS=[x.strip() for x in os.environ.get("ETH_RPC_URLS","https://ethereum-rpc.publicnode.com,https://eth.llamarpc.com,https://rpc.flashbots.net,https://eth.drpc.org").split(",") if x.strip()]
WORKERS=int(os.environ.get("SNAPSHOT_WORKERS","6")); RETRIES=int(os.environ.get("HTTP_RETRIES","6")); TIMEOUT=int(os.environ.get("HTTP_TIMEOUT","45")); BATCH=int(os.environ.get("RPC_BATCH_SIZE","80"))
S=requests.Session(); S.headers.update({"Accept":"application/json","User-Agent":"MONTREALAI-Becoming-Genesis-556-Snapshot/4.0"})

def utc_now(): return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
def backoff(i): time.sleep(min(20,0.6*(2**i)))
def norm(v: Any):
    if isinstance(v,dict):
        for k in ("hash","address_hash","address"):
            a=norm(v.get(k))
            if a: return a
        return None
    if not isinstance(v,str): return None
    v=v.strip().lower()
    return v if re.fullmatch(r"0x[0-9a-f]{40}",v) and v!=ZERO else None

def http_get(url,params=None,allow404=False):
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
            backoff(i)
    raise RuntimeError(f"GET {url} failed: {err}")

def pages(url,params=None,allow404=False,max_pages=500):
    p=dict(params or {}); seen=set()
    for _ in range(max_pages):
        key=json.dumps(p,sort_keys=True,default=str)
        if key in seen: raise RuntimeError("pagination loop")
        seen.add(key); x=http_get(url,p,allow404=allow404)
        if x is None: return
        for item in x.get("items") or []:
            if isinstance(item,dict): yield item
        p=x.get("next_page_params")
        if not isinstance(p,dict) or not p: return
        p={k:v for k,v in p.items() if v is not None}
    raise RuntimeError("pagination limit exceeded")

def rpc(url,method,params,retries=5):
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
            backoff(i)
    raise RuntimeError(f"RPC {method} at {url} failed: {err}")

def choose_finalized_block():
    attempts=[]
    for url in RPCS:
        try:
            cid=int(rpc(url,"eth_chainId",[],2),16)
            if cid!=1: raise RuntimeError(f"wrong chain {cid}")
            b=rpc(url,"eth_getBlockByNumber",["finalized",False],2)
            if not isinstance(b,dict) or not b.get("number") or not b.get("hash") or not b.get("timestamp"): raise RuntimeError("incomplete finalized block")
            if rpc(url,"eth_getCode",[CONTRACT,b["number"]],2)=="0x": raise RuntimeError("contract code unavailable")
            attempts.append({"url":url,"status":"PASS","block_number":int(b["number"],16),"block_hash":b["hash"]})
            return url,b,attempts
        except Exception as e: attempts.append({"url":url,"status":"FAIL","error":str(e)})
    raise RuntimeError("No Ethereum RPC supplied a finalized mainnet state: "+json.dumps(attempts))

def block_crosschecks(primary,block):
    out=[]
    for url in RPCS:
        if url==primary: continue
        try:
            x=rpc(url,"eth_getBlockByNumber",[block["number"],False],2); actual=str((x or {}).get("hash","")).lower(); expected=block["hash"].lower()
            out.append({"url":url,"status":"PASS" if actual==expected else "MISMATCH","actual_hash":actual,"expected_hash":expected})
        except Exception as e: out.append({"url":url,"status":"ERROR","error":str(e)})
    return out

def manifest():
    nonces=[n for n in range(5,564) if n not in {159,495,523}]
    assert len(nonces)==556
    nonces[257],nonces[258]=nonces[258],nonces[257]
    creator=int(CREATOR,16); rows=[]
    for number,n in enumerate(nonces,1):
        tid=(creator<<96)|(n<<40)|1
        rows.append({"canonical_number":number,"title":f"Crypto AI Art #{number:03d}","chain_id":1,"contract":CONTRACT,"standard":"ERC-1155","token_id_decimal":str(tid),"token_id_hex":f"0x{tid:064x}","creator_encoded":CREATOR,"internal_nonce":n,"edition_supply":1,"opensea_item_url":f"https://opensea.io/item/ethereum/{CONTRACT}/{tid}"})
    assert len({r["token_id_decimal"] for r in rows})==556
    return rows

def abi_balance_batch(accounts,ids):
    n=len(accounts)
    aa=n.to_bytes(32,"big")+b"".join(bytes.fromhex(a[2:]).rjust(32,b"\0") for a in accounts)
    ii=n.to_bytes(32,"big")+b"".join(int(i).to_bytes(32,"big") for i in ids)
    head=(64).to_bytes(32,"big")+(64+len(aa)).to_bytes(32,"big")
    return "0x"+(bytes.fromhex("4e1273f4")+head+aa+ii).hex()

def decode_uint_array(value):
    data=bytes.fromhex(value[2:]); off=int.from_bytes(data[:32],"big"); n=int.from_bytes(data[off:off+32],"big"); start=off+32
    return [int.from_bytes(data[start+32*i:start+32*(i+1)],"big") for i in range(n)]

def query_balances(rpc_url,block_hex,pairs,label="Balances"):
    result=[]
    for i in range(0,len(pairs),BATCH):
        q=pairs[i:i+BATCH]
        raw=rpc(rpc_url,"eth_call",[{"to":CONTRACT,"data":abi_balance_batch([a for _,a in q],[t for t,_ in q])},block_hex],6)
        vals=decode_uint_array(raw)
        if len(vals)!=len(q): raise RuntimeError("balanceOfBatch result length mismatch")
        result.extend({"token_id_decimal":str(t),"holder_address":a,"balance":v} for (t,a),v in zip(q,vals))
        print(f"{label} {min(i+BATCH,len(pairs))}/{len(pairs)}",flush=True)
    return result

def transfer_token_id(item):
    total=item.get("total") if isinstance(item.get("total"),dict) else {}
    for v in (total.get("token_id"), item.get("token_id"), item.get("id")):
        if v is not None and str(v).isdigit(): return str(v)
    inst=total.get("token_instance") if isinstance(total.get("token_instance"),dict) else {}
    v=inst.get("id")
    return str(v) if v is not None and str(v).isdigit() else None

def transfer_metadata_name(item):
    total=item.get("total") if isinstance(item.get("total"),dict) else {}
    inst=total.get("token_instance") if isinstance(total.get("token_instance"),dict) else {}
    md=inst.get("metadata") if isinstance(inst.get("metadata"),dict) else {}
    return str(md.get("name") or "")

def address_transfer_page(address):
    rows=[]; errors=[]
    try:
        for item in pages(f"{BASE}/addresses/{address}/token-transfers",{"type":"ERC-1155","token":CONTRACT},allow404=True): rows.append(item)
    except Exception as e: errors.append(str(e))
    return address,rows,errors

def crawl_transfer_graph(target_ids,block_number):
    candidates={tid:{CREATOR} for tid in target_ids}; metadata={}; events={}; seen=set(); frontier=[CREATOR]; errors=[]; rounds=0
    while frontier:
        rounds+=1; batch=[a for a in frontier if a not in seen]; frontier=[]
        if not batch: break
        if len(seen)+len(batch)>5000: raise RuntimeError("address crawl safety limit exceeded")
        print(f"Transfer graph round {rounds}: {len(batch)} addresses",flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            fs=[ex.submit(address_transfer_page,a) for a in batch]
            for f in as_completed(fs):
                address,items,errs=f.result(); seen.add(address); errors.extend(f"{address}: {e}" for e in errs)
                for item in items:
                    token=(item.get("token") or {}) if isinstance(item.get("token"),dict) else {}
                    if str(token.get("address_hash") or "").lower()!=CONTRACT: continue
                    tid=transfer_token_id(item)
                    if tid not in target_ids: continue
                    bn=int(item.get("block_number") or 0)
                    if bn and bn>block_number: continue
                    a_from=norm(item.get("from")); a_to=norm(item.get("to"))
                    for a in (a_from,a_to):
                        if a:
                            candidates[tid].add(a)
                            if a not in seen: frontier.append(a)
                    name=transfer_metadata_name(item)
                    if name: metadata.setdefault(tid,set()).add(name)
                    key=(str(item.get("transaction_hash") or ""),int(item.get("log_index") or -1),tid,a_from or ZERO,a_to or ZERO)
                    events[key]={"block_number":bn,"block_hash":str(item.get("block_hash") or ""),"timestamp":str(item.get("timestamp") or ""),"transaction_hash":key[0],"log_index":key[1],"token_id_decimal":tid,"from_address":a_from or ZERO,"to_address":a_to or ZERO,"value":str(((item.get("total") or {}) if isinstance(item.get("total"),dict) else {}).get("value") or ""),"method":str(item.get("method") or ""),"metadata_name":name}
        frontier=sorted(set(frontier)-seen)
    return candidates,metadata,sorted(events.values(),key=lambda x:(x["block_number"],x["log_index"],x["token_id_decimal"])),{"rounds":rounds,"addresses_scanned":len(seen),"errors":errors}

def instance_candidates(tid):
    out=set(); errors=[]; names=set(); root=f"{BASE}/tokens/{CONTRACT}/instances/{tid}"
    try:
        x=http_get(root,allow404=True)
        if x:
            for k in ("owner","holder_address_hash","address"):
                a=norm(x.get(k))
                if a: out.add(a)
            md=x.get("metadata") if isinstance(x.get("metadata"),dict) else {}
            if md.get("name"): names.add(str(md["name"]))
    except Exception as e: errors.append("instance: "+str(e))
    for suffix in ("holders","transfers"):
        try:
            for x in pages(root+"/"+suffix,allow404=True):
                for k in ("address_hash","address","holder_address_hash","holder","from","to","from_address_hash","to_address_hash"):
                    a=norm(x.get(k))
                    if a: out.add(a)
                name=transfer_metadata_name(x)
                if name: names.add(name)
        except Exception as e: errors.append(suffix+": "+str(e))
    return tid,out,names,errors

def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def write_json(path,obj): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True); started=utc_now(); man=manifest(); byid={r["token_id_decimal"]:r for r in man}; target=set(byid)
    primary,block,rpc_attempts=choose_finalized_block(); bn=int(block["number"],16); bh=block["hash"]; bt=datetime.fromtimestamp(int(block["timestamp"],16),timezone.utc); butc=bt.isoformat().replace("+00:00","Z"); bmtl=bt.astimezone(ZoneInfo("America/Montreal")).isoformat(); block_checks=block_crosschecks(primary,block)
    print(f"Frozen finalized block {bn} {bh}",flush=True)
    creator_pairs=[(int(r["token_id_decimal"]),CREATOR) for r in man]
    creator_results=query_balances(primary,block["number"],creator_pairs,"Creator balances")
    creator_balance={x["token_id_decimal"]:x["balance"] for x in creator_results}; non_creator={tid for tid,v in creator_balance.items() if v==0}
    print(f"Creator-held: {556-len(non_creator)}; transferred: {len(non_creator)}",flush=True)
    candidates,metadata,events,crawl=crawl_transfer_graph(target,bn)
    pairs=[(int(tid),a) for tid in sorted(target,key=int) for a in sorted(candidates[tid])]
    queried=query_balances(primary,block["number"],pairs,"Graph balances")
    positive=defaultdict(dict)
    for x in queried:
        if x["balance"]>0: positive[x["token_id_decimal"]][x["holder_address"]]=x["balance"]
    unresolved=[tid for tid in target if sum(positive[tid].values())!=1]; fallback={"tokens_requested":len(unresolved),"new_candidate_addresses":0,"errors":[]}
    if unresolved:
        print(f"Instance fallback for {len(unresolved)} tokens",flush=True); extra=[]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            fs=[ex.submit(instance_candidates,tid) for tid in unresolved]
            for f in as_completed(fs):
                tid,addrs,names,errs=f.result(); metadata.setdefault(tid,set()).update(names); fallback["errors"].extend(f"{tid}: {e}" for e in errs)
                for a in addrs:
                    if a not in candidates[tid]: candidates[tid].add(a); extra.append((int(tid),a))
        fallback["new_candidate_addresses"]=len(extra)
        if extra:
            more=query_balances(primary,block["number"],sorted(set(extra)),"Fallback balances"); queried.extend(more)
            for x in more:
                if x["balance"]>0: positive[x["token_id_decimal"]][x["holder_address"]]=x["balance"]
    holdings=[]; audit=[]; metadata_audit=[]
    for r in man:
        tid=r["token_id_decimal"]; hs=sorted(positive[tid].items()); total=sum(v for _,v in hs); ok=total==1 and len(hs)==1
        audit.append({"canonical_number":r["canonical_number"],"token_id_decimal":tid,"expected_supply":1,"positive_holder_rows":len(hs),"verified_balance_sum":total,"candidate_addresses_queried":sum(1 for x in queried if x["token_id_decimal"]==tid),"status":"PASS" if ok else "FAIL"})
        observed=sorted(metadata.get(tid,set())); expected=r["title"]; matches=[n for n in observed if n.strip().lower()==expected.lower()]
        metadata_audit.append({"canonical_number":r["canonical_number"],"token_id_decimal":tid,"expected_title":expected,"observed_titles":" | ".join(observed),"metadata_observed":bool(observed),"title_match":bool(matches) if observed else None,"status":"PASS" if (not observed or matches) else "FAIL"})
        for address,balance in hs: holdings.append({"chain_id":1,"block_number":bn,"block_hash":bh,"block_timestamp_utc":butc,"block_timestamp_montreal":bmtl,"contract":CONTRACT,"canonical_number":r["canonical_number"],"title":r["title"],"token_id_decimal":tid,"token_id_hex":r["token_id_hex"],"holder_address":address,"balance":balance,"verification_method":"ERC-1155 balanceOfBatch at finalized block","opensea_item_url":r["opensea_item_url"]})
    holdings.sort(key=lambda x:x["canonical_number"]); wallets_map=defaultdict(list)
    for h in holdings: wallets_map[h["holder_address"]].append(h)
    wallets=[{"holder_address":a,"distinct_genesis_tokens":len(v),"total_genesis_units":sum(x["balance"] for x in v),"canonical_numbers":" ".join(str(x["canonical_number"]) for x in sorted(v,key=lambda z:z["canonical_number"]))} for a,v in wallets_map.items()]
    wallets.sort(key=lambda x:(-x["total_genesis_units"],x["holder_address"]))
    secondary_status={"status":"NOT_RUN","url":None,"verified_pairs":0,"mismatches":[]}; exact_pairs=[(int(h["token_id_decimal"]),h["holder_address"]) for h in holdings]
    if len(exact_pairs)==556:
        for url in RPCS:
            if url==primary: continue
            try:
                vals=query_balances(url,block["number"],exact_pairs,"Secondary cross-check"); mism=[x for x in vals if x["balance"]!=1]
                secondary_status={"status":"PASS" if not mism else "FAIL","url":url,"verified_pairs":len(vals),"mismatches":mism}
                if not mism: break
            except Exception as e: secondary_status={"status":"ERROR","url":url,"verified_pairs":0,"mismatches":[],"error":str(e)}
    passed=sum(a["status"]=="PASS" for a in audit); units=sum(h["balance"] for h in holdings); metadata_failures=[x for x in metadata_audit if x["status"]=="FAIL"]; block_crosscheck_pass=any(x["status"]=="PASS" for x in block_checks)
    complete=(len(man)==556 and passed==556 and len(holdings)==556 and units==556 and not metadata_failures and block_crosscheck_pass and secondary_status.get("status")=="PASS")
    write_json(OUT/"manifest"/"manifest.json",man); write_csv(OUT/"manifest"/"manifest.csv",man,list(man[0])); (OUT/"manifest"/"token_ids.txt").write_text("\n".join(r["token_id_decimal"] for r in man)+"\n",encoding="utf-8")
    write_csv(OUT/"snapshot"/"token_holdings.csv",holdings,list(holdings[0]) if holdings else ["chain_id","block_number","block_hash","canonical_number","token_id_decimal","holder_address","balance"]); write_csv(OUT/"snapshot"/"wallet_summary.csv",wallets,["holder_address","distinct_genesis_tokens","total_genesis_units","canonical_numbers"])
    write_csv(OUT/"audit"/"token_supply_audit.csv",audit,list(audit[0])); write_csv(OUT/"audit"/"metadata_title_audit.csv",metadata_audit,list(metadata_audit[0])); write_csv(OUT/"audit"/"transfer_events.csv",events,list(events[0]) if events else ["block_number","token_id_decimal","from_address","to_address"]); write_csv(OUT/"audit"/"all_candidate_balances.csv",sorted(queried,key=lambda x:(int(x["token_id_decimal"]),x["holder_address"])),["token_id_decimal","holder_address","balance"])
    block_info={"chain_id":1,"block_tag":"finalized","block_number":bn,"block_number_hex":block["number"],"block_hash":bh,"block_timestamp_utc":butc,"block_timestamp_montreal":bmtl,"contract":CONTRACT,"primary_rpc":primary,"rpc_attempts":rpc_attempts,"independent_block_hash_crosschecks":block_checks,"secondary_balance_crosscheck":secondary_status}; write_json(OUT/"snapshot"/"snapshot_block.json",block_info)
    status={"schema":"montrealai.becoming.atomic-snapshot.v4","capture_started_utc":started,"capture_finished_utc":utc_now(),"manifest_records":556,"unique_token_ids":556,"manifest_gap_nonces":[159,495,523],"manifest_title_nonce_swap":[258,259],"block_tag":"finalized","block_number":bn,"block_hash":bh,"block_timestamp_utc":butc,"block_timestamp_montreal":bmtl,"creator_held_units":sum(creator_balance.values()),"transferred_units":len(non_creator),"tokens_passed":passed,"tokens_failed":556-passed,"positive_holder_rows":len(holdings),"verified_units":units,"expected_units":556,"distinct_holder_wallets":len(wallets),"metadata_titles_observed":sum(bool(x["metadata_observed"]) for x in metadata_audit),"metadata_title_failures":metadata_failures,"transfer_graph":crawl,"fallback":fallback,"secondary_balance_crosscheck":secondary_status,"atomic_snapshot_complete":complete,"unresolved_tokens":[x for x in audit if x["status"]!="PASS"]}; write_json(OUT/"snapshot"/"snapshot_status.json",status)
    (OUT/"README.md").write_text(f"# MONTREAL.AI — BECOMING: GENESIS 556\n\nAtomic finalized-block ownership snapshot.\n\n- Block: `{bn}`\n- Hash: `{bh}`\n- UTC: `{butc}`\n- Montréal: `{bmtl}`\n- Tokens reconciled: `{passed}/556`\n- Units reconciled: `{units}/556`\n- Distinct holders: `{len(wallets)}`\n- Independent balance cross-check: `{secondary_status.get('status')}`\n- Complete: `{str(complete).lower()}`\n\nBlockscout's public, keyless transfer index was used only to discover candidate addresses and metadata. Every published owner was verified by the ERC-1155 contract at the same finalized block, then checked again through an independent Ethereum RPC.\n",encoding="utf-8")
    src=OUT/"source"; src.mkdir(exist_ok=True); (src/"snapshot_v4.py").write_text(Path(__file__).read_text(encoding="utf-8"),encoding="utf-8")
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS","MONTREALAI_BECOMING_GENESIS_556_COMPLETE_SNAPSHOT.zip","MONTREALAI_BECOMING_GENESIS_556_COMPLETE_SNAPSHOT.zip.sha256"}); (OUT/"SHA256SUMS").write_text("\n".join(f"{sha256(p)}  {p.relative_to(OUT).as_posix()}" for p in files)+"\n",encoding="utf-8")
    z=OUT/"MONTREALAI_BECOMING_GENESIS_556_COMPLETE_SNAPSHOT.zip"
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as arc:
        for p in sorted(x for x in OUT.rglob("*") if x.is_file() and x!=z): arc.write(p,p.relative_to(OUT))
    (OUT/(z.name+".sha256")).write_text(f"{sha256(z)}  {z.name}\n",encoding="utf-8")
    print(json.dumps(status,indent=2),flush=True); return 0 if complete else 2

if __name__=="__main__": raise SystemExit(main())
