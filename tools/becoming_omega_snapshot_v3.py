#!/usr/bin/env python3
import csv, hashlib, json, os, re, time, urllib.error, urllib.parse, urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SLUG="montrealai"; CHAIN="ethereum"; CHAIN_ID=1
CONTRACT="0x495f947276749ce646f68ac8c248420045cb7b5e"
CREATOR="0x054a2e4b3b5ea2c62372e92358fdf7fb74b4f34a"
EXPECTED=556; OUT=Path(os.getenv("SNAPSHOT_OUT","snapshot_output")); OUT.mkdir(parents=True,exist_ok=True)
UA="MONTREAL.AI-Becoming-Omega-Snapshot/3.0 (+https://montreal.ai)"
RPCS=["https://ethereum-rpc.publicnode.com","https://eth.llamarpc.com","https://rpc.flashbots.net"]

def now(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def getaddr(v):
    if isinstance(v,str) and re.fullmatch(r"0x[0-9a-fA-F]{40}",v): return v.lower()
    if isinstance(v,dict):
        for k in ("address","owner","hash","address_hash","wallet"):
            a=getaddr(v.get(k))
            if a:return a
    return None

def http(url,method="GET",data=None,headers=None,tries=7,timeout=90,allow404=False):
    body=None; h={"User-Agent":UA,"Accept":"application/json"}; h.update(headers or {})
    if data is not None: body=json.dumps(data).encode(); h["Content-Type"]="application/json"
    last=None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,data=body,headers=h,method=method),timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if allow404 and e.code==404:return None
            last=f"HTTP {e.code}: "+e.read(800).decode(errors="replace")
            if e.code not in (408,425,429,500,502,503,504): raise RuntimeError(f"{method} {url}: {last}")
            ra=e.headers.get("Retry-After")
            if ra and ra.isdigit(): time.sleep(min(90,int(ra)+1)); continue
        except Exception as e:last=repr(e)
        time.sleep(min(30,1.8**i))
    raise RuntimeError(f"{method} {url} failed: {last}")

def oskey():
    d=http("https://api.opensea.io/api/v2/auth/keys",method="POST")
    k=d.get("api_key")
    if not k: raise RuntimeError(f"OpenSea key response missing api_key: {d}")
    dump("opensea_key_metadata.json",{x:y for x,y in d.items() if x!="api_key"}); return k

def osget(path,key,params=None,allow404=False):
    u="https://api.opensea.io"+path
    if params:u+="?"+urllib.parse.urlencode(params)
    return http(u,headers={"X-API-KEY":key},allow404=allow404)

def rpc(ep,method,params,i=1):
    d=http(ep,method="POST",data={"jsonrpc":"2.0","id":i,"method":method,"params":params})
    if "error" in d: raise RuntimeError(f"{ep} {method}: {d['error']}")
    return d.get("result")

def batch(ep,calls,size=70):
    out={}
    for s in range(0,len(calls),size):
        p=calls[s:s+size]
        try:
            d=http(ep,method="POST",data=[{"jsonrpc":"2.0","id":i,"method":m,"params":a} for i,m,a in p],tries=5)
            by={int(x["id"]):x for x in d}
            for i,_,_ in p: out[i]=by.get(i,{"error":"missing"})
        except Exception:
            for i,m,a in p:
                try:out[i]={"result":rpc(ep,m,a,i)}
                except Exception as e:out[i]={"error":repr(e)}
    return out

def number(*xs):
    pats=(r"#\s*0*(\d{1,4})(?:\D|$)",r"crypto\s*ai\s*art\D*0*(\d{1,4})(?:\D|$)",r"cryptoaiart0*(\d{1,4})(?:\D|$)")
    for x in xs:
        for p in pats:
            m=re.search(p,str(x or ""),re.I)
            if m and 1<=int(m.group(1))<=1000:return int(m.group(1))

def nfts(key):
    rows={}; pages=[]; nxt=None
    while True:
        q={"limit":200}
        if nxt:q["next"]=nxt
        d=osget(f"/api/v2/collection/{SLUG}/nfts",key,q); pages.append(d)
        for x in d.get("nfts") or d.get("items") or []:
            tid=str(x.get("identifier") or x.get("token_id") or x.get("tokenId") or "")
            c=str(x.get("contract") or CONTRACT).lower(); title=x.get("name") or ""
            if tid.isdigit() and c==CONTRACT:
                rows[tid]={"canonical_number":number(title,x.get("image_url"),x.get("metadata_url")),"title":title,"token_id_decimal":tid,"image":x.get("image_url"),"metadata_url":x.get("metadata_url")}
        nn=d.get("next") or d.get("next_cursor")
        if not nn or nn==nxt:break
        nxt=str(nn); time.sleep(.15)
        if len(pages)>20:raise RuntimeError("OpenSea pagination exceeded 20 pages")
    dump("opensea_nfts_raw.json",pages); return list(rows.values())

def decode(r):
    t=int(r["token_id_decimal"]); r["token_id_hex"]="0x"+t.to_bytes(32,"big").hex()
    r["encoded_creator"]="0x"+((t>>96)&((1<<160)-1)).to_bytes(20,"big").hex()
    mid=(t>>32)&((1<<64)-1); r["nonce"]=mid>>8; r["nonce_low_byte"]=mid&255; r["encoded_supply"]=t&((1<<32)-1)

def snapshot_block():
    fin=[]; errs={}
    for ep in RPCS:
        try:
            b=rpc(ep,"eth_getBlockByNumber",["finalized",False]); fin.append({"endpoint":ep,"number":int(b["number"],16),"hash":b["hash"].lower(),"timestamp":int(b["timestamp"],16)})
        except Exception as e:errs[ep]=repr(e)
    if len(fin)<2:raise RuntimeError(f"Only {len(fin)} finalized RPC observations: {fin}; {errs}")
    n=min(x["number"] for x in fin); exact=[]
    for x in fin:
        try:
            b=rpc(x["endpoint"],"eth_getBlockByNumber",[hex(n),False]); exact.append({"endpoint":x["endpoint"],"number":n,"hash":b["hash"].lower(),"timestamp":int(b["timestamp"],16)})
        except Exception as e:errs[x["endpoint"]+":exact"]=repr(e)
    common=Counter(x["hash"] for x in exact).most_common(1)
    if not common or common[0][1]<2:raise RuntimeError(f"No two RPCs agreed: {exact}; {errs}")
    h=common[0][0]; agree=[x for x in exact if x["hash"]==h]; t=agree[0]["timestamp"]
    return {"selection_rule":"Minimum block independently reported finalized by at least two public Ethereum RPC endpoints; exact hash cross-checked.","block_number":n,"block_number_hex":hex(n),"block_hash":h,"block_timestamp_unix":t,"block_timestamp_utc":datetime.fromtimestamp(t,timezone.utc).isoformat().replace("+00:00","Z"),"finalized_observations":fin,"exact_block_observations":exact,"agreed_endpoints":[x["endpoint"] for x in agree],"errors":errs}

def owners(key,rows):
    raw={}
    for i,r in enumerate(rows,1):
        d=osget(f"/api/v2/chain/{CHAIN}/contract/{CONTRACT}/nfts/{r['token_id_decimal']}/owners",key,allow404=True) or {}
        raw[r["token_id_decimal"]]=d; cand=[]
        for o in d.get("owners") or d.get("items") or []:
            a=getaddr(o); q=o.get("quantity",o.get("balance",1)) if isinstance(o,dict) else 1
            try:q=int(q)
            except:q=1
            if a and q>0:cand.append((a,q))
        r["candidate_owner"]=cand[0][0] if cand else CREATOR
        r["owner_resolution"]="opensea_owner_endpoint" if cand else "implicit_creator_fallback"
        if i%50==0:print(f"owners {i}/{len(rows)}",flush=True)
        time.sleep(.12)
    dump("opensea_owners_raw.json",raw)

def calldata(a,tid): return "0x00fdd58e"+a[2:].zfill(64)+hex(int(tid))[2:].zfill(64)
def verify(rows,block,eps=None):
    eps=(eps or block["agreed_endpoints"][:2])[:2]; calls=[(i,"eth_call",[{"to":CONTRACT,"data":calldata(r["candidate_owner"],r["token_id_decimal"])},block["block_number_hex"]]) for i,r in enumerate(rows,1)]
    allres={ep:batch(ep,calls) for ep in eps}
    for i,r in enumerate(rows,1):
        vals={}
        for ep in eps:
            x=allres[ep].get(i,{}).get("result")
            try:vals[ep]=int(x,16)
            except:vals[ep]=None
        r["balance_verification"]=vals; good=[x for x in vals.values() if x is not None]
        r["verified_balance"]=good[0] if len(good)>=2 and len(set(good))==1 else None; r["rpc_agreement"]=len(good)>=2 and len(set(good))==1

def bshistory(tid,blocknum):
    base=f"https://eth.blockscout.com/api/v2/tokens/{CONTRACT}/instances/{tid}/transfers"; params=None; ev=[]
    for _ in range(30):
        u=base+("?"+urllib.parse.urlencode(params) if params else ""); d=http(u,allow404=True) or {}
        for x in d.get("items") or []:
            try:bn=int(x.get("block_number") or 0)
            except:continue
            if bn<=blocknum:ev.append((bn,int(x.get("log_index") or 0),getaddr(x.get("to")),x.get("transaction_hash")))
        params=d.get("next_page_params")
        if not params:break
    return sorted(ev)

def resolve(rows,block):
    bad=[r for r in rows if r.get("verified_balance")!=r["encoded_supply"] or not r.get("rpc_agreement")]
    for r in bad:
        r["candidate_owner"]=CREATOR; r["owner_resolution"]="creator_historical_crosscheck"
    if bad:verify(bad,block)
    bad=[r for r in bad if r.get("verified_balance")!=r["encoded_supply"] or not r.get("rpc_agreement")]
    for r in bad:
        try:
            ev=bshistory(r["token_id_decimal"],block["block_number"])
            if ev and ev[-1][2]:
                r["candidate_owner"]=ev[-1][2]; r["owner_resolution"]="blockscout_last_transfer_before_snapshot"; r["last_transfer_block"]=ev[-1][0]; r["last_transfer_tx"]=ev[-1][3]
        except Exception as e:r["blockscout_error"]=repr(e)
    if bad:verify(bad,block)

def csvout(name,rows,fields):
    with (OUT/name).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)

def main():
    started=now(); key=oskey(); rows=nfts(key)
    for r in rows:decode(r)
    rows=[r for r in rows if r["canonical_number"] and 1<=r["canonical_number"]<=EXPECTED]; rows.sort(key=lambda x:x["canonical_number"])
    nc=Counter(r["canonical_number"] for r in rows); tc=Counter(r["token_id_decimal"] for r in rows); missing=sorted(set(range(1,EXPECTED+1))-set(nc)); dups=sorted(n for n,c in nc.items() if c!=1)
    if len(rows)!=EXPECTED or missing or dups or any(c!=1 for c in tc.values()):
        dump("manifest_failure_diagnostics.json",{"numbered_records":len(rows),"missing_numbers":missing,"duplicate_numbers":dups,"sample":rows[:30]}); raise RuntimeError("manifest validation failed")
    if sum(r["encoded_creator"]==CREATOR for r in rows)!=EXPECTED or sum(r["encoded_supply"]==1 for r in rows)!=EXPECTED or sum(r["nonce_low_byte"]==0 for r in rows)!=EXPECTED:raise RuntimeError("token-ID encoding validation failed")
    block=snapshot_block(); dump("snapshot_block.json",block); owners(key,rows); verify(rows,block); resolve(rows,block)
    bad=[r for r in rows if r.get("verified_balance")!=1 or not r.get("rpc_agreement")]
    if bad:dump("unresolved_records.json",bad); raise RuntimeError(f"{len(bad)} holdings unresolved")
    items=[{"canonical_number":r["canonical_number"],"title":r["title"],"chain_id":CHAIN_ID,"contract":CONTRACT,"token_id_decimal":r["token_id_decimal"],"token_id_hex":r["token_id_hex"],"nonce":r["nonce"],"encoded_creator":r["encoded_creator"],"encoded_supply":str(r["encoded_supply"]),"image":r["image"],"metadata_url":r["metadata_url"],"opensea_item_url":f"https://opensea.io/item/ethereum/{CONTRACT}/{r['token_id_decimal']}"} for r in rows]
    manifest={"schema":"montrealai.becoming.genesis-manifest.v1","generated_at_utc":now(),"chain_id":CHAIN_ID,"collection_slug":SLUG,"contract":CONTRACT,"creator":CREATOR,"item_count":EXPECTED,"items":items}; dump("genesis_manifest_v1.json",manifest); msha=hashlib.sha256((OUT/"genesis_manifest_v1.json").read_bytes()).hexdigest()
    tr=[]
    for r in rows:
        vs=list(r["balance_verification"].values()); tr.append({"canonical_number":r["canonical_number"],"title":r["title"],"chain_id":CHAIN_ID,"contract":CONTRACT,"token_id_decimal":r["token_id_decimal"],"token_id_hex":r["token_id_hex"],"nonce":r["nonce"],"encoded_supply":1,"snapshot_block_number":block["block_number"],"snapshot_block_hash":block["block_hash"],"snapshot_timestamp_utc":block["block_timestamp_utc"],"holder_address":r["candidate_owner"],"balance":1,"holder_category":"creator" if r["candidate_owner"]==CREATOR else "collector_or_contract","owner_resolution":r["owner_resolution"],"last_transfer_block":r.get("last_transfer_block",""),"last_transfer_tx":r.get("last_transfer_tx",""),"rpc_1_balance":vs[0],"rpc_2_balance":vs[1],"opensea_item_url":f"https://opensea.io/item/ethereum/{CONTRACT}/{r['token_id_decimal']}"})
    csvout("snapshot_token_holdings.csv",tr,list(tr[0]))
    by=defaultdict(list)
    for x in tr:by[x["holder_address"]].append(x)
    wr=[]
    for a,xs in by.items():
        nums=sorted(x["canonical_number"] for x in xs); wr.append({"wallet":a,"holder_category":"creator" if a==CREATOR else "collector_or_contract","distinct_genesis_tokens":len(xs),"total_genesis_units":len(xs),"canonical_numbers":" ".join(f"{n:03d}" for n in nums),"first_canonical_number":nums[0],"last_canonical_number":nums[-1],"snapshot_block_number":block["block_number"],"snapshot_block_hash":block["block_hash"]})
    wr.sort(key=lambda x:(-x["distinct_genesis_tokens"],x["wallet"])); csvout("snapshot_wallet_summary.csv",wr,list(wr[0]))
    audit={"schema":"montrealai.becoming.snapshot-audit.v1","status":"PASS","started_at_utc":started,"completed_at_utc":now(),"manifest":{"sha256":msha,"items":len(rows),"unique_ids":len(tc),"numbers_complete":True,"creator_prefix_matches":EXPECTED,"supply_one":EXPECTED,"nonce_padding_zero":EXPECTED,"nonce_minus_number_distribution":dict(sorted(Counter(r["nonce"]-r["canonical_number"] for r in rows).items()))},"snapshot":block,"ownership":{"resolved_tokens":EXPECTED,"unresolved_tokens":0,"distinct_holder_addresses":len(wr),"creator_held_tokens":sum(r["candidate_owner"]==CREATOR for r in rows),"non_creator_held_tokens":sum(r["candidate_owner"]!=CREATOR for r in rows),"verified_units":EXPECTED,"rpc_checks_per_token":2,"blockscout_fallbacks":sum(r["owner_resolution"].startswith("blockscout") for r in rows)}}; dump("snapshot_audit_report.json",audit)
    dump("snapshot_metadata.json",{"schema":"montrealai.becoming.snapshot-metadata.v1","title":"MONTREAL.AI — BECOMING Ω: THE FINAL 444 — Genesis Ownership Snapshot","generated_at_utc":audit["completed_at_utc"],"manifest_sha256":msha,"snapshot_block":block,"contract":CONTRACT,"creator":CREATOR,"item_count":EXPECTED,"holder_count":len(wr),"important":"Decimal token IDs are strings; never import them as floating-point numbers."})
    (OUT/"token_ids_decimal.txt").write_text("\n".join(r["token_id_decimal"] for r in rows)+"\n")
    (OUT/"token_ids_hex.txt").write_text("\n".join(r["token_id_hex"] for r in rows)+"\n")
    (OUT/"snapshot_methodology.md").write_text(f"# MONTREAL.AI — BECOMING Ω: THE FINAL 444\n\nExact 556-token Genesis ownership snapshot at finalized Ethereum block `{block['block_number']}` (`{block['block_hash']}`), timestamp `{block['block_timestamp_utc']}`. Collection membership and owner candidates came from OpenSea's official endpoints using an ephemeral key. Every final holder was independently accepted only after two public Ethereum RPC endpoints returned `balanceOf(holder, tokenId) = 1` at the exact snapshot block. Manifest SHA-256: `{msha}`.\n",encoding="utf-8")
    fs=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!="SHA256SUMS"); (OUT/"SHA256SUMS").write_text("\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in fs)+"\n")
    print(json.dumps({"status":"PASS","items":EXPECTED,"holders":len(wr),"block":block["block_number"],"hash":block["block_hash"],"manifest_sha256":msha},indent=2))

if __name__=="__main__":
    try:main()
    except Exception as e:dump("run_failure.json",{"status":"FAIL","at_utc":now(),"error":repr(e)});raise
