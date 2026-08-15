#!/usr/bin/env python3
import hashlib, json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import becoming_omega_snapshot_v3 as s

GAPS={159,495,523}
TOKEN_LIST_SHA256="aee23ed3fc1e68a759d5d57972846d3530fb3bef93f622fb913ae2b1ec473ce4"
ZERO="0x0000000000000000000000000000000000000000"


def build_rows():
    nonces=[n for n in range(4,563) if n not in GAPS]
    if len(nonces)!=556:raise RuntimeError(f"nonce derivation produced {len(nonces)} IDs")
    creator=int(s.CREATOR,16); rows=[]
    for zero_index,nonce in enumerate(nonces):
        tid=(creator<<96)|(nonce<<40)|1
        declared=316 if zero_index==317 else zero_index
        r={"canonical_number":zero_index+1,"sequence_zero_based":zero_index,"declared_title_number":declared,"title":f"Crypto AI Art #{declared:03d}","token_id_decimal":str(tid),"nonce":nonce,"image":None,"metadata_url":f"https://api.opensea.io/api/v1/metadata/{s.CONTRACT}/{tid}"}
        s.decode(r); rows.append(r)
    payload=("\n".join(r["token_id_decimal"] for r in rows)+"\n").encode()
    got=hashlib.sha256(payload).hexdigest()
    if got!=TOKEN_LIST_SHA256:raise RuntimeError(f"token-list checksum mismatch: {got}")
    return rows


def resolve_transferred(rows,block):
    transferred=[r for r in rows if r.get("verified_balance")!=1 or not r.get("rpc_agreement")]
    histories={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs={pool.submit(s.bshistory,r["token_id_decimal"],block["block_number"]):r for r in transferred}
        for fut in as_completed(futs):
            r=futs[fut]
            try:histories[r["token_id_decimal"]]=fut.result()
            except Exception as e:r["blockscout_error"]=repr(e); histories[r["token_id_decimal"]]=[]
    to_verify=[]; burned=[]
    for r in transferred:
        ev=histories.get(r["token_id_decimal"],[]); r["transfer_count_to_snapshot"]=len(ev)
        if not ev:continue
        bn,li,to,tx=ev[-1]; r["last_transfer_block"]=bn; r["last_transfer_log_index"]=li; r["last_transfer_tx"]=tx
        if to==ZERO:
            r["candidate_owner"]=ZERO; r["owner_resolution"]="burned_at_or_before_snapshot"; r["verified_balance"]=0; r["rpc_agreement"]=True; burned.append(r)
        elif to:
            r["candidate_owner"]=to; r["owner_resolution"]="blockscout_last_transfer_at_or_before_snapshot"; to_verify.append(r)
    if to_verify:s.verify(to_verify,block)
    summary={r["token_id_decimal"]:{"event_count":r.get("transfer_count_to_snapshot",0),"last_transfer_block":r.get("last_transfer_block"),"last_transfer_log_index":r.get("last_transfer_log_index"),"last_transfer_tx":r.get("last_transfer_tx"),"resolved_holder":r.get("candidate_owner"),"resolution":r.get("owner_resolution"),"error":r.get("blockscout_error")} for r in transferred}
    s.dump("transfer_history_summary.json",summary)
    return transferred,burned


def main():
    started=s.now(); rows=build_rows(); block=s.snapshot_block(); s.dump("snapshot_block.json",block)
    for r in rows:r["candidate_owner"]=s.CREATOR; r["owner_resolution"]="creator_balance_at_snapshot"
    s.verify(rows,block)
    transferred,burned=resolve_transferred(rows,block)
    unresolved=[r for r in rows if r["candidate_owner"]!=ZERO and (r.get("verified_balance")!=1 or not r.get("rpc_agreement"))]
    if unresolved:
        s.dump("unresolved_records.json",unresolved); raise RuntimeError(f"{len(unresolved)} holdings unresolved")
    manifest={"schema":"montrealai.becoming.genesis-manifest.v1","generated_at_utc":s.now(),"chain_id":s.CHAIN_ID,"collection_slug":s.SLUG,"contract":s.CONTRACT,"creator":s.CREATOR,"item_count":556,"sequence_rule":"canonical_number 1–556 ordered by encoded mint nonce; immutable declared titles preserved separately","items":[]}
    for r in rows:
        manifest["items"].append({"canonical_number":r["canonical_number"],"sequence_zero_based":r["sequence_zero_based"],"declared_title_number":r["declared_title_number"],"title":r["title"],"chain_id":s.CHAIN_ID,"contract":s.CONTRACT,"token_id_decimal":r["token_id_decimal"],"token_id_hex":r["token_id_hex"],"nonce":r["nonce"],"encoded_creator":r["encoded_creator"],"encoded_supply":str(r["encoded_supply"]),"metadata_url":r["metadata_url"],"opensea_item_url":f"https://opensea.io/item/ethereum/{s.CONTRACT}/{r['token_id_decimal']}"})
    s.dump("genesis_manifest_v1.json",manifest); msha=hashlib.sha256((s.OUT/"genesis_manifest_v1.json").read_bytes()).hexdigest()
    token_rows=[]
    for r in rows:
        vals=list(r.get("balance_verification",{}).values())
        token_rows.append({"canonical_number":r["canonical_number"],"sequence_zero_based":r["sequence_zero_based"],"declared_title_number":r["declared_title_number"],"title":r["title"],"chain_id":s.CHAIN_ID,"contract":s.CONTRACT,"token_id_decimal":r["token_id_decimal"],"token_id_hex":r["token_id_hex"],"nonce":r["nonce"],"encoded_supply":1,"snapshot_block_number":block["block_number"],"snapshot_block_hash":block["block_hash"],"snapshot_timestamp_utc":block["block_timestamp_utc"],"holder_address":r["candidate_owner"],"balance":r["verified_balance"],"holder_category":"burned" if r["candidate_owner"]==ZERO else ("creator" if r["candidate_owner"]==s.CREATOR else "collector_or_contract"),"owner_resolution":r["owner_resolution"],"transfer_count_to_snapshot":r.get("transfer_count_to_snapshot",0),"last_transfer_block":r.get("last_transfer_block",""),"last_transfer_tx":r.get("last_transfer_tx",""),"rpc_1_balance":vals[0] if vals else "","rpc_2_balance":vals[1] if len(vals)>1 else "","opensea_item_url":f"https://opensea.io/item/ethereum/{s.CONTRACT}/{r['token_id_decimal']}"})
    s.csvout("snapshot_token_holdings.csv",token_rows,list(token_rows[0]))
    by=defaultdict(list)
    for x in token_rows:
        if x["holder_address"]!=ZERO:by[x["holder_address"]].append(x)
    wallet_rows=[]
    for a,xs in by.items():
        nums=sorted(x["canonical_number"] for x in xs)
        wallet_rows.append({"wallet":a,"holder_category":"creator" if a==s.CREATOR else "collector_or_contract","distinct_genesis_tokens":len(xs),"total_genesis_units":sum(int(x["balance"]) for x in xs),"canonical_numbers":" ".join(f"{n:03d}" for n in nums),"first_canonical_number":nums[0],"last_canonical_number":nums[-1],"snapshot_block_number":block["block_number"],"snapshot_block_hash":block["block_hash"]})
    wallet_rows.sort(key=lambda x:(-x["distinct_genesis_tokens"],x["wallet"])); s.csvout("snapshot_wallet_summary.csv",wallet_rows,list(wallet_rows[0]))
    title_counts=Counter(r["declared_title_number"] for r in rows)
    anomaly={"schema":"montrealai.becoming.title-numbering-anomalies.v1","principle":"Original titles are preserved verbatim; canonical_number is a separate deterministic sequence.","declared_title_range":"000–555","duplicated_declared_title_numbers":[n for n,c in title_counts.items() if c>1],"absent_declared_title_numbers":[n for n in range(556) if title_counts[n]==0],"anomalies":[{"canonical_number":318,"declared_title":"Crypto AI Art #316","encoded_nonce":322,"token_id_decimal":"2392630434290240917728431095880785304289144848761899072947382790124747096065","finding":"Second distinct #316 token; no immutable title #317 exists."}]}; s.dump("title_numbering_anomalies.json",anomaly)
    derivation={"schema":"montrealai.becoming.token-id-derivation.v1","creator":s.CREATOR,"formula":"token_id = (uint160(creator) << 96) | (nonce << 40) | encoded_supply","encoded_supply":1,"included_nonce_range":[4,562],"excluded_nonces":sorted(GAPS),"item_count":556,"token_ids_decimal_sha256":TOKEN_LIST_SHA256,"validation":"Every generated token ID decodes to the MONTREAL.AI creator and supply 1."}; s.dump("token_id_derivation.json",derivation)
    audit={"schema":"montrealai.becoming.snapshot-audit.v1","status":"PASS","started_at_utc":started,"completed_at_utc":s.now(),"manifest":{"sha256":msha,"items":556,"unique_ids":len({r['token_id_decimal'] for r in rows}),"token_list_sha256":TOKEN_LIST_SHA256,"creator_prefix_matches":sum(r['encoded_creator']==s.CREATOR for r in rows),"supply_one":sum(r['encoded_supply']==1 for r in rows),"nonce_padding_zero":sum(r['nonce_low_byte']==0 for r in rows)},"title_numbering":{"declared_range":"000–555","duplicate_declared_number":316,"absent_declared_number":317,"status":"PRESERVED_AND_DISCLOSED"},"snapshot":block,"ownership":{"resolved_tokens":556,"unresolved_tokens":0,"distinct_holder_addresses":len(wallet_rows),"creator_held_tokens":sum(r['candidate_owner']==s.CREATOR for r in rows),"non_creator_held_tokens":sum(r['candidate_owner'] not in (s.CREATOR,ZERO) for r in rows),"burned_tokens":len(burned),"tokens_requiring_transfer_history":len(transferred),"rpc_checks_per_positive_holder":2,"verified_positive_units":sum(r.get('verified_balance',0) for r in rows)}}; s.dump("snapshot_audit_report.json",audit)
    s.dump("snapshot_metadata.json",{"schema":"montrealai.becoming.snapshot-metadata.v1","title":"MONTREAL.AI — BECOMING Ω: THE FINAL 444 — Genesis Ownership Snapshot","generated_at_utc":audit["completed_at_utc"],"manifest_sha256":msha,"token_ids_decimal_sha256":TOKEN_LIST_SHA256,"snapshot_block":block,"contract":s.CONTRACT,"creator":s.CREATOR,"item_count":556,"holder_count":len(wallet_rows),"title_numbering_note":"The immutable titles run #000–#555, with two distinct #316 tokens and no #317. canonical_number is the separate 1–556 manifest sequence."})
    (s.OUT/"token_ids_decimal.txt").write_text("\n".join(r["token_id_decimal"] for r in rows)+"\n",encoding="utf-8")
    (s.OUT/"token_ids_hex.txt").write_text("\n".join(r["token_id_hex"] for r in rows)+"\n",encoding="utf-8")
    (s.OUT/"snapshot_methodology.md").write_text(f"# MONTREAL.AI — BECOMING Ω: THE FINAL 444\n\n## Exact Genesis ownership snapshot\n\n- Ethereum block: `{block['block_number']}`\n- Block hash: `{block['block_hash']}`\n- Block timestamp: `{block['block_timestamp_utc']}`\n- Legacy ERC-1155 contract: `{s.CONTRACT}`\n- Manifest SHA-256: `{msha}`\n- Token-list SHA-256: `{TOKEN_LIST_SHA256}`\n\nThe 556 exact token IDs are deterministically reconstructed from the creator encoded in OpenSea's legacy token IDs and the verified nonce set 4–562 excluding 159, 495, and 523. No marketplace owner index or persistent API key is used for ownership. At the finalized snapshot block, every token is first tested against the creator with ERC-1155 `balanceOf`. Only tokens that had left the creator are followed through public Blockscout instance transfer history, and the resulting holder is accepted only after two independent public Ethereum RPC endpoints both return the expected historical balance at the exact block.\n\nThe immutable titles are preserved: #000–#555, with two distinct #316 tokens and no #317. `canonical_number` is a separate deterministic 1–556 manifest sequence ordered by encoded mint nonce.\n",encoding="utf-8")
    files=sorted(p for p in s.OUT.iterdir() if p.is_file() and p.name!="SHA256SUMS"); (s.OUT/"SHA256SUMS").write_text("\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in files)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","items":556,"holders":len(wallet_rows),"creator_held":audit['ownership']['creator_held_tokens'],"non_creator_held":audit['ownership']['non_creator_held_tokens'],"block":block['block_number'],"hash":block['block_hash'],"manifest_sha256":msha},indent=2))


if __name__=="__main__":
    try:main()
    except Exception as e:s.dump("run_failure.json",{"status":"FAIL","at_utc":s.now(),"error":repr(e)});raise
