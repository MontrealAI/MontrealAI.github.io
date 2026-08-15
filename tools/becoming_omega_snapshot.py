#!/usr/bin/env python3
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

SLUG = "montrealai"
CHAIN_ID = 1
CONTRACT = "0x495f947276749ce646f68ac8c248420045cb7b5e"
CREATOR = "0x054a2e4b3b5ea2c62372e92358fdf7fb74b4f34a"
EXPECTED_COUNT = 556
OUT = Path(os.environ.get("SNAPSHOT_OUT", "snapshot_output"))
OUT.mkdir(parents=True, exist_ok=True)
USER_AGENT = "MONTREAL.AI-Becoming-Omega-Snapshot/1.0 (+https://montreal.ai)"
RPC_ENDPOINTS = [
    "https://ethereum-rpc.publicnode.com",
    "https://eth.llamarpc.com",
    "https://rpc.flashbots.net",
]
ZERO = "0x0000000000000000000000000000000000000000"


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path, obj):
    path = Path(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def request_json(url, method="GET", payload=None, headers=None, attempts=8, timeout=60, allow_404=False):
    body = None
    hdr = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        hdr.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        hdr["Content-Type"] = "application/json"
    last = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, data=body, headers=hdr, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            if allow_404 and e.code == 404:
                return None
            last = f"HTTP {e.code}: {e.read(500).decode('utf-8', 'replace')}"
            if e.code not in (408, 425, 429, 500, 502, 503, 504):
                raise RuntimeError(f"GET {url}: {last}") from e
        except Exception as e:
            last = repr(e)
        time.sleep(min(45, 1.7 ** attempt))
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}: {last}")


def rpc_call(endpoint, method, params, req_id=1):
    obj = request_json(endpoint, method="POST", payload={"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
    if "error" in obj:
        raise RuntimeError(f"RPC {endpoint} {method}: {obj['error']}")
    return obj.get("result")


def rpc_batch(endpoint, calls, batch_size=100):
    results = {}
    for start in range(0, len(calls), batch_size):
        part = calls[start:start + batch_size]
        payload = [{"jsonrpc": "2.0", "id": i, "method": method, "params": params} for i, method, params in part]
        try:
            response = request_json(endpoint, method="POST", payload=payload, attempts=6, timeout=90)
            if not isinstance(response, list):
                raise RuntimeError("Batch RPC returned non-list")
            by_id = {int(x["id"]): x for x in response}
            for i, _, _ in part:
                item = by_id.get(i)
                if not item:
                    results[i] = {"error": "missing response"}
                elif "error" in item:
                    results[i] = {"error": item["error"]}
                else:
                    results[i] = {"result": item.get("result")}
        except Exception:
            for i, method, params in part:
                try:
                    results[i] = {"result": rpc_call(endpoint, method, params, i)}
                except Exception as e:
                    results[i] = {"error": repr(e)}
    return results


def pick_collection():
    url = "https://api.reservoir.tools/collections/v7?" + urllib.parse.urlencode({"slug": SLUG, "limit": 20, "includeAttributes": "false"})
    data = request_json(url)
    write_json(OUT / "reservoir_collection_raw.json", data)
    cols = data.get("collections") or data.get("items") or []
    if not cols:
        raise RuntimeError("Reservoir returned no collection for slug montrealai")
    exact = [c for c in cols if str(c.get("slug", "")).lower() == SLUG]
    candidates = exact or cols
    def score(c):
        contract = str(c.get("primaryContract") or c.get("contract") or "").lower()
        tc = c.get("tokenCount") or c.get("tokensCount") or 0
        try: tc = int(tc)
        except Exception: tc = 0
        return (contract == CONTRACT.lower(), tc == EXPECTED_COUNT, tc)
    collection = sorted(candidates, key=score, reverse=True)[0]
    collection_id = collection.get("id") or collection.get("collectionId")
    if not collection_id:
        raise RuntimeError(f"Reservoir collection has no id: {collection}")
    return collection, str(collection_id)


def extract_owner(value):
    if not value:
        return None
    if isinstance(value, str):
        return value.lower() if re.fullmatch(r"0x[0-9a-fA-F]{40}", value) else None
    if isinstance(value, dict):
        for key in ("address", "hash", "owner", "addressHash"):
            out = extract_owner(value.get(key))
            if out:
                return out
    return None


def extract_canonical_number(name, image=None, token=None):
    texts = [str(x) for x in (name, image, token) if x]
    patterns = [r"#\s*0*(\d{1,4})(?:\D|$)", r"crypto\s*ai\s*art\D*0*(\d{1,4})(?:\D|$)", r"cryptoaiart0*(\d{1,4})(?:\D|$)"]
    for text in texts:
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 1000:
                    return n
    return None


def normalize_token(item, collection_id):
    tok = item.get("token", item) if isinstance(item, dict) else {}
    tid = tok.get("tokenId") or tok.get("token_id") or tok.get("id") or item.get("tokenId")
    if tid is None:
        return None
    tid = str(tid)
    if not tid.isdigit():
        return None
    contract = str(tok.get("contract") or item.get("contract") or CONTRACT).lower()
    name = tok.get("name") or item.get("name") or ""
    image = tok.get("image") or tok.get("imageSmall") or tok.get("image_url") or item.get("image")
    metadata = tok.get("metadata") or item.get("metadata") or {}
    if not name and isinstance(metadata, dict):
        name = metadata.get("name") or ""
    if not image and isinstance(metadata, dict):
        image = metadata.get("image") or metadata.get("image_url")
    owner = extract_owner(tok.get("owner") or item.get("owner"))
    canonical = extract_canonical_number(name, image, metadata)
    return {
        "collection_id": collection_id,
        "contract": contract,
        "token_id_decimal": tid,
        "title": name,
        "image": image,
        "metadata": metadata,
        "reservoir_owner": owner,
        "canonical_number": canonical,
        "reservoir_raw": item,
    }


def fetch_reservoir_tokens(collection_id):
    tokens = {}
    raw_pages = []
    continuation = None
    page = 0
    while True:
        params = {"collection": collection_id, "limit": 100, "includeAttributes": "true", "flagStatus": -1}
        if continuation:
            params["continuation"] = continuation
        url = "https://api.reservoir.tools/tokens/v7?" + urllib.parse.urlencode(params)
        data = request_json(url)
        page += 1
        raw_pages.append({"page": page, "continuation_in": continuation, "response": data})
        items = data.get("tokens") or data.get("items") or []
        for item in items:
            rec = normalize_token(item, collection_id)
            if rec and rec["contract"] == CONTRACT.lower():
                tokens[rec["token_id_decimal"]] = rec
        new_cont = data.get("continuation") or data.get("next") or data.get("nextContinuation")
        if not new_cont or new_cont == continuation:
            break
        continuation = str(new_cont)
        if page > 50:
            raise RuntimeError("Reservoir token pagination exceeded 50 pages")
        time.sleep(0.12)
    write_json(OUT / "reservoir_tokens_raw.json", raw_pages)
    return list(tokens.values())


def derive_token_fields(rec):
    tid = int(rec["token_id_decimal"])
    rec["token_id_hex"] = "0x" + tid.to_bytes(32, "big").hex()
    rec["encoded_creator"] = "0x" + ((tid >> 96) & ((1 << 160) - 1)).to_bytes(20, "big").hex()
    middle = (tid >> 32) & ((1 << 64) - 1)
    rec["nonce"] = middle >> 8
    rec["nonce_low_byte"] = middle & 0xFF
    rec["encoded_supply"] = tid & ((1 << 32) - 1)
    return rec


def determine_snapshot_block():
    finalized = []
    errors = {}
    for endpoint in RPC_ENDPOINTS:
        try:
            b = rpc_call(endpoint, "eth_getBlockByNumber", ["finalized", False])
            if not b:
                raise RuntimeError("null finalized block")
            finalized.append({"endpoint": endpoint, "number": int(b["number"], 16), "hash": b["hash"].lower(), "timestamp": int(b["timestamp"], 16)})
        except Exception as e:
            errors[endpoint] = repr(e)
    if len(finalized) < 2:
        raise RuntimeError(f"Fewer than two RPC endpoints returned finalized blocks: {finalized}; errors={errors}")
    chosen_number = min(x["number"] for x in finalized)
    exact = []
    for x in finalized:
        try:
            b = rpc_call(x["endpoint"], "eth_getBlockByNumber", [hex(chosen_number), False])
            exact.append({"endpoint": x["endpoint"], "number": chosen_number, "hash": b["hash"].lower(), "timestamp": int(b["timestamp"], 16)})
        except Exception as e:
            errors[x["endpoint"] + ":exact"] = repr(e)
    hash_counts = Counter(x["hash"] for x in exact)
    if not hash_counts or hash_counts.most_common(1)[0][1] < 2:
        raise RuntimeError(f"No two RPC endpoints agreed on chosen block hash: {exact}; errors={errors}")
    chosen_hash = hash_counts.most_common(1)[0][0]
    agreed = [x for x in exact if x["hash"] == chosen_hash]
    chosen = agreed[0]
    return {
        "selection_rule": "Minimum block number independently reported as finalized by at least two public Ethereum RPC endpoints; exact block hash then cross-checked.",
        "block_number": chosen_number,
        "block_number_hex": hex(chosen_number),
        "block_hash": chosen_hash,
        "block_timestamp_unix": chosen["timestamp"],
        "block_timestamp_utc": datetime.fromtimestamp(chosen["timestamp"], timezone.utc).isoformat().replace("+00:00", "Z"),
        "finalized_observations": finalized,
        "exact_block_observations": exact,
        "errors": errors,
        "agreed_endpoints": [x["endpoint"] for x in agreed],
    }


def balance_call_data(owner, token_id):
    return "0x00fdd58e" + owner[2:].lower().zfill(64) + hex(int(token_id))[2:].zfill(64)


def verify_balances(records, block, endpoints):
    calls = []
    by_id = {}
    for i, rec in enumerate(records, start=1):
        owner = rec.get("candidate_owner")
        if not owner:
            continue
        calls.append((i, "eth_call", [{"to": CONTRACT, "data": balance_call_data(owner, rec["token_id_decimal"])}, block["block_number_hex"]]))
        by_id[i] = rec
    endpoint_results = {}
    for endpoint in endpoints:
        batch = rpc_batch(endpoint, calls, batch_size=75)
        parsed = {}
        for i, result in batch.items():
            if "result" in result and result["result"] is not None:
                try:
                    parsed[i] = int(result["result"], 16)
                except Exception:
                    parsed[i] = None
            else:
                parsed[i] = None
        endpoint_results[endpoint] = parsed
    for i, rec in by_id.items():
        vals = {ep: endpoint_results[ep].get(i) for ep in endpoints}
        rec["balance_verification"] = vals
        non_null = [v for v in vals.values() if v is not None]
        rec["verified_balance"] = non_null[0] if non_null and len(set(non_null)) == 1 else None
        rec["rpc_agreement"] = bool(non_null and len(non_null) >= 2 and len(set(non_null)) == 1)
    return endpoint_results


def addr_from(value):
    if isinstance(value, str) and re.fullmatch(r"0x[0-9a-fA-F]{40}", value):
        return value.lower()
    if isinstance(value, dict):
        for key in ("hash", "address_hash", "address"):
            out = addr_from(value.get(key))
            if out:
                return out
    return None


def fetch_transfer_history(token_id, snapshot_block):
    base = f"https://eth.blockscout.com/api/v2/tokens/{CONTRACT}/instances/{token_id}/transfers"
    params = None
    events = []
    seen = set()
    pages = 0
    while True:
        url = base if not params else base + "?" + urllib.parse.urlencode(params)
        data = request_json(url, allow_404=True, attempts=6)
        if data is None:
            return []
        pages += 1
        for item in data.get("items") or []:
            try:
                bn = int(item.get("block_number") or item.get("blockNumber") or 0)
            except Exception:
                continue
            if bn > snapshot_block:
                continue
            tx = item.get("transaction_hash") or item.get("tx_hash") or item.get("transactionHash")
            try:
                li = int(item.get("log_index") or item.get("logIndex") or item.get("index") or 0)
            except Exception:
                li = 0
            key = (bn, li, tx)
            if key in seen:
                continue
            seen.add(key)
            events.append({
                "block_number": bn,
                "log_index": li,
                "transaction_hash": tx,
                "from": addr_from(item.get("from")),
                "to": addr_from(item.get("to")),
                "timestamp": item.get("timestamp"),
                "raw": item,
            })
        nxt = data.get("next_page_params")
        if not nxt or pages >= 50:
            break
        params = {k: v for k, v in nxt.items() if v is not None}
        time.sleep(0.08)
    return sorted(events, key=lambda x: (x["block_number"], x["log_index"]))


def resolve_failed(records, block):
    failed = [r for r in records if r.get("verified_balance") != r.get("encoded_supply") or not r.get("rpc_agreement")]
    if not failed:
        return
    histories = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(fetch_transfer_history, r["token_id_decimal"], block["block_number"]): r for r in failed}
        for fut in as_completed(futs):
            rec = futs[fut]
            try:
                histories[rec["token_id_decimal"]] = fut.result()
            except Exception as e:
                rec["transfer_history_error"] = repr(e)
                histories[rec["token_id_decimal"]] = []
    for rec in failed:
        events = histories.get(rec["token_id_decimal"], [])
        rec["transfer_count_to_snapshot"] = len(events)
        if events:
            last = events[-1]
            rec["last_transfer_block"] = last["block_number"]
            rec["last_transfer_log_index"] = last["log_index"]
            rec["last_transfer_tx"] = last["transaction_hash"]
            rec["candidate_owner"] = last["to"]
            rec["owner_resolution"] = "blockscout_last_transfer_at_or_before_snapshot"
        else:
            rec["candidate_owner"] = CREATOR
            rec["owner_resolution"] = "implicit_creator_balance_no_indexed_transfer"
    endpoints = block["agreed_endpoints"][:2]
    verify_balances(failed, block, endpoints)


def build_manifest(records, collection_id):
    manifest = {
        "schema": "montrealai.becoming.genesis-manifest.v1",
        "generated_at_utc": utc_now(),
        "chain_id": CHAIN_ID,
        "collection_slug": SLUG,
        "collection_id": collection_id,
        "contract": CONTRACT,
        "creator": CREATOR,
        "expected_item_count": EXPECTED_COUNT,
        "items": [],
    }
    for rec in records:
        manifest["items"].append({
            "canonical_number": rec["canonical_number"],
            "title": rec["title"],
            "chain_id": CHAIN_ID,
            "contract": CONTRACT,
            "token_id_decimal": rec["token_id_decimal"],
            "token_id_hex": rec["token_id_hex"],
            "nonce": rec["nonce"],
            "encoded_creator": rec["encoded_creator"],
            "encoded_supply": str(rec["encoded_supply"]),
            "image": rec.get("image"),
            "opensea_item_url": f"https://opensea.io/item/ethereum/{CONTRACT}/{rec['token_id_decimal']}",
        })
    return manifest


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    started = utc_now()
    collection, collection_id = pick_collection()
    records = fetch_reservoir_tokens(collection_id)
    if not records:
        raise RuntimeError("No token records retrieved")
    for rec in records:
        derive_token_fields(rec)
    records = [r for r in records if r.get("canonical_number") is not None and 1 <= r["canonical_number"] <= EXPECTED_COUNT]
    records.sort(key=lambda r: r["canonical_number"])
    canonical_counts = Counter(r["canonical_number"] for r in records)
    token_counts = Counter(r["token_id_decimal"] for r in records)
    missing_numbers = sorted(set(range(1, EXPECTED_COUNT + 1)) - set(canonical_counts))
    duplicates = sorted(n for n, c in canonical_counts.items() if c != 1)
    duplicate_ids = sorted(t for t, c in token_counts.items() if c != 1)
    if len(records) != EXPECTED_COUNT or missing_numbers or duplicates or duplicate_ids:
        diagnostics = {
            "retrieved_numbered_records": len(records),
            "missing_numbers": missing_numbers,
            "duplicate_numbers": duplicates,
            "duplicate_token_ids": duplicate_ids,
            "sample": [{k: r.get(k) for k in ("canonical_number", "title", "token_id_decimal")} for r in records[:25]],
        }
        write_json(OUT / "manifest_failure_diagnostics.json", diagnostics)
        raise RuntimeError(f"Canonical manifest validation failed: {diagnostics}")
    block = determine_snapshot_block()
    write_json(OUT / "snapshot_block.json", block)
    for rec in records:
        rec["candidate_owner"] = rec.get("reservoir_owner") or CREATOR
        rec["owner_resolution"] = "reservoir_current_owner_cross_checked_at_snapshot" if rec.get("reservoir_owner") else "creator_fallback_cross_checked_at_snapshot"
    verify_balances(records, block, block["agreed_endpoints"][:2])
    resolve_failed(records, block)
    unresolved = [r for r in records if r.get("verified_balance") != r.get("encoded_supply") or not r.get("rpc_agreement")]
    if unresolved:
        write_json(OUT / "unresolved_records.json", [{k: r.get(k) for k in ("canonical_number", "token_id_decimal", "title", "reservoir_owner", "candidate_owner", "owner_resolution", "verified_balance", "balance_verification", "transfer_history_error")} for r in unresolved])
        raise RuntimeError(f"{len(unresolved)} holdings remained unresolved after historical transfer fallback")
    manifest = build_manifest(records, collection_id)
    write_json(OUT / "genesis_manifest_v1.json", manifest)
    manifest_sha = hashlib.sha256((OUT / "genesis_manifest_v1.json").read_bytes()).hexdigest()
    token_rows = []
    for rec in records:
        token_rows.append({
            "canonical_number": rec["canonical_number"],
            "title": rec["title"],
            "chain_id": CHAIN_ID,
            "contract": CONTRACT,
            "token_id_decimal": rec["token_id_decimal"],
            "token_id_hex": rec["token_id_hex"],
            "nonce": rec["nonce"],
            "encoded_supply": rec["encoded_supply"],
            "snapshot_block_number": block["block_number"],
            "snapshot_block_hash": block["block_hash"],
            "snapshot_timestamp_utc": block["block_timestamp_utc"],
            "holder_address": rec["candidate_owner"],
            "balance": rec["verified_balance"],
            "holder_category": "creator" if rec["candidate_owner"] == CREATOR else "collector_or_contract",
            "owner_resolution": rec["owner_resolution"],
            "reservoir_owner_at_execution": rec.get("reservoir_owner") or "",
            "last_transfer_block": rec.get("last_transfer_block", ""),
            "last_transfer_tx": rec.get("last_transfer_tx", ""),
            "rpc_1_balance": list(rec["balance_verification"].values())[0],
            "rpc_2_balance": list(rec["balance_verification"].values())[1],
            "opensea_item_url": f"https://opensea.io/item/ethereum/{CONTRACT}/{rec['token_id_decimal']}",
        })
    fields = list(token_rows[0].keys())
    write_csv(OUT / "snapshot_token_holdings.csv", token_rows, fields)
    by_wallet = defaultdict(list)
    for row in token_rows:
        by_wallet[row["holder_address"]].append(row)
    wallet_rows = []
    for wallet, items in by_wallet.items():
        nums = sorted(int(x["canonical_number"]) for x in items)
        wallet_rows.append({
            "wallet": wallet,
            "holder_category": "creator" if wallet == CREATOR else "collector_or_contract",
            "distinct_genesis_tokens": len(items),
            "total_genesis_units": sum(int(x["balance"]) for x in items),
            "canonical_numbers": " ".join(f"{n:03d}" for n in nums),
            "first_canonical_number": nums[0],
            "last_canonical_number": nums[-1],
            "snapshot_block_number": block["block_number"],
            "snapshot_block_hash": block["block_hash"],
        })
    wallet_rows.sort(key=lambda r: (-r["distinct_genesis_tokens"], r["wallet"]))
    write_csv(OUT / "snapshot_wallet_summary.csv", wallet_rows, list(wallet_rows[0].keys()))
    offsets = Counter(r["nonce"] - r["canonical_number"] for r in records)
    audit = {
        "schema": "montrealai.becoming.snapshot-audit.v1",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "status": "PASS",
        "collection": {"slug": SLUG, "reservoir_collection_id": collection_id, "reservoir_name": collection.get("name"), "expected_items": EXPECTED_COUNT, "manifest_items": len(records)},
        "manifest": {"sha256": manifest_sha, "unique_token_ids": len({r["token_id_decimal"] for r in records}), "canonical_numbers_complete_1_to_556": [r["canonical_number"] for r in records] == list(range(1, EXPECTED_COUNT + 1)), "creator_prefix_matches": sum(r["encoded_creator"] == CREATOR for r in records), "encoded_supply_equals_one": sum(r["encoded_supply"] == 1 for r in records), "nonce_low_byte_zero": sum(r["nonce_low_byte"] == 0 for r in records), "nonce_minus_canonical_distribution": dict(sorted(offsets.items()))},
        "snapshot": block,
        "ownership": {"resolved_tokens": len(records), "unresolved_tokens": 0, "distinct_holder_addresses": len(wallet_rows), "creator_held_tokens": sum(r["candidate_owner"] == CREATOR for r in records), "non_creator_held_tokens": sum(r["candidate_owner"] != CREATOR for r in records), "total_verified_units": sum(r["verified_balance"] for r in records), "rpc_cross_checks_per_token": 2, "fallback_transfer_history_resolutions": sum(r["owner_resolution"].startswith("blockscout") for r in records)},
        "sources": {"collection_membership_and_metadata": "Reservoir collection index, cross-validated against token ID encoding and canonical numbering", "ownership": "Ethereum eth_call balanceOf(address,uint256) at the exact finalized snapshot block, independently through two public RPC endpoints", "historical_fallback": "Blockscout per-instance transfer history, used only when the current index owner did not match the finalized block"},
    }
    write_json(OUT / "snapshot_audit_report.json", audit)
    metadata = {
        "schema": "montrealai.becoming.snapshot-metadata.v1",
        "title": "MONTREAL.AI — BECOMING Ω: THE FINAL 444 — Genesis Ownership Snapshot",
        "generated_at_utc": audit["completed_at_utc"],
        "manifest_file": "genesis_manifest_v1.json",
        "manifest_sha256": manifest_sha,
        "token_holdings_file": "snapshot_token_holdings.csv",
        "wallet_summary_file": "snapshot_wallet_summary.csv",
        "snapshot_block": block,
        "contract": CONTRACT,
        "creator": CREATOR,
        "item_count": len(records),
        "holder_count": len(wallet_rows),
        "important": "Token IDs are serialized as decimal strings and 32-byte hexadecimal strings. Do not import decimal token IDs as floating-point numbers.",
    }
    write_json(OUT / "snapshot_metadata.json", metadata)
    methodology = f"""# MONTREAL.AI — BECOMING Ω: THE FINAL 444\n\n## Genesis ownership snapshot methodology\n\n- Collection slug: `{SLUG}`\n- Ethereum chain ID: `{CHAIN_ID}`\n- Legacy ERC-1155 contract: `{CONTRACT}`\n- Creator encoded in token IDs: `{CREATOR}`\n- Canonical Genesis works: `{EXPECTED_COUNT}`\n- Snapshot block: `{block['block_number']}`\n- Snapshot block hash: `{block['block_hash']}`\n- Snapshot block time: `{block['block_timestamp_utc']}`\n- Genesis manifest SHA-256: `{manifest_sha}`\n\nThe 556 collection records were retrieved from the public Reservoir collection index for the OpenSea slug, then strictly validated as one unique token for every canonical number 001–556. Each 256-bit ERC-1155 token ID was decoded to verify the MONTREAL.AI creator prefix, zero nonce padding, and encoded supply of one.\n\nThe snapshot block is the minimum block independently reported as finalized by at least two public Ethereum RPC endpoints. Each holder candidate was verified by calling the legacy contract's `balanceOf(address,uint256)` at that exact block through two independent endpoints. If a current index owner did not match the finalized block, the holder was reconstructed from public Blockscout transfer history at or before the snapshot block and checked again through both RPCs.\n\nThe authoritative ownership table is `snapshot_token_holdings.csv`. The wallet table is a derived convenience summary.\n"""
    (OUT / "snapshot_methodology.md").write_text(methodology, encoding="utf-8")
    (OUT / "token_ids_decimal.txt").write_text("\n".join(r["token_id_decimal"] for r in records) + "\n", encoding="utf-8")
    (OUT / "token_ids_hex.txt").write_text("\n".join(r["token_id_hex"] for r in records) + "\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    sums = []
    for p in files:
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT), "block": block["block_number"], "hash": block["block_hash"], "items": len(records), "holders": len(wallet_rows), "manifest_sha256": manifest_sha}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(OUT / "run_failure.json", {"status": "FAIL", "at_utc": utc_now(), "error": repr(exc)})
        raise
