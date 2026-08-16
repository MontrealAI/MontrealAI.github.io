#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import requests

COLLECTION = "MONTREAL.AI — BECOMING: GENESIS 556"
CONTRACT = "0x495f947276749ce646f68ac8c248420045cb7b5e"
CREATOR = "0x054a2e4b3b5ea2c62372e92358fdf7fb74b4f34a"
CHAIN_ID = 1
EXPECTED_COUNT = 556
EXPECTED_UNIT_SUPPLY = 1
EXCLUDED_NONCES = {161, 495, 523}
BASE_URL = "https://eth.blockscout.com/api/v2"
RPC_URLS = [
    value.strip()
    for value in os.environ.get(
        "ETH_RPC_URLS",
        "https://ethereum-rpc.publicnode.com,https://eth.llamarpc.com,https://rpc.flashbots.net,https://eth.drpc.org",
    ).split(",")
    if value.strip()
]
OUT = Path(os.environ.get("SNAPSHOT_OUT", "snapshot_output"))
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "35"))
HTTP_RETRIES = int(os.environ.get("HTTP_RETRIES", "7"))
WORKERS = int(os.environ.get("SNAPSHOT_WORKERS", "6"))
RPC_BATCH_SIZE = int(os.environ.get("RPC_BATCH_SIZE", "100"))
_thread_local = threading.local()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def session() -> requests.Session:
    value = getattr(_thread_local, "session", None)
    if value is None:
        value = requests.Session()
        value.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "MONTREALAI-Becoming-Genesis-556-Snapshot/2.0",
        })
        _thread_local.session = value
    return value


def backoff(attempt: int) -> None:
    time.sleep(min(25.0, 0.8 * (2 ** attempt)))


def get_json(url: str, params: dict[str, Any] | None = None, allow_404: bool = False) -> dict[str, Any] | None:
    last: Exception | None = None
    for attempt in range(HTTP_RETRIES + 1):
        try:
            response = session().get(url, params=params, timeout=HTTP_TIMEOUT)
            if allow_404 and response.status_code == 404:
                return None
            if response.status_code == 429 or response.status_code >= 500:
                last = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
                backoff(attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object, received {type(payload).__name__}")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last = exc
            if attempt >= HTTP_RETRIES:
                break
            backoff(attempt)
    raise RuntimeError(f"GET failed: {url}: {last}")


def rpc_call(url: str, method: str, params: list[Any], retries: int = 5) -> Any:
    last: Exception | None = None
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    for attempt in range(retries + 1):
        try:
            response = session().post(url, json=payload, timeout=HTTP_TIMEOUT)
            if response.status_code == 429 or response.status_code >= 500:
                last = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
                backoff(attempt)
                continue
            response.raise_for_status()
            body = response.json()
            if "error" in body:
                raise RuntimeError(json.dumps(body["error"], ensure_ascii=False))
            if "result" not in body:
                raise RuntimeError(f"Malformed RPC response: {str(body)[:500]}")
            return body["result"]
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last = exc
            if attempt >= retries:
                break
            backoff(attempt)
    raise RuntimeError(f"RPC {method} failed at {url}: {last}")


def normalize_address(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("hash", "address_hash", "address"):
            candidate = value.get(key)
            result = normalize_address(candidate)
            if result:
                return result
        return None
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if re.fullmatch(r"0x[0-9a-f]{40}", value) else None


def extract_address(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in payload:
            result = normalize_address(payload.get(key))
            if result:
                return result
    return None


def build_manifest() -> list[dict[str, Any]]:
    nonces = [n for n in range(5, 564) if n not in EXCLUDED_NONCES]
    if len(nonces) != EXPECTED_COUNT:
        raise AssertionError(f"Nonce construction returned {len(nonces)} entries")
    nonces[257], nonces[258] = nonces[258], nonces[257]
    creator_int = int(CREATOR, 16)
    rows: list[dict[str, Any]] = []
    for canonical_number, nonce in enumerate(nonces, start=1):
        token_id = (creator_int << 96) | (nonce << 40) | EXPECTED_UNIT_SUPPLY
        token_decimal = str(token_id)
        rows.append({
            "canonical_number": canonical_number,
            "title": f"Crypto AI Art #{canonical_number:03d}",
            "chain_id": CHAIN_ID,
            "contract": CONTRACT,
            "standard": "ERC-1155",
            "token_id_decimal": token_decimal,
            "token_id_hex": f"0x{token_id:064x}",
            "creator_encoded": CREATOR,
            "internal_nonce": nonce,
            "edition_supply": EXPECTED_UNIT_SUPPLY,
            "opensea_item_url": f"https://opensea.io/item/ethereum/{CONTRACT}/{token_decimal}",
            "blockscout_item_url": f"https://eth.blockscout.com/token/{CONTRACT}/instance/{token_decimal}",
        })
    if len({row["token_id_decimal"] for row in rows}) != EXPECTED_COUNT:
        raise AssertionError("Token IDs are not unique")
    return rows


def title_number(name: Any) -> int | None:
    if not isinstance(name, str):
        return None
    match = re.search(r"#\s*0*([0-9]{1,4})\b", name)
    return int(match.group(1)) if match else None


def paginate(url: str, initial_params: dict[str, Any] | None = None) -> Iterable[dict[str, Any]]:
    params = dict(initial_params or {})
    seen: set[str] = set()
    for _ in range(200):
        fingerprint = json.dumps(params, sort_keys=True, default=str)
        if fingerprint in seen:
            raise RuntimeError(f"Pagination loop detected for {url}: {params}")
        seen.add(fingerprint)
        payload = get_json(url, params=params)
        assert payload is not None
        for item in payload.get("items") or []:
            if isinstance(item, dict):
                yield item
        next_params = payload.get("next_page_params")
        if not isinstance(next_params, dict) or not next_params:
            return
        params = {k: v for k, v in next_params.items() if v is not None}
    raise RuntimeError(f"Pagination exceeded safety limit for {url}")


def discover_token(row: dict[str, Any]) -> dict[str, Any]:
    token_id = row["token_id_decimal"]
    instance_url = f"{BASE_URL}/tokens/{CONTRACT}/instances/{token_id}"
    holders_url = f"{instance_url}/holders"
    candidates: set[str] = {CREATOR}
    holder_values: dict[str, int] = defaultdict(int)
    ens_by_address: dict[str, str] = {}
    instance_error = ""
    holders_error = ""
    instance_status = "MISSING"
    returned_id = ""
    metadata_name = ""
    metadata_number: int | None = None
    returned_contract = ""
    blockscout_owner = ""
    is_unique: Any = None

    try:
        instance = get_json(instance_url, allow_404=True)
        if instance is not None:
            returned_id = str(instance.get("id", ""))
            is_unique = instance.get("is_unique")
            metadata = instance.get("metadata") if isinstance(instance.get("metadata"), dict) else {}
            metadata_name = str(metadata.get("name") or "")
            metadata_number = title_number(metadata_name)
            token = instance.get("token") if isinstance(instance.get("token"), dict) else {}
            returned_contract = str(token.get("address_hash") or token.get("address") or "").lower()
            owner = extract_address(instance, "holder_address_hash", "owner", "address")
            if owner:
                candidates.add(owner)
                blockscout_owner = owner
            instance_status = "FOUND"
    except Exception as exc:
        instance_error = str(exc)
        instance_status = "ERROR"

    try:
        for item in paginate(holders_url):
            holder = extract_address(item, "address_hash", "address", "holder_address_hash", "holder")
            if not holder:
                continue
            candidates.add(holder)
            raw_value = item.get("value", item.get("balance", "0"))
            try:
                value = int(str(raw_value), 0) if str(raw_value).lower().startswith("0x") else int(str(raw_value))
            except Exception:
                value = 0
            holder_values[holder] += value
            address_obj = item.get("address_hash") if isinstance(item.get("address_hash"), dict) else item.get("address")
            if isinstance(address_obj, dict):
                ens = address_obj.get("ens_domain_name")
                if isinstance(ens, str) and ens.strip():
                    ens_by_address[holder] = ens.strip()
    except Exception as exc:
        holders_error = str(exc)

    id_match = returned_id == token_id if returned_id else False
    contract_match = (not returned_contract) or returned_contract == CONTRACT
    metadata_number_match = metadata_number == row["canonical_number"] if metadata_number is not None else None
    return {
        "canonical_number": row["canonical_number"],
        "token_id_decimal": token_id,
        "expected_title": row["title"],
        "instance_status": instance_status,
        "returned_id": returned_id,
        "id_match": id_match,
        "returned_contract": returned_contract,
        "contract_match": contract_match,
        "metadata_name": metadata_name,
        "metadata_number": metadata_number,
        "metadata_number_match": metadata_number_match,
        "is_unique": is_unique,
        "blockscout_owner": blockscout_owner,
        "reported_holder_count": len(holder_values),
        "reported_holder_balance_sum": sum(holder_values.values()),
        "candidate_addresses": sorted(candidates),
        "holder_values": dict(holder_values),
        "ens_by_address": ens_by_address,
        "instance_error": instance_error,
        "holders_error": holders_error,
        "instance_source_url": instance_url,
        "holders_source_url": holders_url,
    }


def select_rpc() -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for url in RPC_URLS:
        try:
            chain_id_hex = rpc_call(url, "eth_chainId", [], retries=2)
            chain_id = int(chain_id_hex, 16)
            if chain_id != CHAIN_ID:
                raise RuntimeError(f"wrong chain ID {chain_id}")
            block = rpc_call(url, "eth_getBlockByNumber", ["finalized", False], retries=2)
            if not isinstance(block, dict) or not block.get("number") or not block.get("hash"):
                raise RuntimeError("no finalized block returned")
            code = rpc_call(url, "eth_getCode", [CONTRACT, block["number"]], retries=2)
            if not isinstance(code, str) or code == "0x":
                raise RuntimeError("shared storefront contract has no code at finalized block")
            attempts.append({"url": url, "status": "PASS", "chain_id": chain_id, "block_number": int(block["number"], 16), "block_hash": block["hash"]})
            return url, block, attempts
        except Exception as exc:
            attempts.append({"url": url, "status": "FAIL", "error": str(exc)})
    raise RuntimeError(f"No Ethereum RPC endpoint succeeded: {json.dumps(attempts, ensure_ascii=False)}")


def crosscheck_block(primary_url: str, block: dict[str, Any]) -> list[dict[str, Any]]:
    number_hex = block["number"]
    expected_hash = str(block["hash"]).lower()
    results: list[dict[str, Any]] = []
    for url in RPC_URLS:
        if url == primary_url:
            continue
        try:
            other = rpc_call(url, "eth_getBlockByNumber", [number_hex, False], retries=1)
            actual_hash = str((other or {}).get("hash", "")).lower() if isinstance(other, dict) else ""
            results.append({
                "url": url,
                "status": "PASS" if actual_hash == expected_hash else "MISMATCH",
                "block_number": int(number_hex, 16),
                "expected_hash": expected_hash,
                "actual_hash": actual_hash,
            })
        except Exception as exc:
            results.append({"url": url, "status": "ERROR", "error": str(exc)})
    return results


def encode_balance_of_batch(accounts: list[str], ids: list[int]) -> str:
    if len(accounts) != len(ids):
        raise ValueError("accounts and ids length mismatch")
    selector = bytes.fromhex("4e1273f4")
    n = len(accounts)
    offset_accounts = 64
    accounts_blob = n.to_bytes(32, "big") + b"".join(bytes.fromhex(address[2:]).rjust(32, b"\x00") for address in accounts)
    offset_ids = offset_accounts + len(accounts_blob)
    ids_blob = n.to_bytes(32, "big") + b"".join(value.to_bytes(32, "big") for value in ids)
    head = offset_accounts.to_bytes(32, "big") + offset_ids.to_bytes(32, "big")
    return "0x" + (selector + head + accounts_blob + ids_blob).hex()


def decode_uint_array(value: str) -> list[int]:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError("Invalid eth_call result")
    data = bytes.fromhex(value[2:])
    if len(data) < 64:
        raise ValueError("Truncated dynamic array result")
    offset = int.from_bytes(data[:32], "big")
    if offset + 32 > len(data):
        raise ValueError("Invalid dynamic array offset")
    length = int.from_bytes(data[offset:offset + 32], "big")
    start = offset + 32
    end = start + 32 * length
    if end > len(data):
        raise ValueError("Truncated dynamic array values")
    return [int.from_bytes(data[start + i * 32:start + (i + 1) * 32], "big") for i in range(length)]


def query_balances(rpc_url: str, block_hex: str, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(pairs), RPC_BATCH_SIZE):
        chunk = pairs[start:start + RPC_BATCH_SIZE]
        accounts = [address for _, address in chunk]
        ids = [int(token_id) for token_id, _ in chunk]
        data = encode_balance_of_batch(accounts, ids)
        result = rpc_call(rpc_url, "eth_call", [{"to": CONTRACT, "data": data}, block_hex], retries=5)
        balances = decode_uint_array(result)
        if len(balances) != len(chunk):
            raise RuntimeError(f"balanceOfBatch returned {len(balances)} values for {len(chunk)} pairs")
        for (token_id, address), balance in zip(chunk, balances):
            rows.append({"token_id_decimal": token_id, "holder_address": address, "balance": balance})
        print(f"Verified {min(start + RPC_BATCH_SIZE, len(pairs))}/{len(pairs)} candidate balances", flush=True)
    return rows


def transfer_candidates(row: dict[str, Any]) -> tuple[set[str], str]:
    token_id = row["token_id_decimal"]
    url = f"{BASE_URL}/tokens/{CONTRACT}/instances/{token_id}/transfers"
    addresses: set[str] = {CREATOR}
    try:
        for item in paginate(url):
            for key in ("from", "to", "from_address_hash", "to_address_hash"):
                address = normalize_address(item.get(key))
                if address:
                    addresses.add(address)
        return addresses, ""
    except Exception as exc:
        return addresses, str(exc)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    manifest = build_manifest()
    by_id = {row["token_id_decimal"]: row for row in manifest}
    print(f"Built {len(manifest)} deterministic Genesis IDs", flush=True)

    discoveries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
        futures = {pool.submit(discover_token, row): row for row in manifest}
        for completed, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                discoveries.append(future.result())
            except Exception as exc:
                discoveries.append({
                    "canonical_number": row["canonical_number"],
                    "token_id_decimal": row["token_id_decimal"],
                    "expected_title": row["title"],
                    "instance_status": "ERROR",
                    "returned_id": "",
                    "id_match": False,
                    "returned_contract": "",
                    "contract_match": False,
                    "metadata_name": "",
                    "metadata_number": None,
                    "metadata_number_match": None,
                    "is_unique": None,
                    "blockscout_owner": "",
                    "reported_holder_count": 0,
                    "reported_holder_balance_sum": 0,
                    "candidate_addresses": [CREATOR],
                    "holder_values": {},
                    "ens_by_address": {},
                    "instance_error": str(exc),
                    "holders_error": "",
                    "instance_source_url": f"{BASE_URL}/tokens/{CONTRACT}/instances/{row['token_id_decimal']}",
                    "holders_source_url": f"{BASE_URL}/tokens/{CONTRACT}/instances/{row['token_id_decimal']}/holders",
                })
            if completed % 25 == 0 or completed == len(futures):
                print(f"Blockscout discovery: {completed}/{len(futures)}", flush=True)
    discoveries.sort(key=lambda x: x["canonical_number"])

    rpc_url, block, rpc_attempts = select_rpc()
    block_number = int(block["number"], 16)
    block_hash = str(block["hash"])
    block_timestamp_dt = datetime.fromtimestamp(int(block["timestamp"], 16), timezone.utc)
    block_timestamp_utc = block_timestamp_dt.isoformat(timespec="seconds").replace("+00:00", "Z")
    block_timestamp_montreal = block_timestamp_dt.astimezone(ZoneInfo("America/Montreal")).isoformat(timespec="seconds")
    block_hex = block["number"]
    rpc_crosschecks = crosscheck_block(rpc_url, block)
    print(f"Snapshot block: {block_number} {block_hash} at {block_timestamp_utc}", flush=True)

    candidates: dict[str, set[str]] = {row["token_id_decimal"]: {CREATOR} for row in manifest}
    ens_by_address: dict[str, str] = {}
    for discovery in discoveries:
        token_id = discovery["token_id_decimal"]
        for address in discovery.get("candidate_addresses") or []:
            normalized = normalize_address(address)
            if normalized:
                candidates[token_id].add(normalized)
        for address, ens in (discovery.get("ens_by_address") or {}).items():
            normalized = normalize_address(address)
            if normalized and isinstance(ens, str) and ens:
                ens_by_address[normalized] = ens

    def make_pairs() -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for row in manifest:
            token_id = row["token_id_decimal"]
            for address in sorted(candidates[token_id]):
                result.append((token_id, address))
        return result

    queried = query_balances(rpc_url, block_hex, make_pairs())

    def positive_by_token(values: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in values:
            if int(value["balance"]) > 0:
                result[value["token_id_decimal"]].append(value)
        return result

    positives = positive_by_token(queried)
    unresolved = [
        row for row in manifest
        if sum(int(x["balance"]) for x in positives.get(row["token_id_decimal"], [])) != row["edition_supply"]
    ]

    fallback_errors: dict[str, str] = {}
    if unresolved:
        print(f"{len(unresolved)} tokens unresolved after current-holder discovery; loading transfer histories", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, min(WORKERS, 4))) as pool:
            futures = {pool.submit(transfer_candidates, row): row for row in unresolved}
            for completed, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                addresses, error = future.result()
                candidates[row["token_id_decimal"]].update(addresses)
                if error:
                    fallback_errors[row["token_id_decimal"]] = error
                if completed % 10 == 0 or completed == len(futures):
                    print(f"Transfer-history fallback: {completed}/{len(futures)}", flush=True)
        queried = query_balances(rpc_url, block_hex, make_pairs())
        positives = positive_by_token(queried)
        unresolved = [
            row for row in manifest
            if sum(int(x["balance"]) for x in positives.get(row["token_id_decimal"], [])) != row["edition_supply"]
        ]

    holdings: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for row in manifest:
        token_id = row["token_id_decimal"]
        token_positives = sorted(positives.get(token_id, []), key=lambda x: x["holder_address"])
        total = sum(int(value["balance"]) for value in token_positives)
        for value in token_positives:
            holder = value["holder_address"]
            holdings.append({
                "chain_id": CHAIN_ID,
                "block_number": block_number,
                "block_hash": block_hash,
                "block_timestamp_utc": block_timestamp_utc,
                "block_timestamp_montreal": block_timestamp_montreal,
                "contract": CONTRACT,
                "canonical_number": row["canonical_number"],
                "title": row["title"],
                "token_id_decimal": token_id,
                "token_id_hex": row["token_id_hex"],
                "holder_address": holder,
                "holder_ens": ens_by_address.get(holder, ""),
                "balance": int(value["balance"]),
                "verification_method": "ERC1155.balanceOfBatch eth_call at one finalized Ethereum block",
                "opensea_item_url": row["opensea_item_url"],
            })
        audit.append({
            "canonical_number": row["canonical_number"],
            "token_id_decimal": token_id,
            "expected_supply": row["edition_supply"],
            "positive_holder_rows": len(token_positives),
            "verified_balance_sum": total,
            "status": "PASS" if total == row["edition_supply"] else "FAIL",
            "candidate_addresses_queried": len(candidates[token_id]),
            "fallback_error": fallback_errors.get(token_id, ""),
        })

    wallet_agg: dict[str, dict[str, Any]] = defaultdict(lambda: {"distinct_tokens": 0, "total_units": 0, "canonical_numbers": []})
    for row in holdings:
        agg = wallet_agg[row["holder_address"]]
        agg["distinct_tokens"] += 1
        agg["total_units"] += int(row["balance"])
        agg["canonical_numbers"].append(int(row["canonical_number"]))
    wallet_rows = [
        {
            "holder_address": address,
            "holder_ens": ens_by_address.get(address, ""),
            "distinct_genesis_tokens": values["distinct_tokens"],
            "total_genesis_units": values["total_units"],
            "canonical_numbers": " ".join(str(n) for n in sorted(values["canonical_numbers"])),
        }
        for address, values in wallet_agg.items()
    ]
    wallet_rows.sort(key=lambda row: (-row["total_genesis_units"], row["holder_address"]))

    manifest_payload = {
        "schema": "montrealai.becoming.genesis-manifest.v2",
        "collection": COLLECTION,
        "chain_id": CHAIN_ID,
        "contract": CONTRACT,
        "token_standard": "ERC-1155",
        "canonical_work_count": EXPECTED_COUNT,
        "creator_encoded_in_token_ids": CREATOR,
        "construction": {
            "formula": "token_id = (uint160(creator) << 96) | (internal_nonce << 40) | edition_supply",
            "nonce_range_inclusive": [5, 563],
            "excluded_internal_nonces": sorted(EXCLUDED_NONCES),
            "canonical_order_exception": {"canonical_258_nonce": 264, "canonical_259_nonce": 263},
        },
        "records": manifest,
    }
    write_json(OUT / "manifest" / "montrealai_becoming_genesis_556_manifest.json", manifest_payload)
    write_csv(
        OUT / "manifest" / "montrealai_becoming_genesis_556_manifest.csv",
        manifest,
        [
            "canonical_number", "title", "chain_id", "contract", "standard",
            "token_id_decimal", "token_id_hex", "creator_encoded", "internal_nonce",
            "edition_supply", "opensea_item_url", "blockscout_item_url",
        ],
    )
    (OUT / "manifest" / "montrealai_becoming_genesis_556_token_ids.txt").write_text(
        "\n".join(row["token_id_decimal"] for row in manifest) + "\n", encoding="utf-8"
    )
    (OUT / "manifest" / "montrealai_becoming_genesis_556_opensea_urls.txt").write_text(
        "\n".join(row["opensea_item_url"] for row in manifest) + "\n", encoding="utf-8"
    )

    discovery_csv_rows = []
    for row in discoveries:
        discovery_csv_rows.append({
            **row,
            "candidate_addresses": " ".join(row.get("candidate_addresses") or []),
            "holder_values": json.dumps(row.get("holder_values") or {}, sort_keys=True),
            "ens_by_address": json.dumps(row.get("ens_by_address") or {}, sort_keys=True),
        })
    write_csv(
        OUT / "audit" / "blockscout_manifest_and_holder_discovery.csv",
        discovery_csv_rows,
        [
            "canonical_number", "token_id_decimal", "expected_title", "instance_status",
            "returned_id", "id_match", "returned_contract", "contract_match", "metadata_name",
            "metadata_number", "metadata_number_match", "is_unique", "blockscout_owner",
            "reported_holder_count", "reported_holder_balance_sum", "candidate_addresses",
            "holder_values", "ens_by_address", "instance_error", "holders_error",
            "instance_source_url", "holders_source_url",
        ],
    )
    write_csv(
        OUT / "snapshot" / "atomic_token_holdings.csv",
        holdings,
        [
            "chain_id", "block_number", "block_hash", "block_timestamp_utc", "block_timestamp_montreal",
            "contract", "canonical_number", "title", "token_id_decimal", "token_id_hex",
            "holder_address", "holder_ens", "balance", "verification_method", "opensea_item_url",
        ],
    )
    write_csv(
        OUT / "snapshot" / "atomic_wallet_summary.csv",
        wallet_rows,
        ["holder_address", "holder_ens", "distinct_genesis_tokens", "total_genesis_units", "canonical_numbers"],
    )
    write_csv(
        OUT / "snapshot" / "atomic_token_supply_audit.csv",
        audit,
        [
            "canonical_number", "token_id_decimal", "expected_supply", "positive_holder_rows",
            "verified_balance_sum", "status", "candidate_addresses_queried", "fallback_error",
        ],
    )
    write_csv(
        OUT / "audit" / "all_candidate_balances_at_snapshot_block.csv",
        sorted(queried, key=lambda x: (int(by_id[x["token_id_decimal"]]["canonical_number"]), x["holder_address"])),
        ["token_id_decimal", "holder_address", "balance"],
    )

    instance_found = sum(row["instance_status"] == "FOUND" for row in discoveries)
    exact_id_matches = sum(bool(row.get("id_match")) for row in discoveries)
    metadata_present = sum(bool(row.get("metadata_name")) for row in discoveries)
    metadata_number_matches = sum(row.get("metadata_number_match") is True for row in discoveries)
    tokens_passed = sum(row["status"] == "PASS" for row in audit)
    verified_units = sum(int(row["balance"]) for row in holdings)
    block_crosscheck_passes = sum(row.get("status") == "PASS" for row in rpc_crosschecks)
    complete = (
        len(manifest) == EXPECTED_COUNT
        and len({row["token_id_decimal"] for row in manifest}) == EXPECTED_COUNT
        and tokens_passed == EXPECTED_COUNT
        and verified_units == EXPECTED_COUNT
        and len(holdings) == EXPECTED_COUNT
    )

    block_payload = {
        "chain_id": CHAIN_ID,
        "block_tag": "finalized",
        "block_number": block_number,
        "block_number_hex": block_hex,
        "block_hash": block_hash,
        "block_timestamp_utc": block_timestamp_utc,
        "block_timestamp_montreal": block_timestamp_montreal,
        "contract": CONTRACT,
        "primary_rpc_endpoint": rpc_url,
        "rpc_selection_attempts": rpc_attempts,
        "independent_rpc_block_crosschecks": rpc_crosschecks,
    }
    write_json(OUT / "snapshot" / "atomic_snapshot_block.json", block_payload)

    status = {
        "schema": "montrealai.becoming.atomic-snapshot.v2",
        "collection": COLLECTION,
        "capture_started_utc": started,
        "capture_finished_utc": utc_now(),
        "chain_id": CHAIN_ID,
        "block_tag": "finalized",
        "block_number": block_number,
        "block_hash": block_hash,
        "block_timestamp_utc": block_timestamp_utc,
        "block_timestamp_montreal": block_timestamp_montreal,
        "contract": CONTRACT,
        "creator": CREATOR,
        "manifest_records": len(manifest),
        "unique_token_ids": len({row["token_id_decimal"] for row in manifest}),
        "blockscout_instances_found": instance_found,
        "blockscout_exact_id_matches": exact_id_matches,
        "blockscout_metadata_present": metadata_present,
        "blockscout_metadata_number_matches": metadata_number_matches,
        "candidate_address_token_pairs": len(queried),
        "positive_holder_rows": len(holdings),
        "wallets": len(wallet_rows),
        "tokens_passed": tokens_passed,
        "tokens_failed": EXPECTED_COUNT - tokens_passed,
        "verified_units": verified_units,
        "expected_units": EXPECTED_COUNT,
        "independent_rpc_block_crosscheck_passes": block_crosscheck_passes,
        "atomic_snapshot_complete": complete,
        "unresolved_tokens": [row for row in audit if row["status"] != "PASS"],
        "acceptance_rule": (
            "Every one of the 556 canonical ERC-1155 token IDs must have exactly one unit accounted for "
            "by direct balanceOfBatch calls at the same finalized Ethereum block."
        ),
    }
    write_json(OUT / "snapshot" / "atomic_snapshot_status.json", status)

    methodology = f"""# MONTREAL.AI — BECOMING: GENESIS 556
## Atomic ownership snapshot methodology

- Collection contract: `{CONTRACT}`
- Token standard: ERC-1155
- Canonical token IDs: 556
- Snapshot block: `{block_number}`
- Snapshot block hash: `{block_hash}`
- Block time (UTC): `{block_timestamp_utc}`
- Block time (Montréal): `{block_timestamp_montreal}`

### ID construction and verification

The 556 token IDs use OpenSea's legacy shared-storefront encoding. The creator address is encoded in the high 160 bits, the internal nonce occupies the next 56 bits, and the edition supply occupies the low 40 bits. The canonical sequence uses nonces 5 through 563, excludes nonces 161, 495, and 523, and preserves the verified creation-order exchange between canonical works 258 and 259.

Blockscout's public, keyless token-instance and token-instance-holder endpoints were queried for every ID. These responses supplied independent instance metadata and current candidate holder addresses. The creator address was always included as a candidate because the legacy storefront can represent untransferred lazy-minted inventory through its encoded balance rules.

### Atomic ownership rule

Marketplace or explorer observations were used only to discover candidate addresses. Ownership was accepted only after the legacy ERC-1155 contract's `balanceOfBatch` function was called at the single finalized Ethereum block recorded above. Every token had to reconcile to its expected supply of one. A missing candidate causes the audit to fail rather than assigning an owner by assumption.

### Completion result

- Tokens passed: {tokens_passed}/556
- Verified units: {verified_units}/556
- Positive holder rows: {len(holdings)}
- Distinct holder wallets: {len(wallet_rows)}
- Atomic snapshot complete: `{str(complete).lower()}`

The exact block number, block hash, manifest, token-level holdings, wallet summary, supply audit, discovery evidence, source code, and SHA-256 checksums are included in this package.
"""
    (OUT / "METHODOLOGY.md").write_text(methodology, encoding="utf-8")
    summary_text = (
        f"MONTREAL.AI — BECOMING: GENESIS 556\n"
        f"Atomic snapshot complete: {complete}\n"
        f"Finalized Ethereum block: {block_number}\n"
        f"Block hash: {block_hash}\n"
        f"Block timestamp UTC: {block_timestamp_utc}\n"
        f"Block timestamp Montréal: {block_timestamp_montreal}\n"
        f"Manifest IDs: {len(manifest)}\n"
        f"Tokens passed: {tokens_passed}\n"
        f"Verified units: {verified_units}\n"
        f"Distinct holder wallets: {len(wallet_rows)}\n"
    )
    (OUT / "SNAPSHOT_SUMMARY.txt").write_text(summary_text, encoding="utf-8")

    source_dir = OUT / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_dir.joinpath("becoming_snapshot_runner.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    checksum_files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "MONTREALAI_BECOMING_GENESIS_556_ATOMIC_SNAPSHOT.zip"})
    checksum_lines = [f"{sha256(path)}  {path.relative_to(OUT).as_posix()}" for path in checksum_files]
    (OUT / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    archive_path = OUT / "MONTREALAI_BECOMING_GENESIS_556_ATOMIC_SNAPSHOT.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p != archive_path):
            archive.write(path, path.relative_to(OUT))
    (OUT / "MONTREALAI_BECOMING_GENESIS_556_ATOMIC_SNAPSHOT.zip.sha256").write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )

    print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
    if not complete:
        print("Snapshot failed strict completion criteria.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
