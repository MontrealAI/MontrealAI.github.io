#!/usr/bin/env python3
import hashlib, json, re
from pathlib import Path
import becoming_omega_snapshot_v3 as s


def manifest_sequence(title, image=None, metadata_url=None):
    m=re.search(r"#\s*0*(\d{1,4})(?:\D|$)",str(title or ""),re.I)
    if not m:return None
    declared=int(m.group(1))
    if not 0<=declared<=555:return None
    if declared==316:
        tm=re.search(r"/(\d+)(?:\?.*)?$",str(metadata_url or ""))
        if tm:
            token_id=int(tm.group(1)); nonce=(((token_id>>32)&((1<<64)-1))>>8)
            if nonce==322:return 318
    return declared+1


def postprocess():
    out=s.OUT
    anomaly={
        "schema":"montrealai.becoming.title-numbering-anomalies.v1",
        "principle":"Original onchain/OpenSea titles are preserved verbatim. canonical_number is a separate deterministic 1–556 manifest sequence ordered by the encoded mint nonce.",
        "declared_title_range":"Crypto AI Art #000 through Crypto AI Art #555",
        "anomalies":[{
            "canonical_number":318,
            "declared_title":"Crypto AI Art #316",
            "declared_title_number":316,
            "encoded_nonce":322,
            "finding":"This is the second distinct token titled #316. No token is titled #317. Its immutable manifest sequence position is 318 (corresponding to zero-based sequence position 317).",
            "token_id_decimal":"2392630434290240917728431095880785304289144848761899072947382790124747096065"
        }],
        "absent_declared_title_numbers":[317],
        "duplicated_declared_title_numbers":[316],
        "item_count":556
    }
    s.dump("title_numbering_anomalies.json",anomaly)
    audit_path=out/"snapshot_audit_report.json"; audit=json.loads(audit_path.read_text())
    audit["title_numbering"]={"declared_range":"000–555","duplicate_declared_number":316,"absent_declared_number":317,"manifest_sequence":"canonical_number 1–556, deterministic by encoded mint nonce","status":"PRESERVED_AND_DISCLOSED"}
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    meta_path=out/"snapshot_metadata.json"; meta=json.loads(meta_path.read_text())
    meta["title_numbering_note"]="The 556 immutable titles run #000–#555, with two distinct #316 tokens and no #317. canonical_number is the separate 1–556 manifest sequence."
    meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    meth=out/"snapshot_methodology.md"
    meth.write_text(meth.read_text()+"\n## Original title-numbering anomaly\n\nThe 556 immutable works are titled #000 through #555. Two distinct tokens are titled #316 and no token is titled #317. No historical title was altered. The package therefore uses a separate deterministic `canonical_number` from 1 through 556, ordered by the token ID's encoded mint nonce, and records the anomaly in `title_numbering_anomalies.json`.\n",encoding="utf-8")
    files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="SHA256SUMS")
    (out/"SHA256SUMS").write_text("\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in files)+"\n",encoding="utf-8")


if __name__=="__main__":
    s.number=manifest_sequence
    try:
        s.main(); postprocess()
    except Exception as e:
        s.dump("run_failure.json",{"status":"FAIL","at_utc":s.now(),"error":repr(e)})
        raise
