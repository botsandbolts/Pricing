"""Offline, lossless EU listing linkage. Never imported by the Streamlit app.

This tool proposes potential vehicle identities, NOT merely similar specifications.
A semantic neighbour is never sufficient for an automatic merge.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz

DATASET = "KTHESAT/european-used-car-market-2m-listings"
MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
FIELDS = ("rid", "site", "extid", "country", "make", "model", "version",
          "label_id", "reg_date", "reg_year", "km", "power", "city", "color",
          "fuel", "gear", "price")
SQL_ROW = "rid,site,extid,country,make,model,version,label_id,reg_date,reg_year,km,power,city,color,fuel,gear,price"
REQUIRED = {"id", "source_site", "seller_country", "make", "model",
            "version", "mileage_km", "first_registration", "reg_year",
            "power_hp", "seller_city", "price_eur"}


def norm(value):
    if value is None or str(value).lower() in ("nan", "none", "<na>"):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.casefold())).strip()


def integer(value):
    try:
        num = float(value)
        return int(num) if math.isfinite(num) else None
    except (ValueError, TypeError):
        return None


def floating(value):
    try:
        num = float(value)
        return num if math.isfinite(num) else None
    except (ValueError, TypeError):
        return None


def label_hash(label):
    return int.from_bytes(hashlib.blake2b(label.encode(), digest_size=8).digest(),
                          "big") & ((1 << 63) - 1)


def batches(path, size=50000):
    """Preserve row order, which is used in the reversible mapping."""
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(str(path)).iter_batches(batch_size=size):
            yield batch.to_pandas()
    elif path.suffix.lower() == ".csv":
        import pandas as pd
        yield from pd.read_csv(path, chunksize=size, low_memory=False)
    else:
        raise ValueError("Input must be .parquet or .csv")


def create_db(path):
    db = sqlite3.connect(str(path))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("""CREATE TABLE rows(
        rid INTEGER PRIMARY KEY, site TEXT, extid TEXT, country TEXT,
        make TEXT, model TEXT, version TEXT, label_id INTEGER,
        reg_date TEXT, reg_year INTEGER, km INTEGER, km_bin INTEGER,
        power INTEGER, city TEXT, color TEXT, fuel TEXT, gear TEXT,
        price REAL, first_seen TEXT)""")
    db.execute("""CREATE TABLE labels(
        label_id INTEGER PRIMARY KEY, make TEXT, model TEXT, label TEXT)""")
    db.execute("""CREATE TABLE pairs(
        a INTEGER NOT NULL, b INTEGER NOT NULL, origin TEXT NOT NULL,
        semantic REAL, PRIMARY KEY(a,b))""")
    return db


def ingest(db, path, sample=0):
    count = 0
    labels_seen = set()
    for frame in batches(path):
        missing = REQUIRED.difference(frame.columns)
        if missing:
            raise ValueError("Missing expected dataset fields: " + ", ".join(sorted(missing)))
        if sample and count >= sample:
            break
        if sample:
            frame = frame.iloc[:max(0, sample - count)]
        cols = frame.to_dict("records")
        rows, labels = [], []
        for item in cols:
            make, model, version = (norm(item.get(key)) for key in ("make","model","version"))
            site, extid = norm(item.get("source_site")), str(item.get("id") or "")
            if extid.lower() in ("nan", "none", "<na>"):
                extid = ""
            country = str(item.get("seller_country") or "").upper().strip()
            city = norm(item.get("seller_city"))
            year = integer(item.get("reg_year"))
            km, power = integer(item.get("mileage_km")), integer(item.get("power_hp"))
            if km is not None and km < 0:
                km = None
            if power is not None and power <= 0:
                power = None
            registration = str(item.get("first_registration") or "")[:10]
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", registration):
                registration = ""
            label = " ".join(x for x in (make, model, version) if x)
            lid = label_hash(label)
            if lid not in labels_seen:
                labels.append((lid, make, model, label))
                labels_seen.add(lid)
            rows.append((count, site, extid, country, make, model, version,
                         lid, registration, year, km, km // 1000 if km is not None else None,
                         power, city, norm(item.get("color")), norm(item.get("fuel")),
                         norm(item.get("transmission")), floating(item.get("price_eur")),
                         str(item.get("first_seen_at") or "")))
            count += 1
        db.executemany("INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        db.executemany("INSERT OR IGNORE INTO labels VALUES (?,?,?,?)", labels)
        db.commit()
    db.executescript("""
        CREATE INDEX idx_src ON rows(site,extid);
        CREATE INDEX idx_block ON rows(country,make,model,reg_year,city,km_bin);
        CREATE INDEX idx_date ON rows(country,make,model,city,reg_date);
        CREATE INDEX idx_label ON rows(label_id,country,city,reg_year,km_bin);
    """)
    db.commit()
    return count


def pair_count(db):
    return db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]


def insert_candidates(db, max_pairs):
    """Finite, independently useful blocking rules. No O(N^2) search."""
    queries = [
        ("""SELECT a.rid,b.rid,'source-id',NULL FROM rows a JOIN rows b
              ON a.site=b.site AND a.extid=b.extid AND a.rid<b.rid
              WHERE a.site<>'' AND a.extid<>'' AND a.make=b.make AND a.model=b.model""", "source-id"),
        ("""SELECT a.rid,b.rid,'structured',NULL FROM rows a JOIN rows b
              ON a.country=b.country AND a.make=b.make AND a.model=b.model
              AND a.reg_year=b.reg_year AND a.city=b.city
              AND b.km_bin BETWEEN a.km_bin-1 AND a.km_bin+1 AND a.rid<b.rid
              WHERE a.country<>'' AND a.make<>'' AND a.model<>''
              AND a.city<>'' AND a.reg_year BETWEEN 1950 AND 2027
              AND a.km IS NOT NULL AND b.km IS NOT NULL
              AND ABS(a.km-b.km)<=1300
              AND (a.power IS NULL OR b.power IS NULL OR ABS(a.power-b.power)<=5)""", "structured"),
        ("""SELECT a.rid,b.rid,'registration',NULL FROM rows a JOIN rows b
              ON a.country=b.country AND a.make=b.make AND a.model=b.model
              AND a.city=b.city AND a.reg_date=b.reg_date AND a.rid<b.rid
              WHERE a.country<>'' AND a.make<>'' AND a.model<>'' AND a.city<>''
              AND a.reg_date<>'' AND a.km IS NOT NULL AND b.km IS NOT NULL
              AND ABS(a.km-b.km)<=3000
              AND (a.power IS NULL OR b.power IS NULL OR ABS(a.power-b.power)<=5)""", "registration"),
    ]
    stats = {}
    for query, name in queries:
        remaining = max_pairs - pair_count(db)
        if remaining <= 0:
            stats[name] = "skipped: candidate cap"
            continue
        db.execute("INSERT OR IGNORE INTO pairs(a,b,origin,semantic) " + query + " LIMIT ?", (remaining,))
        db.commit()
        stats[name] = pair_count(db)
    return stats


def add_granite_faiss(db, max_pairs, cache, top_k=10, min_sim=0.88):
    """Unique labels are encoded once; FAISS HNSW retrieves cross-language variants.

    ANN candidates still require same country, city, model, year and near odometer.
    No semantic match can by itself authorize a merge.
    """
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    labels = db.execute("SELECT label_id,make,model,label FROM labels ORDER BY label_id").fetchall()
    ids = [t[0] for t in labels]
    digest = hashlib.sha256(("\n".join(str(v) + ":" + s for v,_,_,s in labels) + MODEL).encode()).hexdigest()
    cache_path = Path(cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        saved = np.load(cache_path, allow_pickle=False)
        if str(saved["digest"].item()) != digest:
            raise ValueError("Embedding cache belongs to a different dataset/model")
        vectors = saved["vectors"]
    else:
        encoder = SentenceTransformer(MODEL)
        vectors = encoder.encode([t[3] for t in labels], batch_size=64,
                                 normalize_embeddings=True, show_progress_bar=True,
                                 convert_to_numpy=True).astype("float32")
        np.savez_compressed(cache_path, digest=np.array(digest), vectors=vectors)
        del encoder
    if not len(vectors):
        return {"labels":0,"semantic_candidates":0}
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexHNSWFlat(vectors.shape[1], 24, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 80
    index.add(vectors)
    db.execute("CREATE TABLE label_neighbors(l INTEGER,r INTEGER,score REAL, PRIMARY KEY(l,r))")
    db.execute("CREATE INDEX idx_neighbor_r ON label_neighbors(r)")
    for start in range(0, len(vectors), 1024):
        dist, idx = index.search(vectors[start:start+1024], top_k+1)
        entries = []
        for j in range(len(idx)):
            current = start+j
            for neighbor, similarity in zip(idx[j], dist[j]):
                if neighbor < 0 or neighbor == current or similarity < min_sim:
                    continue
                if labels[current][1:3] != labels[neighbor][1:3]:
                    continue
                entries.append((ids[current],ids[neighbor],float(similarity)))
        db.executemany("INSERT OR REPLACE INTO label_neighbors VALUES(?,?,?)", entries)
        db.commit()
    remaining = max_pairs - pair_count(db)
    if remaining > 0:
        sql = """SELECT a.rid,b.rid,'granite-hnsw',n.score
                 FROM label_neighbors n
                 JOIN rows a ON a.label_id=n.l
                 JOIN rows b ON b.label_id=n.r
                 WHERE a.rid<b.rid AND a.country=b.country AND a.city=b.city
                 AND a.make=b.make AND a.model=b.model
                 AND a.reg_year=b.reg_year AND a.country<>'' AND a.city<>''
                 AND a.km IS NOT NULL AND b.km IS NOT NULL
                 AND ABS(a.km-b.km)<=1300
                 AND (a.power IS NULL OR b.power IS NULL OR ABS(a.power-b.power)<=5)"""
        db.execute("INSERT OR IGNORE INTO pairs(a,b,origin,semantic) " + sql + " LIMIT ?", (remaining,))
        db.commit()
    return {"labels":len(labels),"semantic_candidates":pair_count(db),
            "granite_model":MODEL,"cache":str(cache_path)}


def source_identity(a,b):
    return (bool(a["site"] and a["extid"]) and a["site"] == b["site"]
            and a["extid"] == b["extid"])


def compatible(a,b):
    """Hard contradiction filter. Unknown means unknown, not disagreement."""
    if not a["make"] or not a["model"] or a["make"] != b["make"] or a["model"] != b["model"]:
        return False
    if a["reg_year"] and b["reg_year"] and a["reg_year"] != b["reg_year"]:
        return False
    if a["power"] and b["power"] and abs(a["power"]-b["power"]) > 5:
        return False
    if a["km"] is not None and b["km"] is not None and abs(a["km"]-b["km"]) > 3000:
        return False
    if a["fuel"] and b["fuel"] and a["fuel"] != b["fuel"]:
        return False
    if a["gear"] and b["gear"] and a["gear"] != b["gear"]:
        return False
    return True


def classify(a,b,semantic=None):
    if not compatible(a,b):
        return "reject", 0.0, "conflicting vehicle identity"
    if source_identity(a,b):
        return "auto", 1.0, "same source listing ID"
    name_score = max(fuzz.ratio(a["version"],b["version"])/100.0 if a["version"] and b["version"] else 0.0,
                     float(semantic) if semantic is not None else 0.0)
    city_match = bool(a["city"] and a["city"] == b["city"])
    date_match = bool(a["reg_date"] and a["reg_date"] == b["reg_date"])
    km_delta = abs(a["km"]-b["km"]) if a["km"] is not None and b["km"] is not None else 1e12
    power_match = bool(a["power"] and a["power"] == b["power"])
    same_country = bool(a["country"] and a["country"] == b["country"])
    # Name similarity is NOT identity. Deliberately require extraordinarily
    # distinctive coincidences and independent marketplaces for automatic merge.
    if (a["site"] != b["site"] and same_country and city_match and date_match
        and km_delta <= 25 and power_match
        and a["version"] == b["version"] and len(a["version"]) >= 8
        and (not a["color"] or not b["color"] or a["color"] == b["color"])
        and (not a["fuel"] or not b["fuel"] or a["fuel"] == b["fuel"])
        and (not a["gear"] or not b["gear"] or a["gear"] == b["gear"])):
        return "auto", 0.99, "cross-market strong identifying agreement"
    if (same_country and city_match and a["reg_year"] == b["reg_year"]
        and km_delta <= 1300 and (a["power"] is None or b["power"] is None or abs(a["power"]-b["power"]) <= 5)
        and name_score >= 0.82):
        return "review", round(name_score,4), "plausible cross-post or similar vehicle"
    return "reject", round(name_score,4), "insufficient identifying evidence"


def audit(db, output, rows_total, max_pairs, method):
    output.mkdir(parents=True, exist_ok=True)
    parent = {}
    members = {}
    def find(x):
        parent.setdefault(x,x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    @lru_cache(maxsize=100000)
    def record(rid):
        values = db.execute("SELECT " + SQL_ROW + " FROM rows WHERE rid=?", (rid,)).fetchone()
        return dict(zip(FIELDS,values))

    def safe_join(a,b):
        x,y = find(a),find(b)
        if x == y:
            return True
        left,right = members.get(x,[x]),members.get(y,[y])
        if len(left)+len(right) > 8:
            return False
        for i in left:
            for j in right:
                ai,bj = record(i),record(j)
                if not compatible(ai,bj):
                    return False
                if ai["site"] == bj["site"] and ai["extid"] != bj["extid"]:
                    return False
                if ai["site"] != bj["site"] and ai["km"] is not None and bj["km"] is not None and abs(ai["km"]-bj["km"])>25:
                    return False
                if ai["site"] != bj["site"] and ai["reg_date"] and bj["reg_date"] and ai["reg_date"] != bj["reg_date"]:
                    return False
        parent[y]=x
        members[x]=left+right
        members.pop(y,None)
        return True

    # Fetch matches with tight SQLite cursor buffering and deterministic order.
    cursor = db.execute("""SELECT a,b,origin,semantic FROM pairs
                            ORDER BY CASE origin WHEN 'source-id' THEN 0 ELSE 1 END,a,b""")
    counts = Counter()
    with (output/"automatic_pairs.csv").open("w",newline="",encoding="utf-8") as auto_file, \
         (output/"review_pairs.csv").open("w",newline="",encoding="utf-8") as review_file:
        aw,rw = csv.writer(auto_file),csv.writer(review_file)
        header = ["row_a","row_b","source_a","listing_id_a","source_b","listing_id_b",
                  "score","origin","reason"]
        aw.writerow(header);rw.writerow(header)
        for a,b,origin,sim in cursor:
            one,two=record(a),record(b)
            verdict,score,reason=classify(one,two,sim)
            if verdict == "auto" and not safe_join(a,b):
                verdict,reason="review","cluster coherence check failed"
            counts[verdict]+=1
            line=[a,b,one["site"],one["extid"],two["site"],two["extid"],score,origin,reason]
            if verdict=="auto":aw.writerow(line)
            elif verdict=="review":rw.writerow(line)
    keep = set()
    with (output/"row_mapping.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f)
        wr.writerow(["row_number","canonical_row_number","source_site","listing_id",
                     "seller_country","auto_cluster_size"])
        for batch_start in range(0,rows_total,50000):
            for rid,site,extid,country in db.execute(
                "SELECT rid,site,extid,country FROM rows WHERE rid>=? AND rid<? ORDER BY rid",
                (batch_start,batch_start+50000)):
                root=find(rid) if rid in parent else rid
                canonical=min(members.get(root,[root]))
                keep.add(canonical)
                wr.writerow([rid,canonical,site,extid,country,len(members.get(root,[root]))])
    summary = dict(dataset=DATASET, input_rows=rows_total,
                   canonical_rows=len(keep), automatic_rows_removed=rows_total-len(keep),
                   auto_pairs=counts["auto"], review_pairs=counts["review"],
                   rejected_candidate_pairs=counts["reject"],
                   total_candidates=pair_count(db), candidate_cap=max_pairs,
                   cap_reached=(pair_count(db)>=max_pairs),
                   method=method, safe_to_train=False,
                   reason="Linkage is unaudited; inspect uncertain pairs and provenance before training.")
    (output/"audit.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return keep,summary


def export_canonical(path, output, keep, stem="canonical_listings"):
    """Keep all original columns; never edit raw listings."""
    if path.suffix.lower()==".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        src=pq.ParquetFile(str(path))
        writer=None;offset=0
        try:
            for batch in src.iter_batches(batch_size=50000):
                mask=pa.array([i in keep for i in range(offset,offset+len(batch))])
                sub=pa.Table.from_batches([batch]).filter(mask)
                if writer is None:
                    writer=pq.ParquetWriter(str(output/(stem+".parquet")),sub.schema,
                                            compression="zstd")
                writer.write_table(sub)
                offset+=len(batch)
        finally:
            if writer is not None:writer.close()
    else:
        first=True;offset=0
        for frame in batches(path):
            select=[i in keep for i in range(offset,offset+len(frame))]
            frame.loc[select].to_csv(output/(stem+".csv"), mode="w" if first else "a",
                                     header=first,index=False)
            first=False;offset+=len(frame)


def main(argv=None):
    parser=argparse.ArgumentParser(description="Lossless European listings record linkage")
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--sample",type=int,default=0,help="First N rows; test only")
    parser.add_argument("--max-pairs",type=int,default=5000000)
    parser.add_argument("--granite",action="store_true",help="Use IBM Granite R2 + FAISS HNSW")
    parser.add_argument("--embedding-cache",type=Path)
    parser.add_argument("--no-canonical-export",action="store_true")
    parser.add_argument("--thin-review-candidates",action="store_true",
                        help="Also create a distinct high-information training subsample")
    args=parser.parse_args(argv)
    if args.sample < 0 or args.max_pairs < 1:
        parser.error("Invalid sample/max-pairs")
    if not args.input.is_file():
        parser.error("Input file does not exist")
    args.output.mkdir(parents=True,exist_ok=True)
    db_path=args.output/"linkage.sqlite"
    if db_path.exists():
        parser.error("Output already contains linkage.sqlite; use a fresh directory")
    db=create_db(db_path)
    try:
        count=ingest(db,args.input,args.sample)
        blocks=insert_candidates(db,args.max_pairs)
        method={"blocking":blocks,"granite":False}
        if args.granite:
            method["granite"]=add_granite_faiss(db,args.max_pairs,
                args.embedding_cache or (args.output/"granite_labels.npz"))
        keep,report=audit(db,args.output,count,args.max_pairs,method)
        thin_keep=None
        if args.thin_review_candidates:
            from pricing.europe_prune import thin_review_graph
            thin_keep=thin_review_graph(db,args.output,keep,count,report)
        if not args.no_canonical_export and not args.sample:
            export_canonical(args.input,args.output,keep)
            if thin_keep is not None:
                export_canonical(args.input,args.output,thin_keep,stem="training_thinned_listings")
        print(json.dumps(report,indent=2))
    finally:
        db.close()


if __name__=="__main__":
    main()
