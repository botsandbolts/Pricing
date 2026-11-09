"""Make-pure balanced shards and global audit reconstruction for EU listings.

Every normalized make is assigned to EXACTLY ONE shard. Full source row
indices survive sorting, partitioning, local linkage and global collection.
This module never writes training data back to the repository or to Streamlit.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pricing.europe_dedupe import norm

GLOBAL_ID = "__eu_global_row_id"
ALIASES = {
    "vw": "volkswagen",
    "mercedes": "mercedes benz",
    "mercedes benz": "mercedes benz",
    "mercedes benz cars": "mercedes benz",
    "mercedesbenz": "mercedes benz",
}


def canonical_make(value):
    name = norm(value)
    return ALIASES.get(name, name) or "__unknown__"


def balance_makes(counts, n_shards):
    """Deterministic largest-first assignment; never split a make."""
    if n_shards < 1:
        raise ValueError("shards must be >= 1")
    loads = [0] * n_shards
    allocations = {}
    for make, n in sorted(counts.items(), key=lambda t: (-t[1], t[0])):
        target = min(range(n_shards), key=lambda i: (loads[i], i))
        allocations[make] = target
        loads[target] += n
    return allocations, loads


def partition(input_file: Path, directory: Path, n_shards=12, batch_size=25000):
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Shard directory already exists and is nonempty")
    directory.mkdir(parents=True, exist_ok=True)
    source = pq.ParquetFile(str(input_file))
    if "make" not in source.schema_arrow.names:
        raise ValueError("Missing make column")
    if GLOBAL_ID in source.schema_arrow.names:
        raise ValueError("Input already contains reserved global ID column")
    counts = Counter()
    for batch in source.iter_batches(columns=["make"], batch_size=batch_size):
        counts.update(canonical_make(x) for x in batch.column(0).to_pylist())
    allocation, target_loads = balance_makes(counts, n_shards)
    schema = source.schema_arrow.append(pa.field(GLOBAL_ID,pa.int64()))
    writers = []
    for i in range(n_shards):
        writers.append(pq.ParquetWriter(
            str(directory / ("shard-%02d.parquet" % i)), schema,
            compression="zstd"))
    actual = [0]*n_shards
    offset=0
    try:
        for record_batch in source.iter_batches(batch_size=batch_size):
            batch = pa.Table.from_batches([record_batch])
            which = [[] for _ in range(n_shards)]
            for i, make in enumerate(batch.column("make").to_pylist()):
                which[allocation[canonical_make(make)]].append(i)
            for shard, positions in enumerate(which):
                if not positions:
                    continue
                data = batch.take(pa.array(positions, type=pa.int32()))
                global_ids = pa.array([offset+i for i in positions], type=pa.int64())
                data = data.append_column(GLOBAL_ID, global_ids)
                writers[shard].write_table(data)
                actual[shard] += len(positions)
            offset += batch.num_rows
    finally:
        for writer in writers:
            writer.close()
    if offset != sum(counts.values()) or actual != target_loads:
        raise AssertionError("Sharding lost or duplicated rows")
    report = {
        "source_rows":offset,
        "shard_count":n_shards,
        "unique_makes":len(counts),
        "shard_rows":actual,
        "makes_per_shard":[sum(s==i for s in allocation.values()) for i in range(n_shards)],
        "make_to_shard":allocation,
        "make_counts":dict(sorted(counts.items())),
        "balanced_by":"largest-first make frequency; never split normalized make",
        "global_id_column":GLOBAL_ID,
    }
    (directory/"manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def make_global_mapping(shard_file: Path, audit_dir: Path):
    """Translate local per-shard row indices to original source positions."""
    ids = pq.read_table(str(shard_file),columns=[GLOBAL_ID]).column(0).to_pylist()
    mapping = {}
    with (audit_dir/"row_mapping.csv").open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            local = int(row["row_number"])
            if local in mapping or local>=len(ids) or local<0:
                raise AssertionError("Invalid/duplicate local mapping index")
            mapping[local] = int(row["canonical_row_number"])
    thinning = {}
    path=audit_dir/"training_thinning_mapping.csv"
    if not path.is_file():
        raise ValueError("Missing optional hard-rule thinning mapping")
    with path.open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            local=int(row["raw_row"])
            if local in thinning or local>=len(ids) or local<0:
                raise AssertionError("Invalid/duplicate pruning index")
            thinning[local]=(int(row["training_representative"]),int(row["training_row_kept"]))
    if len(mapping)!=len(ids) or len(thinning)!=len(ids):
        raise AssertionError("Incomplete shard mapping")
    out=audit_dir/"global_mapping.csv"
    with out.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["global_row","global_identity_rep","global_training_rep","training_keep"])
        for local,gid in enumerate(ids):
            canon = mapping[local]
            rep, keep = thinning[local]
            if canon<0 or canon>=len(ids) or rep<0 or rep>=len(ids):
                raise AssertionError("Canonical/representative not in shard")
            writer.writerow([gid,ids[canon],ids[rep],keep])
    return {"rows":len(ids),"file":str(out)}


def collect(manifest_file: Path, root: Path, output: Path):
    """Validate each original global index appears once, then aggregate 12 jobs."""
    manifest=json.loads(manifest_file.read_text(encoding="utf-8"))
    n=int(manifest["source_rows"])
    seen=bytearray(n)
    original=0
    agg=Counter()
    countries=Counter()
    by_shard=[]
    by_brand=Counter()
    output.mkdir(parents=True,exist_ok=True)
    with (output/"global_training_mapping.csv").open("w",newline="",encoding="utf-8") as dest:
        wr=csv.writer(dest)
        wr.writerow(["global_row","global_identity_rep","global_training_rep","training_keep"])
        for i in range(manifest["shard_count"]):
            shard=root/("eu-make-audit-%02d" % i)
            stats_path=shard/"audit.json"
            path=shard/"global_mapping.csv"
            if not stats_path.is_file() or not path.is_file():
                raise ValueError("Missing report files for shard "+str(i))
            stats=json.loads(stats_path.read_text(encoding="utf-8"))
            if stats.get("cap_reached"):
                raise ValueError("Candidate cap reached in shard "+str(i))
            if stats.get("input_rows")!=manifest["shard_rows"][i]:
                raise ValueError("Row count mismatch in shard "+str(i))
            thin=stats.get("training_thinning")
            if not thin:
                raise ValueError("Missing thinning summary in shard "+str(i))
            for c,v in thin.get("excluded_by_country",{}).items():
                countries[c] += int(v)
            for key in ("input_rows","automatic_rows_removed","auto_pairs",
                        "review_pairs","rejected_candidate_pairs","total_candidates"):
                agg[key] += int(stats[key])
            agg["additional_rows_excluded"]+=int(thin["additional_rows_excluded"])
            by_shard.append({
                "shard":i,
                "input_rows":stats["input_rows"],
                "auto_merged_rows":stats["automatic_rows_removed"],
                "similarity_thinned_rows":thin["additional_rows_excluded"],
                "review_candidate_pairs":stats["review_pairs"],
            })
            with path.open(newline="",encoding="utf-8") as src:
                for row in csv.DictReader(src):
                    g=int(row["global_row"])
                    if g<0 or g>=n or seen[g]:
                        raise ValueError("Source row duplicate/out of range in shard "+str(i))
                    seen[g]=1
                    wr.writerow([g,row["global_identity_rep"],
                                 row["global_training_rep"],row["training_keep"]])
                    original += 1
    if original != n or seen.count(0)!=0 or agg["input_rows"]!=n:
        raise ValueError("Incomplete global coverage")
    after=n-agg["automatic_rows_removed"]-agg["additional_rows_excluded"]
    if after<0:
        raise AssertionError("Invalid output row counts")
    summary={
        "source_rows":n,
        "unique_makes":manifest["unique_makes"],
        "parallel_shards":manifest["shard_count"],
        "automatic_identity_rows_removed":agg["automatic_rows_removed"],
        "additional_similarity_thinned_rows":agg["additional_rows_excluded"],
        "training_rows_after":after,
        "percent_removed":round(100*(n-after)/n,4) if n else 0,
        "auto_pairs":agg["auto_pairs"],
        "review_pairs":agg["review_pairs"],
        "rejected_candidate_pairs":agg["rejected_candidate_pairs"],
        "total_candidate_pairs":agg["total_candidates"],
        "similarity_thinned_by_country":dict(sorted(countries.items())),
        "shards":by_shard,
        "physical_identity_verified":False,
        "split_rule":"all normalized make values belong to exactly one shard",
        "audit_complete":True,
        "safe_to_train":False,
        "reason":"Training ablation, licence validation and leakage checks are still required.",
    }
    (output/"combined_audit.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description="Partition and recombine European make-pure dedupe shards")
    subs=p.add_subparsers(dest="action",required=True)
    q=subs.add_parser("partition")
    q.add_argument("--input",required=True,type=Path)
    q.add_argument("--output",required=True,type=Path)
    q.add_argument("--shards",type=int,default=12)
    q=subs.add_parser("map")
    q.add_argument("--input",required=True,type=Path)
    q.add_argument("--audit",required=True,type=Path)
    q=subs.add_parser("collect")
    q.add_argument("--manifest",required=True,type=Path)
    q.add_argument("--root",required=True,type=Path)
    q.add_argument("--output",required=True,type=Path)
    args=p.parse_args(argv)
    if args.action=="partition":
        print(json.dumps(partition(args.input,args.output,args.shards),indent=2))
    elif args.action=="map":
        print(json.dumps(make_global_mapping(args.input,args.audit),indent=2))
    else:
        print(json.dumps(collect(args.manifest,args.root,args.output),indent=2))


if __name__=="__main__":
    main()
