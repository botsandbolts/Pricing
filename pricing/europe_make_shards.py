"""Balanced 50-way make + production/model-year shards for European listings.

Candidate comparisons MUST stay inside exactly the same normalized make and
manufacturing/model year (falling back to registration year when necessary).
Every global source ID is tracked for reversible final aggregation.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pricing.europe_dedupe import norm

GLOBAL_ID = "__eu_global_row_id"
SEGMENT = "__eu_segment_key"
ALIASES = {
    "vw": "volkswagen",
    "mercedes": "mercedes benz",
    "mercedes benz": "mercedes benz",
    "mercedes benz cars": "mercedes benz",
    "mercedesbenz": "mercedes benz",
}
YEAR_COLUMNS = (
    "production_year", "manufacturing_year", "manufacture_year",
    "production_date", "model_year", "year", "reg_year",
    "first_registration",
)


def canonical_make(value):
    name = norm(value)
    return ALIASES.get(name, name) or "__unknown__"


def canonical_year(value):
    """Four-digit production/model-year normalization, avoiding false years."""
    if value is None:
        return "unknown"
    match = re.match(r"^\s*(19\d\d|20\d\d)(?:\D|$)", str(value))
    if not match:
        return "unknown"
    year=int(match.group(1))
    return str(year) if 1950 <= year <= 2027 else "unknown"


def segment(make, year):
    return canonical_make(make) + "::year::" + str(year)


def balance_makes(counts, n_shards):
    """Largest-first group assignment; groups, not brands, are indivisible."""
    if n_shards < 1:
        raise ValueError("shards must be >= 1")
    loads = [0] * n_shards
    allocations = {}
    for group, n in sorted(counts.items(), key=lambda t: (-t[1], t[0])):
        target = min(range(n_shards), key=lambda i: (loads[i], i))
        allocations[group] = target
        loads[target] += n
    return allocations, loads


def _groups(batch, year_columns):
    make_values=batch.column("make").to_pylist()
    years={c:batch.column(c).to_pylist() for c in year_columns}
    selected=[]
    for i,make in enumerate(make_values):
        year="unknown"
        origin="missing"
        for field in year_columns:
            result=canonical_year(years[field][i])
            if result!="unknown":
                year=result
                origin=field
                break
        selected.append((segment(make,year),origin))
    return selected


def partition(input_file: Path, directory: Path, n_shards=50, batch_size=25000):
    """Two streaming passes: count groups, then write 50 balanced data shards."""
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Shard directory already exists and is nonempty")
    directory.mkdir(parents=True, exist_ok=True)
    source = pq.ParquetFile(str(input_file))
    columns=source.schema_arrow.names
    if "make" not in columns:
        raise ValueError("Missing make column")
    if GLOBAL_ID in columns or SEGMENT in columns:
        raise ValueError("Input already contains a reserved partition column")
    year_columns=[x for x in YEAR_COLUMNS if x in columns]
    if not year_columns:
        raise ValueError("Missing production/model/registration year columns")
    counts=Counter()
    provenance=Counter()
    for batch in source.iter_batches(columns=["make"]+year_columns,batch_size=batch_size):
        for group,origin in _groups(pa.Table.from_batches([batch]),year_columns):
            counts[group]+=1
            provenance[origin]+=1
    allocation,loads=balance_makes(counts,n_shards)
    schema=source.schema_arrow.append(pa.field(GLOBAL_ID,pa.int64()))
    schema=schema.append(pa.field(SEGMENT,pa.string()))
    writers=[
        pq.ParquetWriter(str(directory/("shard-%02d.parquet"%i)),
                         schema,compression="zstd")
        for i in range(n_shards)
    ]
    actual=[0]*n_shards
    offset=0
    try:
        for record_batch in source.iter_batches(batch_size=batch_size):
            batch=pa.Table.from_batches([record_batch])
            groups=[v[0] for v in _groups(batch,year_columns)]
            which=[[] for _ in range(n_shards)]
            for i,key in enumerate(groups):
                which[allocation[key]].append(i)
            for shard,positions in enumerate(which):
                if not positions:
                    continue
                part=batch.take(pa.array(positions,type=pa.int32()))
                ids=pa.array([offset+i for i in positions],type=pa.int64())
                keys=pa.array([groups[i] for i in positions],type=pa.string())
                part=part.append_column(GLOBAL_ID,ids).append_column(SEGMENT,keys)
                writers[shard].write_table(part)
                actual[shard]+=len(positions)
            offset+=batch.num_rows
    finally:
        for writer in writers:
            writer.close()
    if sum(counts.values())!=offset or actual!=loads:
        raise AssertionError("Partition lost rows or misassigned make-year groups")
    unique_makes={k.split("::year::",1)[0] for k in counts}
    year_sources={field:int(provenance[field]) for field in sorted(provenance)}
    report={
        "source_rows":offset,
        "shard_count":n_shards,
        "unique_makes":len(unique_makes),
        "unique_make_year_segments":len(counts),
        "shard_rows":actual,
        "segments_per_shard":[sum(v==i for v in allocation.values()) for i in range(n_shards)],
        "segment_to_shard":dict(sorted(allocation.items())),
        "segment_counts":dict(sorted(counts.items())),
        "year_sources":year_sources,
        "year_precedence":year_columns,
        "year_definition":"production/model year if available, registration year as clearly labelled fallback",
        "balanced_by":"largest-first normalized make + year group",
        "global_id_column":GLOBAL_ID,
        "segment_column":SEGMENT,
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
    """Validate exactly-once coverage across all 50 make/year shard audits."""
    manifest=json.loads(manifest_file.read_text(encoding="utf-8"))
    n=int(manifest["source_rows"])
    seen=bytearray(n)
    original=0
    agg=Counter()
    countries=Counter()
    by_shard=[]
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
        "unique_make_year_segments":manifest["unique_make_year_segments"],
        "year_sources":manifest["year_sources"],
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
        "split_rule":"matching restricted to exactly equal normalized make and production/model/registration year segments",
        "audit_complete":True,
        "safe_to_train":False,
        "reason":"Training ablation, licence validation and leakage checks are still required.",
    }
    (output/"combined_audit.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description="Partition and recombine European make-year-blocked dedupe shards")
    subs=p.add_subparsers(dest="action",required=True)
    q=subs.add_parser("partition")
    q.add_argument("--input",required=True,type=Path)
    q.add_argument("--output",required=True,type=Path)
    q.add_argument("--shards",type=int,default=50)
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
