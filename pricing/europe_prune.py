"""Optional reversible near-duplicate thinning for European price training.

Deliberately not a vehicle-identity resolution engine. Review candidates are
similar vehicles; removing one is sampling, not proof of the same physical car.
"""
from __future__ import annotations
import csv
import json
import sqlite3
from collections import Counter
from functools import lru_cache


def information_score(row):
    """Only documented informativeness matters; asking price *size* never votes."""
    score = 0.0
    score += 5 if row["reg_date"] else 0
    score += 4 if row["km"] is not None else 0
    score += 3 if row["power"] is not None else 0
    score += 2 if row["city"] else 0
    score += 2 if row["version"] else 0
    score += min(len(row["version"].split()), 8) * 0.2
    score += 1 if row["fuel"] else 0
    score += 1 if row["gear"] else 0
    score += 0.75 if row["color"] else 0
    score += 0.5 if row["price"] is not None and row["price"] > 0 else 0
    score += 0.25 if row["extid"] else 0
    return round(score,4)


def thin_review_graph(db: sqlite3.Connection, output, canonical_keep: set[int],
                      total: int, report: dict) -> set[int]:
    from pricing.europe_dedupe import FIELDS, SQL_ROW
    edges=[]
    touched=set()
    with (output/"review_pairs.csv").open(newline="",encoding="utf-8") as f:
        for line in csv.DictReader(f):
            a,b=int(line["row_a"]),int(line["row_b"])
            edges.append((a,b))
            touched.update((a,b))
            if len(edges)>2000000:
                raise RuntimeError("More than 2M review edges: shard by market before thinning")
    canonical={}
    with (output/"row_mapping.csv").open(newline="",encoding="utf-8") as f:
        for item in csv.DictReader(f):
            rid=int(item["row_number"])
            if rid in touched:
                canonical[rid]=int(item["canonical_row_number"])
    graph={}
    for a,b in edges:
        a,b=canonical[a],canonical[b]
        if a==b:
            continue
        graph.setdefault(a,set()).add(b)
        graph.setdefault(b,set()).add(a)
    del edges,touched,canonical

    @lru_cache(maxsize=200000)
    def record(rid):
        values=db.execute("SELECT "+SQL_ROW+" FROM rows WHERE rid=?",(rid,)).fetchone()
        return dict(zip(FIELDS,values))
    scores={rid:information_score(record(rid)) for rid in graph}
    retained=set()
    removed={}
    # Greedy maximal independent set. Higher-quality rows are considered first,
    # then stable row ID breaks ties. Every review edge loses at least one end.
    for rid in sorted(graph,key=lambda k:(-scores[k],k)):
        if rid in removed:
            continue
        retained.add(rid)
        for neighbour in sorted(graph[rid]):
            if neighbour not in retained and neighbour not in removed:
                removed[neighbour]=rid
    if any(a in retained and b in retained for a,adj in graph.items() for b in adj):
        raise AssertionError("Uncovered near-duplicate edge")
    if not set(removed).issubset(canonical_keep):
        raise AssertionError("Removed noncanonical source record")
    choices=canonical_keep.difference(removed)
    by_country=Counter(record(r)["country"] for r in removed)
    with (output/"training_thinning_decisions.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["excluded_canonical_row","preferred_canonical_row",
                         "excluded_information_score","preferred_information_score",
                         "status"])
        for r in sorted(removed):
            k=removed[r]
            writer.writerow([r,k,scores[r],scores[k],"similarity_thinned_not_proven_duplicate"])
    with (output/"training_thinning_mapping.csv").open("w",newline="",encoding="utf-8") as f, \
         (output/"row_mapping.csv").open(newline="",encoding="utf-8") as inp:
        writer=csv.writer(f)
        writer.writerow(["raw_row","identity_canonical_row","training_representative",
                         "training_row_kept","reason"])
        for item in csv.DictReader(inp):
            raw=int(item["row_number"])
            canon=int(item["canonical_row_number"])
            representative=removed.get(canon,canon)
            writer.writerow([raw,canon,representative,
                             int(raw==canon and canon not in removed),
                             "similarity_thinned" if canon in removed else
                             "identity_collapsed" if raw!=canon else "kept"])
    report["training_thinning"]={
        "policy":"greedy_high_information_maximal_independent_set",
        "interpretation":"Similar listing sampling; NOT proven duplicate vehicles",
        "candidate_edges":sum(len(v) for v in graph.values())//2,
        "candidate_vertices":len(graph),
        "canonical_rows_before":len(canonical_keep),
        "training_rows_after":len(choices),
        "additional_rows_excluded":len(removed),
        "additional_pct_of_raw":round(100*len(removed)/total,4) if total else 0,
        "excluded_by_country":dict(sorted(by_country.items())),
        "verified_duplicates":False,
        "model_ablation_required":True
    }
    (output/"audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return choices
