"""Aggregate 200 independent validation-only neural experiments, never promote them."""
import argparse
import json
import math
from pathlib import Path

from scripts.neural_200_sweep import GROUPS,N_TRIALS


def assemble(folder):
    folder=Path(folder)
    rows={}
    problems=[]
    for path in sorted(folder.rglob("result.json")):
        try:
            r=json.loads(path.read_text())
            group=r["group"];trial=r["trial"]
            if group not in GROUPS or not isinstance(trial,int) or not 0<=trial<50:
                raise ValueError("Unknown trial identity")
            if r["status"]!="completed" or r.get("test_evaluated") is not False:
                raise ValueError("Result violates validation-only contract")
            v=r["validation"]
            required=("CVaR95_absolute_USD","RMSE_USD","MAE_USD","rows")
            if not all(k in v and math.isfinite(float(v[k])) for k in required):
                raise ValueError("Missing or non-finite held-out dollar risk")
            expected=31668 if group.startswith("ask") else 82380
            if v["rows"]!=expected or not r.get("provenance",{}).get("holdout_row_sha256"):
                raise ValueError("Holdout cohort size or fingerprint differs")
            key=(group,trial)
            if key in rows:raise ValueError("Duplicate result for "+str(key))
            rows[key]=r
        except (OSError,ValueError,TypeError,KeyError) as e:
            problems.append({"path":str(path),"error":str(e)})
    keys={(g,i) for g in GROUPS for i in range(N_TRIALS)}
    missing=sorted(keys-set(rows))
    groups={}
    for group in GROUPS:
        group_rows=[r for (g,_),r in rows.items() if g==group]
        group_rows.sort(key=lambda r:(r["validation"]["CVaR95_absolute_USD"],
                  r["validation"]["RMSE_USD"],r["trial"]))
        groups[group]=group_rows
        hashes={r["provenance"]["holdout_row_sha256"]["validation"]
                for r in group_rows}
        if len(hashes)>1:
            problems.append({"group":group,"error":"Mixed original validation row identities"})
    all_results=sorted(rows.values(),key=lambda r:(GROUPS.index(r["group"]),r["trial"]))
    complete=(len(rows)==200 and not missing and not problems)
    leaders={g:({"trial":r[0]["trial"],
                  "CVaR95_absolute_USD":r[0]["validation"]["CVaR95_absolute_USD"],
                  "RMSE_USD":r[0]["validation"]["RMSE_USD"],
                  "MAE_USD":r[0]["validation"]["MAE_USD"],
                  "config":r[0]["trial_configuration"]} if r else None)
             for g,r in groups.items()}
    return {
        "status":"complete" if complete else "partial_or_invalid",
        "planned_trials":200,"completed_valid_trials":len(rows),
        "counts":{g:len(groups[g]) for g in GROUPS},
        "missing":[{"group":g,"trial":i} for g,i in missing],
        "invalid":problems,
        "primary_metric":"original validation CVaR95 of absolute price error in USD, worst 5%",
        "secondary_metrics":["RMSE_USD","MAE_USD"],
        "evaluation_split":"original validation only; original test untouched",
        "source_periods":{"ask":"2021 asking listings","bid":"2014–2015 auctions, 2015 dollars"},
        "not_verified_on_november_2026_sales":True,
        "promotion":"NEVER automatic; a fresh original test assessment and inference parity are required",
        "group_winners":leaders,
        "experiments":all_results
    }


def publish(report,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=["# Neural experiment grid · CVaR95-first","",
        "**Status:** "+report["status"],
        f"**Trials:** {report['completed_valid_trials']}/200; "
        "50 each for Ask hyperparameters, Bid hyperparameters, Ask structure, Bid structure.",
        "",
        "Source-time validation only. No 2026-sale validation or automatic deployment.",
        "All 200 configurations are deterministic and original TEST stays unexamined.",
        "","| Group | Valid trials | Best trial | CVaR95 (USD) | RMSE (USD) | MAE (USD) |",
        "|---|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        r=report["group_winners"][group]
        n=report["counts"][group]
        if r:lines.append(f"| {group} | {n} | {r['trial']} | "
                          f"{r['CVaR95_absolute_USD']:,.0f} | "
                          f"{r['RMSE_USD']:,.0f} | {r['MAE_USD']:,.0f} |")
        else:lines.append(f"| {group} | 0 | — | — | — | — |")
    lines.extend(["","## Required next gate",
        "Choose challengers using original validation CVaR95; compare the chosen "
        "winner(s) on the untouched original test only once, then verify full "
        "native-checkpoint serving parity and current-date transport limitations.",
        "Do not call a candidate production-ready solely because it wins validation."])
    if report["missing"]:
        lines.extend(["","## Missing trials",str(report["missing"][:30])])
    if report["invalid"]:
        lines.extend(["","## Invalid results",str(report["invalid"][:20])])
    (output/"README.md").write_text("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    report=assemble(args.artifacts)
    publish(report,args.output)
    print("CVaR200_REPORT",report["status"],report["counts"],flush=True)
    if report["status"]!="complete":raise SystemExit(2)


if __name__=="__main__":main()
