"""Strictly compare all 18 completed MiniLM hybrids with original neural and XGB."""
import argparse
import json
from pathlib import Path
import pandas as pd
from scripts.neural_ask_report import analyze
REV="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",default="training-cache/ask-minilm-results")
    p.add_argument("--reference",default="benchmark-reference/ask/predictions.csv.gz")
    p.add_argument("--original",default="reports/neural-ask-benchmark/comparison.json")
    p.add_argument("--full",default="models/fivefold-live/ask/evaluation.json")
    p.add_argument("--output",default="reports/ask-minilm-neural-benchmark")
    a=p.parse_args()
    files=sorted(Path(a.results).rglob("result.json"))
    if len(files)!=18:
        raise ValueError("Need all 18 independently completed MiniLM arms")
    for f in files:
        d=json.loads(f.read_text())
        if (d.get("feature_variant")!="minilm-2-oob-price" or
            d.get("minilm_2_oob") is not True or
            d.get("minilm_encoder_revision")!=REV or
            d.get("miniLM_original_text_svd_retained") is not True):
            raise ValueError("Unverified text feature provenance "+str(f))
    analyze(a.results,a.reference,a.full,a.output)
    dest=Path(a.output)/"comparison.json"
    current=json.loads(dest.read_text())
    before=json.loads(Path(a.original).read_text())
    if before.get("status")!="complete" or len(before.get("results",[]))!=36:
        raise ValueError("Original 18-arm benchmark missing")
    old={(r["architecture"],r["loss"],r["train_cohort"],r["partition"]):r for r in before["results"]}
    if len(old)!=36:
        raise ValueError("Duplicate original benchmark arm")
    for item in current["results"]:
        name=(item["architecture"],item["loss"],item["train_cohort"],item["partition"])
        b=old[name]
        if item["rows"]!=b["rows"]:
            raise ValueError("Neural original holdout count mismatch")
        item["original_neural_mae_usd"]=b["mae_usd"]
        item["delta_mae_vs_original_neural_usd"]=item["mae_usd"]-b["mae_usd"]
    current["benchmark"]="Original 18-arm neural ask vs frozen MiniLM two-OOB price hybrid"
    current["feature_set"]={
        "additional":"log of pretrained MiniLM text-only predicted price, scaled using FIT OOB rows",
        "encoder":"sentence-transformers/all-MiniLM-L6-v2",
        "revision":REV,
        "head_fitting":"three Ridge(alpha=100) heads each trained on one disjoint FIT-only fold",
        "neural_fit_rows":"average of two heads not fitted to own fold",
        "early_stopper_and_original_holdouts":"average of all three FIT-only heads",
        "early_stopper_and_holdout_price_leakage":False,
        "original_TFIDF_SVD_24_coordinates":"retained"}
    dest.write_text(json.dumps(current,indent=2,allow_nan=False))
    pd.DataFrame(current["results"]).to_csv(Path(a.output)/"comparison.csv",index=False)
    (Path(a.output)/"README.md").write_text(
      "# MiniLM two-out-of-bag neural ask benchmark\n\n"
      "All 18 arms audited against original chronological heldout records, "
      "the original 18 non-MiniLM neural arms and frozen Lemon XGBoost. "
      "The full-data XGBoost aggregate is descriptive only, not independently paired.\n")
    print("Validation-selected MiniLM winner:",current["winner_by_original_validation"],flush=True)

if __name__=="__main__":main()
