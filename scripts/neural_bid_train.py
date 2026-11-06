"""Auction neural training: reuse exactly the tested 3 asking NN architectures.

3 architectures x 3 losses x 2 bid train cohorts. Inputs are audited historical
2015-normalized wholesale sales, without seller descriptions or hierarchy.
"""
import argparse
import json
from pathlib import Path
from scripts.neural_ask_train import ARCHS, LOSSES, COHORTS, run as train_shared

def run(args):
    evidence=json.loads(Path(args.data).with_suffix(".json").read_text())
    if evidence.get("market")!="bid" or evidence.get("no_description") is not True:
        raise ValueError("Requires authenticated 2015 auction bid features")
    train_shared(args)
    result=Path(args.output)/"result.json"
    report=json.loads(result.read_text())
    report.update({
      "market":"bid",
      "objective":"original historical 2015-normalized auction selling price (USD)",
      "target_kind":"reported wholesale sale, not live submitted bids",
      "text_features":"none; constant-zero compatibility dimension only",
      "outlier_policy":"full vs exact frozen Lemon training-only exclusion",
      "source_baseline":"fivefold-live/bid and xgboost-hierarchy/bid",
      "censoring":"no fabricated bounds; all realized sales observed exactly",
      "hierarchical_correction":False})
    result.write_text(json.dumps(report,indent=2,allow_nan=False))
    print("Completed verified auction arm",args.arch,args.loss,args.cohort,
          report["original_validation"]["mae_usd"],
          report["original_test"]["mae_usd"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch",choices=ARCHS,required=True)
    p.add_argument("--loss",choices=LOSSES,required=True)
    p.add_argument("--cohort",choices=COHORTS,required=True)
    p.add_argument("--data",default="training-cache/neural-bid/features.npz")
    p.add_argument("--output",required=True)
    p.add_argument("--epochs",type=int,default=35)
    p.add_argument("--patience",type=int,default=7)
    p.add_argument("--batch-size",type=int,default=768)
    run(p.parse_args())
