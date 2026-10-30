"""Generate CatBoost residuals and fit the hierarchy in one offline command."""
import argparse,json
from pathlib import Path
import pandas as pd
from pricing.catboost_residuals import generate
from pricing.hierarchical import experiment

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--observations',required=True)
    p.add_argument('--side',choices=['ask','bid'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();d,provenance=generate(pd.read_csv(a.observations),a.side)
    bundle,report,_=experiment(d,a.side);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/'catboost-residuals.csv',index=False)
    (out/'model.json').write_text(json.dumps(bundle,indent=2))
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    (out/'residual-provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(report,indent=2))
