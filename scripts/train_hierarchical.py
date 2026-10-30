"""Fit an offline residual bundle; never predicts with a deployment refit on its training rows."""
import argparse,json
from pathlib import Path
import pandas as pd
from pricing.hierarchical import experiment

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--residuals',required=True)
    p.add_argument('--side',choices=['ask','bid'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();b,r,_=experiment(pd.read_csv(a.residuals),a.side)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'model.json').write_text(json.dumps(b,indent=2));(out/'evaluation.json').write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2))
