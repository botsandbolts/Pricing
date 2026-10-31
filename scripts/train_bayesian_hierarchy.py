"""Fit a joint posterior on the original OOF residuals, then validate intervals."""
import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from pricing.bayesian_hierarchy import fit, assess, save
from pricing.hierarchical import validate


def run(args):
    for side in args.sides:
        source=Path(args.input)/side/'residuals.csv.gz'
        if not source.exists():
            raise FileNotFoundError(f'{source}: regenerate original group-OOF forest residuals first')
        raw=pd.read_csv(source)
        data=validate(raw,max_rows=1000000)
        cal=data[data.split=='calibration']; val=data[data.split=='validation'];test=data[data.split=='test']
        print(f'{side}: sampling joint posterior from {len(cal):,} OOF errors',flush=True)
        meta,draws=fit(cal,warmup=args.warmup,draws=args.draws)
        if max(meta['rhat'].values())>1.05:
            raise RuntimeError(f'{side}: sampler has not converged; raise warmup/draw counts: {meta["rhat"]}')
        print(f'{side}: calibrating on {len(val):,} and measuring {len(test):,} untouched cars',flush=True)
        report,pv,pt=assess(meta,draws,val,test)
        report['side']=side;report['rhat']=meta['rhat'];report['train_rows']=len(cal)
        report['base_model_id']=meta['base_model_id'];report['price_basis']=meta['price_basis']
        destination=Path(args.output)/side
        save(destination,meta,draws,report)
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='training-cache/bayesian-refit')
    p.add_argument('--output',default='models/forest-original')
    p.add_argument('--sides',nargs='+',choices=['ask','bid'],default=['ask','bid'])
    p.add_argument('--warmup',type=int,default=180)
    p.add_argument('--draws',type=int,default=160)
    run(p.parse_args())
