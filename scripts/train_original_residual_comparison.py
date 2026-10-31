"""Full original-cohort, chronological assessment. Never samples or trims residuals."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from pricing.model_suite import load_cohort,predictors
from marketbridge.structural import IndexPanel,known_pricing_vins
from pricing.forest_residuals import generate,CATEGORICAL
from pricing.hierarchical import experiment


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    known=known_pricing_vins(args.ask)
    panel=IndexPanel(args.index)
    for side,path in [('ask',args.ask),('bid',args.bid)]:
        if args.side and args.side!=side:continue
        dest=out/side;dest.mkdir(exist_ok=True)
        print(f'Loading full original {side} cohort',flush=True)
        d=load_cohort(path);original=len(d)
        x,valid,dates=predictors(d,side);ratios,_=panel.ratios(dates)
        overlap=pd.Series(False,index=d.index)
        if side=='bid':
            overlap=d.vehicle_group.astype(str).str.removeprefix('vin:').isin(known)
            valid&=ratios.notna()&~overlap
        d=d.loc[valid].copy();x=x.loc[valid]
        factor=ratios.loc[d.index].to_numpy(float) if side=='bid' else np.ones(len(d))
        groups=d.vehicle_group if 'vehicle_group' in d else d.row_id
        frame=pd.DataFrame({'row_id':d.row_id.astype(str),'vehicle_group':groups.astype(str).map(lambda s:hashlib.sha256(s.encode()).hexdigest()),
            'split':d.phase.map({'train':'calibration','validation':'validation','test':'test'}),
            'manufacturer':x.manufacturer,'model':x.model,'age':x.age,'odometer':d.odometer,
            'auction_grade':x.auction_grade,'observed_price':d.price.to_numpy(float)/factor,
            'price_basis':'2021 advertised asking USD' if side=='ask' else '2015-01 CPI-reference auction USD'})
        for c in CATEGORICAL:frame[c]=x[c]
        frame=frame.reset_index(drop=True)
        provenance={'side':side,'source':'Craigslist Used Cars / Tom / Zenodo 10457828' if side=='ask' else 'Used Car Auction Prices / tunguz / Kaggle',
            'source_url':'https://zenodo.org/records/10457828' if side=='ask' else 'https://www.kaggle.com/datasets/tunguz/used-car-auction-prices',
            'license':'CC-BY-4.0' if side=='ask' else 'CC0-1.0','fraction_of_eligible_original':1.0,
            'canonical_rows':original,'eligible_rows':len(frame),'cross_source_vin_exclusions':int(overlap.sum()),
            'splits':frame.split.value_counts().to_dict(),'date_min':str(dates.loc[valid].min()),'date_max':str(dates.loc[valid].max()),
            'source_archive_sha256':sha(Path(path).parent/('ask.zip' if side=='ask' else 'bid.zip')),'canonical_sha256':sha(path),'source_preparation':json.loads(Path(path).with_suffix('.meta.json').read_text()),
            'price_basis':frame.price_basis.iloc[0],'residual_trimming':False,
            'scope':f'Fresh {args.engine} log-price assessment; excludes MiniLM and five-stage production cleaning. All eligible original rows, not a sample. Chronological held-out cohorts; calibration OOF folds grouped by vehicle.',
            'ask_eligibility':'Existing full-cohort recipe: $1k–150k, cleaned description >=80 characters, dated, deduplicated text/VIN; valid year and mileage. Text is used for eligibility/deduplication only, not as a feature.'}
        (dest/'provenance.json').write_text(json.dumps(provenance,indent=2))
        residual_path=dest/'residuals.csv.gz'
        if residual_path.exists():
            residuals=pd.read_csv(residual_path);base_meta=json.loads((dest/'base-assessment.json').read_text())
        else:
            residuals,base_meta=generate(frame,side,max_rows=1000000,model_path=dest/'base-assessment.ubj')
            residuals.to_csv(residual_path,index=False,float_format='%.12g')
            (dest/'base-assessment.json').write_text(json.dumps(base_meta,indent=2))
        print(f'Fitting hierarchy over forest residuals on {len(residuals):,} original {side} observations',flush=True)
        bundle,report,rows=experiment(residuals,side,max_rows=1000000)
        report['base_assessment']=base_meta;report['provenance']=provenance
        with gzip.open(dest/'model.json.gz','wt') as f:json.dump(bundle,f)
        (dest/'evaluation.json').write_text(json.dumps(report,indent=2))
        rows.to_csv(dest/'test_predictions.csv.gz',index=False,float_format='%.12g')
        # Display subset only: every score and bootstrap above uses the full held-out cohort.
        rows.sample(n=min(1000,len(rows)),random_state=42).reset_index(drop=True).to_csv(dest/'examples.csv.gz',index=False,float_format='%.12g')
        print(json.dumps(report['cohorts'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ask',default='training-cache/full-cohort.jsonl');p.add_argument('--bid',default='training-cache/bid-cohort.jsonl');p.add_argument('--index',default='models/structural-transfer/used-car-index.csv');p.add_argument('--output',default='training-cache/forest-assessment-fit');p.add_argument('--side',choices=['ask','bid']);p.add_argument('--engine',choices=['forest'],default='forest');run(p.parse_args())
