"""Fresh Cleanlab audits of both full market cohorts; no transformer or test leakage."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from pricing.label_quality import cleanlab_inliers
from pricing.model_suite import load_cohort,predictors
from marketbridge.structural import IndexPanel,known_pricing_vins


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Completed audit exists; choose new output for a fresh run')
    panel=IndexPanel(a.index);known=known_pricing_vins(a.ask);results={}
    for side,path in [('ask',a.ask),('bid',a.bid)]:
        d=load_cohort(path);excluded=0
        if side=='bid':
            overlap=d.vehicle_group.astype(str).str.removeprefix('vin:').isin(known)
            excluded=int(overlap.sum());d=d.loc[~overlap].reset_index(drop=True)
        x,valid,dates=predictors(d,side);ratios,_=panel.ratios(dates)
        if side=='bid':valid&=ratios.notna()
        d=d.loc[valid].copy();x=x.loc[valid].copy()
        y=d.price.to_numpy(float)
        if side=='bid':y=y/ratios.loc[d.index].to_numpy(float)
        stages={}
        for stage,positions in [('assessment',np.flatnonzero(d.phase=='train')),('deployment',np.arange(len(d)))]:
            print(side,stage,'START',len(positions),flush=True)
            keep,report=cleanlab_inliers(x.iloc[positions],y[positions],out/side/stage,d.row_id.iloc[positions])
            stages[stage]=report
            print(side,stage,'COMPLETE','retained',len(keep),'flagged',report['removed_rows'],flush=True)
        results[side]={'source_sha256':hashlib.file_digest(open(path,'rb'),'sha256').hexdigest(),
            'eligible_rows':len(d),'auction_vin_overlap_exclusions':excluded,
            'phase_counts':d.phase.value_counts().to_dict(),'stages':stages}
        del d,x
    summary={'status':'complete','algorithm':'Cleanlab regression 2.9.0','markets':results,
        'scope':'Freshly computed label audits, not a reuse of prior row-level audit caches. Assessment uses training rows only. Deployment audit uses all eligible fitting candidates after assessment. Validation/test labels do not affect assessment filtering.',
        'features':'Vehicle facts only. No descriptions, text-price predictions, MMR, final-model predictions or source outcome fields.',
        'targets':'log1p historical asking price; log1p CPI-reference auction sale price. Reported flags are suspected errors, not confirmed mistakes.',
        'weights_refitted':False,
        'weight_compatibility':'All retained means no exclusions require a refit. If any rows are flagged, existing model weights remain prior artifacts until a separate evaluated refit is completed.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('ALL AUDITS COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ask',required=True);p.add_argument('--bid',required=True)
    p.add_argument('--index',default='models/structural-transfer/used-car-index.csv');p.add_argument('--output',required=True)
    run(p.parse_args())
