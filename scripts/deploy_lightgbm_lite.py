"""Install and smoke-test a compact LightGBM Bayesian release."""
import json,shutil,resource
from pathlib import Path
def main():
    from pricing.lightgbm_live import load,base_predict
    from pricing.bayesian_hierarchy import predict,interval
    import pandas as pd
    root=Path('models/lightgbm-lite')
    for side in ['ask','bid']:
        src=Path('training-cache/lightgbm-lite')/side;dst=root/side;dst.mkdir(parents=True,exist_ok=True)
        for name in ['calibration-refit.txt','category-schema.json','evaluation.json','cleaning.json']:shutil.copy2(src/name,dst/name)
        shutil.copytree(src/'bayesian',dst/'bayesian',dirs_exist_ok=True)
        model,schema,meta,draws,_,report=load('.',side)
        fields={'year':2015,'manufacturer':'toyota','model':'corolla','odometer':95000,'state':'ca'}
        base,_=base_predict(model,schema,None,fields,6.,side)
        bp=predict(meta,draws,pd.DataFrame([{**fields,'age':6.,'base_price':base}]))
        low,high=interval(bp,meta['interval']['multiplier'])
        assert 0<low[0]<=bp.corrected_price.iloc[0]<=high[0]
        print(side,'LIVE INFERENCE PASSED',base,report,flush=True)
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert size<40*1024*1024, 'Serving weights exceed 40 MB budget'
    assert rss<450*1024, 'Peak smoke-test RSS exceeds 450 MiB'
    (root/'release.json').write_text(json.dumps({'method':'lemon-lite','bytes':size,'smoke_peak_rss_kib':rss,'model':'LightGBM log-median + matching Bayesian hierarchy'}))
    print('PACKAGE BYTES',size,'SMOKE PEAK RSS KiB',rss,flush=True)
if __name__=='__main__':main()
