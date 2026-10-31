"""Package verified XGBoost refits and matching Bayesian posteriors."""
import json,shutil
from pathlib import Path
from pricing.suite_storage import split_model,read_model_file
from pricing.fivefold_live import load,base_predict
def main():
    root=Path('models/fivefold-live')
    for side in ['ask','bid']:
        source=Path('training-cache/xgb-source')/side
        dest=root/side;dest.mkdir(parents=True,exist_ok=True)
        for name in ['calibration-refit.ubj','category-schema.json','evaluation.json','cleaning.json']+(['calibration-refit-tfidf-umap.joblib'] if side=='ask' else []):
            shutil.copy2(source/name,dest/name)
        shutil.copytree(source/'bayesian',dest/'bayesian',dirs_exist_ok=True)
        for name in ['calibration-refit.ubj','calibration-refit-tfidf-umap.joblib']:
            p=dest/name
            if p.exists() and p.stat().st_size>8*1024*1024:split_model(p)
        model,schema,meta,draws,projector,report=load('.',side)
        fields={'year':2015,'manufacturer':'toyota','model':'corolla','odometer':95000,'state':'ca','text':'Well maintained Toyota Corolla, clean title, automatic transmission.'}
        base,unknown=base_predict(model,schema,projector,fields,6.,side)
        from pricing.bayesian_hierarchy import predict,interval
        import pandas as pd
        bp=predict(meta,draws,pd.DataFrame([{**fields,'age':6.,'base_price':base}]))
        lo,hi=interval(bp,meta['interval']['multiplier'])
        assert 0<lo[0]<=bp.corrected_price.iloc[0]<=hi[0]
        print(side,'SMOKE OK',base,report,flush=True)
    (root/'release.json').write_text(json.dumps({'method':'lemon-squeezer-25','model':'XGBoost trained after 25% OOF residual cleaning + matching Bayesian log residuals','ask_run':41189327994,'bid_run':41190411926}))
if __name__=='__main__':main()
