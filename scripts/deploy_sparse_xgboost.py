"""Package verified XGBoost refits and matching Bayesian posteriors."""
import json,shutil
from pathlib import Path
from pricing.suite_storage import split_model,read_model_file
from pricing.sparse_live import load,base_predict
def main():
    root=Path('models/xgboost-sparse-minilm')
    for side in ['ask','bid']:
        source=Path('training-cache/xgboost-sparse')/side
        dest=root/side;dest.mkdir(parents=True,exist_ok=True)
        for name in ['calibration-refit.ubj','category-schema.json','evaluation.json','cleaning.json']+(['tfidf.joblib'] if side=='ask' else []):
            shutil.copy2(source/name,dest/name)
        if side=='ask':
            shutil.copy2(source/'minilm-head.json',dest/'minilm-head.json')
            (dest/'minilm').mkdir(exist_ok=True)
            for name in ['encoder-int8.onnx','tokenizer.json','provenance.json']:
                shutil.copy2(Path('training-cache/minilm-signal')/name,dest/'minilm'/name)
        shutil.copytree(source/'bayesian',dest/'bayesian',dirs_exist_ok=True)
        for name in ['calibration-refit.ubj','tfidf.joblib']:
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
    import resource
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('FULL ASK+BID SMOKE PEAK RSS KiB',rss,flush=True)
    assert rss<600*1024,'Serving inference exceeded 600 MiB smoke budget'
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    assert size<100*1024*1024, 'Serving weights exceed 100 MiB budget'
    print('SERVING BYTES',size,flush=True)
    (root/'release.json').write_text(json.dumps({'transformer_feature':'frozen-MiniLM-int8-crossfitted-price-head','residual_source':'fresh-tfidf-fivefold','method':'sparse-lemon-squeezer-25','model':'XGBoost trained after 25% OOF residual cleaning + matching Bayesian log residuals','ask_run':41189327994,'bid_run':41190411926}))
if __name__=='__main__':main()
