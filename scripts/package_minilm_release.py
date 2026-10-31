"""Install the verified MiniLM-only model + Bayesian bundle; enforce serving budget."""
import json,shutil,resource
from pathlib import Path
from pricing.suite_storage import split_model

def main():
    source=Path('training-cache/minilm-cleaned');root=Path('models/xgboost-hierarchy')
    staging=Path('training-cache/minilm-serving');shutil.rmtree(staging,ignore_errors=True)
    for side in ['ask','bid']:
        dest=staging/side;dest.mkdir(parents=True)
        for name in ['calibration-refit.ubj','category-schema.json','evaluation.json','cleaning.json']+(['features.json'] if side=='ask' else []):shutil.copy2(source/side/name,dest/name)
        shutil.copytree(source/side/'bayesian',dest/'bayesian')
        if side=='ask':shutil.copytree('models/xgboost-sparse-minilm/ask/minilm',dest/'minilm')
        if (dest/'calibration-refit.ubj').stat().st_size>8*1024*1024:split_model(dest/'calibration-refit.ubj')
    (staging/'release.json').write_text(json.dumps({'transformer_feature':True,'description_features':'minilm-embeddings','method':'minilm-lemon-squeezer-25','residual_source':'fresh-minilm-fivefold','model':'XGBoost with MiniLM embeddings, 25% training OOF trimming, matching separate Bayesian corrections'},indent=2))
    shutil.rmtree(root);shutil.copytree(staging,root)
    from pricing.hierarchy_sparse_live import estimate
    from datetime import date
    f={'year':2015,'manufacturer':'Toyota','model':'Corolla','odometer':95000,'state':'ca','text':'Full service history. Needs new tires.'}
    results,_,_=estimate(Path('.'),f,date(2026,10,31),True)
    for side,r in results.items():
        if 'error' in r:raise ValueError(side+': '+r['error'])
        print(side,r['price'],flush=True)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    assert rss<550*1024,'Serving smoke exceeded 550 MiB'
    assert size<100*1024*1024,'Serving bundle exceeded 100 MiB'
    print('SERVING VERIFIED',size,'bytes; peak',rss,'KiB',flush=True)
if __name__=='__main__':main()
