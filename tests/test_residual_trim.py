import numpy as np
from scripts.train_residual_trimmed_xgb import trim_mask


def test_quarter_per_phase_plus_union_and_deterministic_ties():
    ids=np.array([str(i) for i in range(12)]);phase=np.repeat(['train','validation','test'],4)
    residual=np.tile([.1,-3.,3.,.2],3)
    keep,report=trim_mask(ids,phase,residual,{'0'})
    assert not keep[0] and not keep[1] and keep[2]
    assert report['train']['top_residual_removed']==1
    assert report['train']['additional_outliertree_removed']==1
    assert report['validation']['removed_rows']==1 and report['test']['removed_rows']==1
    assert keep.sum()==8


def test_inference_reuses_rare_model_mapping():
    import pandas as pd
    from pricing.model_suite import apply_category_pooling,encode
    x=pd.DataFrame({'model':['corolla','rare trim',None],'manufacturer':['toyota','ford','ford']})
    pooled=apply_category_pooling(x,{'model':{'retained_levels':['corolla'],'replacement':'__rare__'}})
    encoded=encode(pooled,{'model':['corolla','__rare__']})
    assert encoded.model.tolist()==['corolla','__rare__','__rare__']
    assert encoded.model.notna().all() and pooled.manufacturer.tolist()==x.manufacturer.tolist()
    assert x.model.iloc[1]=='rare trim'
