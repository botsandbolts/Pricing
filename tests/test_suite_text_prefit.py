"""A deployment prefit cannot masquerade as a completed held-out assessment."""
import hashlib,json
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold
from pricing.suite_text import validated_features


def test_full_oof_can_prefit_but_complete_assessment_still_requires_held(tmp_path):
    rows=pd.DataFrame({'row_id':['car-'+str(i) for i in range(20)],'phase':['train']*12+['validation']*4+['test']*4})
    cohort=tmp_path/'cohort.jsonl';rows.to_json(cohort,orient='records',lines=True)
    root=tmp_path/'features';root.mkdir()
    provenance={'fingerprint':'same-model','revision':'pinned','fitting_rows':12}
    (root/'provenance.json').write_text(json.dumps({'cohort_sha256':hashlib.sha256(cohort.read_bytes()).hexdigest(),'assessment_text_provenance':provenance}))
    (root/'full-text-provenance.json').write_text(json.dumps({**provenance,'fitting_rows':20}))
    for count,name in [(20,'full-oof.csv'),(12,'assessment-oof.csv')]:
        folds=np.empty(count,int)
        for fold,(_,held) in enumerate(KFold(3,shuffle=True,random_state=42).split(range(count))):folds[held]=fold
        pd.DataFrame({'row_id':rows.row_id.iloc[:count],'fold':folds,'predicted_text_price':np.full(count,10000.)}).to_csv(root/name,index=False)
    assert len(validated_features(cohort,root,full_only=True))==20
    with pytest.raises(FileNotFoundError):validated_features(cohort,root)
    pd.DataFrame({'row_id':rows.row_id.iloc[12:],'predicted_text_price':np.full(8,12000.)}).to_csv(root/'held-predictions.csv',index=False)
    result=validated_features(cohort,root)
    assert len(result)==20 and len(result.attrs['full_oof'])==20
    bad=pd.read_csv(root/'full-oof.csv');bad.loc[0,'fold']=9;bad.to_csv(root/'full-oof.csv',index=False)
    with pytest.raises(ValueError,match='fold assignment'):validated_features(cohort,root,full_only=True)
