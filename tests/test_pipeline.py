import json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pricing.oof import cross_fit


def callbacks(y,seen=None):
    def fit(idx,targets):
        if seen is not None:seen.append((idx.copy(),targets.copy()))
        return float(np.mean(targets))
    def predict(mean,idx):return np.full(len(idx),mean)
    return fit,predict


def test_oof_excludes_every_predicted_rows_target():
    y=np.arange(12.)+1000;seen=[];fit,predict=callbacks(y,seen)
    p,fold=cross_fit([str(i) for i in range(12)],y,fit,predict,folds=3)
    for k,(idx,targets) in enumerate(seen):
        held=np.flatnonzero(fold==k)
        assert not set(idx)&set(held)
        np.testing.assert_allclose(p[held],y[idx].mean())
        np.testing.assert_array_equal(targets,y[idx])


def test_changing_held_target_cannot_change_its_oof_prediction():
    y=np.arange(12.)+1000;fit,predict=callbacks(y)
    p,_=cross_fit(list(map(str,range(12))),y,fit,predict,folds=3)
    changed=y.copy();changed[4]=100000
    fit,predict=callbacks(changed)
    q,_=cross_fit(list(map(str,range(12))),changed,fit,predict,folds=3)
    assert p[4]==q[4]


def test_resume_and_reject_changed_targets(tmp_path):
    y=np.arange(12.)+1000;fit,predict=callbacks(y)
    p,f=cross_fit(list(map(str,range(12))),y,fit,predict,cache=tmp_path)
    def no_fit(*args):raise AssertionError('Completed folds should resume')
    q,g=cross_fit(list(map(str,range(12))),y,no_fit,predict,cache=tmp_path)
    np.testing.assert_array_equal(p,q);np.testing.assert_array_equal(f,g)
    with pytest.raises(ValueError):cross_fit(list(map(str,range(12))),y+1,no_fit,predict,cache=tmp_path)


def test_reject_duplicate_ids_or_invalid_predictions():
    fit,predict=callbacks([1000]*3)
    with pytest.raises(ValueError):cross_fit(['a','a','b'],[1000]*3,fit,predict)
    with pytest.raises(ValueError):cross_fit(['a','b','c'],[1000]*3,fit,lambda m,idx:np.full(len(idx),np.nan))


def frame(n=60):
    from pricing.tabular import CATEGORICAL
    rng=np.random.default_rng(42);d=pd.DataFrame({c:['toyota']*n for c in CATEGORICAL})
    d['year']=rng.integers(1990,2021,n);d['odometer']=rng.integers(10000,200000,n)
    d['price']=rng.integers(2000,50000,n).astype(float);d['row_id']=[str(i) for i in range(n)]
    d['text']=['maintained car '+('maintained '*(1 if i%5 else i%19+2))+str(i) for i in range(n)]
    d['date']=pd.date_range('2021-04-01',periods=n,freq='h',tz='UTC')
    d['phase']=['train']*int(n*.7)+['validation']*(int(n*.85)-int(n*.7))+['test']*(n-int(n*.85))
    return d


def test_real_transformer_oof_full_stack_and_all_row_refit(tmp_path,monkeypatch):
    import torch
    from transformers import XLMRobertaConfig,XLMRobertaModel,AutoModel,AutoTokenizer,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    import pricing.train as train
    from pricing.text import MODEL_ID
    from pricing.inference import PricePipeline
    torch.set_num_threads(1)
    tiny=tmp_path/'tiny'
    config=XLMRobertaConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,max_position_embeddings=514,pad_token_id=1)
    XLMRobertaModel(config).save_pretrained(tiny)
    raw=Tokenizer(WordLevel({'<s>':0,'<pad>':1,'</s>':2,'<unk>':3,'query':4,':':5,'car':6,'maintained':7},unk_token='<unk>'))
    raw.pre_tokenizer=Whitespace();raw.post_processor=TemplateProcessing(single='<s> $A </s>',special_tokens=[('<s>',0),('</s>',2)])
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,bos_token='<s>',eos_token='</s>',pad_token='<pad>',unk_token='<unk>');tok.save_pretrained(tiny)
    real_model=AutoModel.from_pretrained;real_tok=AutoTokenizer.from_pretrained
    def load_model(source,*args,**kwargs):
        if str(source) in [MODEL_ID]:source=tiny;kwargs.pop('revision',None)
        return real_model(source,*args,**kwargs)
    def load_tok(source,*args,**kwargs):
        if str(source) in [MODEL_ID]:source=tiny;kwargs.pop('revision',None)
        return real_tok(source,*args,**kwargs)
    monkeypatch.setattr(AutoModel,'from_pretrained',load_model);monkeypatch.setattr(AutoTokenizer,'from_pretrained',load_tok)
    # Record which targets reach each text-regression fit. No holdout prices enter assessment fits.
    fit=train.fit_text;targets=[]
    def record(texts,y,**kwargs):targets.append(set(np.asarray(y)));return fit(texts,y,**kwargs)
    monkeypatch.setattr(train,'fit_text',record)
    d=frame();source=tmp_path/'data.jsonl';d.to_json(source,orient='records',lines=True,date_format='iso')
    out=tmp_path/'bundle';a=Namespace(data=str(source),output=str(out),device='cpu',allow_cpu=True,threads=1,
        revision='synthetic-test',folds=2,epochs=1,batch_size=8,accumulation=1,lr=2e-5,trees=8,seed=42)
    train.run(a)
    tr=set(d.loc[d.phase=='train','price'])
    assert all(s.issubset(tr) for s in targets[:3])
    assert targets[2]==tr and targets[-1]==set(d.price)
    for directory,count in [('assessment',42),('',60)]:
        audit=pd.read_csv(out/directory/'oof-predictions.csv')
        assert len(audit)==count
        assert audit.predicted_text_price.notna().all()
        assert set(audit.fold)=={0,1}
    manifest=json.loads((out/'manifest.json').read_text());evaluation=json.loads((out/'evaluation.json').read_text())
    assert manifest['training_candidates']==60 and manifest['trained_rows']<=60
    assert manifest['outlier_filter']['algorithm']=='IsolationForest'
    assert manifest['outlier_filter']['input']=='predicted_text_price'
    assert evaluation['rows']['outlier_removed']+evaluation['rows']['retained_train']==42
    monkeypatch.undo()
    pipeline=PricePipeline(out)
    p=pipeline.predict('maintained car',d.iloc[0].to_dict())
    assert p['price']>=0 and p['predicted_text_price']>0
    assert 'predicted_text_price' in pipeline.lgb.feature_name()


def test_description_transformer_receives_regression_gradients():
    import torch
    from transformers import XLMRobertaConfig,XLMRobertaModel
    from pricing.text import TextPriceRegressor
    model=TextPriceRegressor(XLMRobertaModel(XLMRobertaConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24)),9.,.5)
    weight=model.backbone.encoder.layer[0].attention.self.query.weight
    loss=(model(torch.tensor([[0,4,5,2]]),torch.ones((1,4),dtype=torch.long),torch.tensor([0]),1)-torch.tensor([1.])).square().mean()
    loss.backward();assert weight.grad is not None and weight.grad.abs().sum()>0


def test_scalar_filter_and_no_make_model_transformer():
    source=Path('pricing/train.py').read_text()
    assert 'IsolationForest' in source and 'LocalOutlierFactor' not in source
    assert 'pricing.identity' not in source
    assert not Path('pricing/identity.py').exists()


def test_full_preparation_retains_empty_free_fields_without_sampling(tmp_path):
    import zipfile
    from scripts.prepare_all import prepare
    d=frame(12);d['id']=np.arange(12)
    d['description']=['listing '+chr(97+i) for i in range(12)]
    d.loc[2,'description']=d.loc[0,'description']
    d.loc[[8,9],'description']=None;d['VIN']=''
    d['posting_date']=d.date.astype(str);d.loc[11,'posting_date']='invalid-date';d.loc[10,'price']=0
    source=tmp_path/'vehicles.zip'
    with zipfile.ZipFile(source,'w') as z:z.writestr('vehicles.csv',d.to_csv(index=False))
    output=tmp_path/'all.jsonl';prepare(source,output)
    meta=json.loads(output.with_suffix('.meta.json').read_text());cohort=pd.read_json(output,lines=True)
    assert meta['source_rows']==12 and meta['sample_fraction']==1.
    assert meta['outlier_filter'] is None and len(cohort)==9
    assert meta['empty_descriptions_retained']==2


def test_isolation_filter_uses_price_scalar(tmp_path):
    from pricing.train import filter_prices
    prices=np.r_[np.random.default_rng(42).normal(10000,100,100),1e6]
    keep=filter_prices(prices,tmp_path,np.arange(len(prices)),42,1)
    assert 100 not in keep
    audit=pd.read_csv(tmp_path/'isolation-audit.csv')
    assert list(audit)==['row_id','predicted_text_price','score','kept']
