import numpy as np
import pandas as pd
import pytest
import torch
from transformers import XLMRobertaConfig, XLMRobertaModel
from pricing.text import TextPriceRegressor, windows, batch, fit_text, predict_text, save_text, load_text
from pricing.tabular import NUMERIC, CATEGORICAL, fit_schema, features
from pricing.train import folds, fit_lgb, validate_cohort


class TinyTokenizer:
    pad_token_id=1
    bos_token_id=0
    eos_token_id=2
    def encode(self,text,add_special_tokens=False):
        return [4+(ord(c)%20) for c in text]
    def num_special_tokens_to_add(self,pair=False):return 2
    def build_inputs_with_special_tokens(self,x):return [0]+x+[2]
    def pad(self,values,padding=True,return_tensors='pt'):
        rows=values['input_ids'];n=max(map(len,rows))
        return {'input_ids':torch.tensor([r+[1]*(n-len(r)) for r in rows]),
                'attention_mask':torch.tensor([[1]*len(r)+[0]*(n-len(r)) for r in rows])}


def test_long_text_windows_keep_every_token():
    t=TinyTokenizer();text='a'*7000;w=windows(t,text)
    assert len(w)>1 and max(map(len,w))<=512
    prefix=len(t.encode('query: '))
    assert sum(len(x)-prefix-2 for x in w)==len(t.encode(text))
    inp=batch(t,[text,'short'],device='cpu')
    assert inp[2].tolist().count(1)==1


def test_actual_transformer_parameters_receive_regression_gradients():
    torch.set_num_threads(2)
    config=XLMRobertaConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,max_position_embeddings=514,pad_token_id=1)
    model=TextPriceRegressor(XLMRobertaModel(config),9.,.5)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    weight=model.backbone.encoder.layer[0].attention.self.query.weight
    before=weight.detach().clone()
    inputs=batch(TinyTokenizer(),['reliable car','needs repairs'],'cpu')
    loss=(model(*inputs)-torch.tensor([1.,-1.])).square().mean()
    loss.backward();assert weight.grad.abs().sum()>0
    optimizer.step();assert not torch.equal(before,weight)
    p=predict_text(model,TinyTokenizer(),['reliable car'],'cpu')
    assert p.shape==(1,) and np.isfinite(p).all() and p[0]>0


def test_cross_fitting_excludes_each_rows_target():
    seen=[]
    for fit_idx,held_idx in folds(1200,3,42):
        assert not set(fit_idx)&set(held_idx)
        assert len(fit_idx)+len(held_idx)==1200
        seen.extend(held_idx)
    assert sorted(seen)==list(range(1200))


def frame(n=180):
    rng=np.random.default_rng(7)
    d=pd.DataFrame({c:['gas' if c=='fuel' else 'toyota']*n for c in CATEGORICAL})
    d['year']=rng.integers(1990,2021,n)
    d['odometer']=rng.integers(10000,200000,n)
    return d


def test_dynamic_scalar_is_numeric_and_unknown_categories_missing():
    d=frame();schema=fit_schema(d)
    x=features(d,schema,np.arange(len(d))+1000.)
    assert x.predicted_text_price.dtype.kind=='f'
    assert 'price' not in x
    d.loc[0,'manufacturer']='unseen-brand'
    assert pd.isna(features(d,schema).manufacturer.iloc[0])
    with pytest.raises(ValueError):features(d,schema,[1.])


def test_lightgbm_round_trip_and_scalar_sensitivity(tmp_path):
    import lightgbm as lgb
    d=frame();schema=fit_schema(d);text=np.linspace(2000,50000,len(d));y=text.copy()
    x=features(d,schema,text)
    m=fit_lgb(x.iloc[:140],y[:140],x.iloc[140:],y[140:],42)
    p=tmp_path/'model.txt';m.booster_.save_model(str(p))
    reloaded=lgb.Booster(model_file=str(p))
    np.testing.assert_allclose(reloaded.predict(x),m.predict(x))
    assert m.booster_.feature_importance()[list(x).index('predicted_text_price')]>0
    low=features(d.iloc[:1],schema,[2000.]);high=features(d.iloc[:1],schema,[40000.])
    assert reloaded.predict(high)[0]>reloaded.predict(low)[0]


def test_real_prepared_cohort_has_no_removed_rows():
    from pathlib import Path
    path=Path('training-cache/stack-cohort.jsonl')
    if not path.exists():pytest.skip('Source data not downloaded')
    d=pd.read_json(path,lines=True);validate_cohort(d)
    assert d.phase.value_counts().to_dict()=={'train':1200,'validation':300,'test':300}


def test_complete_training_bundle_and_live_inference(tmp_path,monkeypatch):
    """Exercise orchestration with a tiny backbone; never report these as real scores."""
    import json
    from argparse import Namespace
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast
    import pricing.train as training
    from pricing.inference import PricePipeline
    tiny=tmp_path/'tiny'
    config=XLMRobertaConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,max_position_embeddings=514,pad_token_id=1)
    XLMRobertaModel(config).save_pretrained(tiny)
    raw=Tokenizer(WordLevel({'<s>':0,'<pad>':1,'</s>':2,'<unk>':3,'query':4,':':5,'car':6,'maintained':7},unk_token='<unk>'))
    raw.pre_tokenizer=Whitespace()
    raw.post_processor=TemplateProcessing(single='<s> $A </s>',special_tokens=[('<s>',0),('</s>',2)])
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,bos_token='<s>',eos_token='</s>',pad_token='<pad>',unk_token='<unk>')
    monkeypatch.setattr(training.AutoTokenizer,'from_pretrained',lambda *args,**kwargs:tok)
    original=training.fit_text
    fitted_targets=[]
    def tiny_fit(texts,prices,**kwargs):
        fitted_targets.append(set(np.asarray(prices)))
        kwargs.update(source=str(tiny),revision=None)
        return original(texts,prices,**kwargs)
    monkeypatch.setattr(training,'fit_text',tiny_fit)
    d=frame(160)
    d['text']=['maintained car '+str(i) for i in range(len(d))]
    d['row_id']=[str(i) for i in range(len(d))]
    d['price']=np.arange(len(d))*100.+2000.
    d['date']=pd.date_range('2021-04-01',periods=len(d),freq='h',tz='UTC')
    d['phase']=['train']*100+['validation']*30+['test']*30
    source=tmp_path/'data.jsonl';d.to_json(source,orient='records',lines=True,date_format='iso')
    out=tmp_path/'bundle'
    training.run(Namespace(data=str(source),output=str(out),revision='synthetic-test-only',
        folds=2,epochs=1,batch_size=16,accumulation=1,lr=2e-5,seed=42,device='cpu',allow_cpu=True))
    assert len(fitted_targets)==3
    assert all(t.issubset(set(d.price[:100])) for t in fitted_targets)
    assert fitted_targets[-1]==set(d.price[:100])
    scores=json.loads((out/'evaluation.json').read_text())
    assert scores['rows']=={'train':100,'validation':30,'test':30,'removed':0}
    audit=pd.read_csv(out/'oof-audit.csv')
    assert len(audit)==100 and audit.predicted_text_price.notna().all()
    # Reuse held-out folds to generate the experiment cache, then exercise all 60 configurations.
    training.run(Namespace(data=str(source),output=str(out),revision='synthetic-test-only',
        folds=2,epochs=1,batch_size=16,accumulation=1,lr=2e-5,seed=42,device='cpu',allow_cpu=True,text_only=True))
    from pricing.experiments import run as run_experiments
    run_experiments(Namespace(data=str(source),output=str(out),trees=12,seed=42,tabular_only=False,threads=1))
    report=json.loads((out/'experiments.json').read_text())
    assert len(report['experiments'])==60
    assert all(r['retained_rows']==100-int(100*r['removed_fraction']) for r in report['experiments'])
    assert len(report['quantile_intervals'])==10
    # All completed grid models must survive a restart without another fit.
    def no_refit(*args,**kwargs):raise AssertionError('Completed experiments should resume')
    monkeypatch.setattr(training.lgb.LGBMRegressor,'fit',no_refit)
    run_experiments(Namespace(data=str(source),output=str(out),trees=12,seed=42,tabular_only=False,threads=1))
    # Restore the real loader so it reads the tokenizer serialized with the bundle.
    monkeypatch.undo()
    pipeline=PricePipeline(out)
    prediction=pipeline.predict('maintained car',d.iloc[0].to_dict())
    assert np.isfinite(prediction['price']) and prediction['predicted_text_price']>0
    assert prediction['range'][0]<=prediction['price']<=prediction['range'][1]


def test_experiment_rates_losses_and_transforms():
    from pricing.experiments import retained_indices,experiment_grid,inverse,pinball
    scores=np.ones(101)
    for fraction in [0,.05,.1,.25,.5]:
        keep=retained_indices(scores,fraction)
        assert len(keep)==101-int(101*fraction)
        assert keep[0]==0
    assert len(experiment_grid())==60
    np.testing.assert_allclose(inverse([0,1],10000,1000,'dollars'),[10000,11000])
    np.testing.assert_allclose(inverse([0],np.log1p(10000),1,'log1p'),[10000])
    assert pinball([10],[8],.9)==pytest.approx(1.8)


def test_huber_only_grid():
    from pricing.experiments import experiment_grid
    grid=experiment_grid(['huber'],[.5,.9,1.5])
    assert len(grid)==30
    assert all(objective=='huber' for _,_,(_,objective,_) in grid)
    assert {delta for _,_,(_,_,delta) in grid}=={.5,.9,1.5}
    with pytest.raises(ValueError):experiment_grid(['huber'],[-1])
