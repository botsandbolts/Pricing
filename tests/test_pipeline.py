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


@pytest.mark.parametrize('namespace',['pricing','carbids'])
def test_real_transformer_oof_full_stack_and_all_row_refit(tmp_path,monkeypatch,namespace):
    import torch
    from transformers import XLMRobertaConfig,XLMRobertaModel,AutoModel,AutoTokenizer,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    import importlib
    train=importlib.import_module(namespace+'.train')
    MODEL_ID=importlib.import_module(namespace+'.text').MODEL_ID
    PricePipeline=importlib.import_module(namespace+'.inference').PricePipeline
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
    if namespace=='carbids':
        d['source']='synthetic-test';d.to_json(source,orient='records',lines=True,date_format='iso')
        monkeypatch.setattr('carbids.data.load_cohort',lambda *args,**kwargs:d.copy())
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
    assert manifest['outlier_filter']['algorithm']=='Cleanlab'
    assert 'predicted_text_price' not in manifest['outlier_filter']['features']
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
    assert 'cleanlab_inliers' in source and 'IsolationForest' not in source
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


def test_baseline_evaluates_before_refitting_every_row(tmp_path):
    from pricing.baseline import run
    from pricing.inference import PricePipeline
    d=frame(60);source=tmp_path/'data.jsonl';d.to_json(source,orient='records',lines=True,date_format='iso')
    source.with_suffix('.meta.json').write_text(json.dumps({'fraction':1.,'sampled_raw_rows':60}))
    output=tmp_path/'model';public=tmp_path/'public'
    run(Namespace(data=str(source),output=str(output),public=str(public),trees=8,threads=1,seed=42))
    report=json.loads((output/'evaluation.json').read_text());manifest=json.loads((output/'manifest.json').read_text())
    assert report['train_rows']==42 and report['test_rows']==9
    assert report['deployment_refit_rows']==60 and manifest['trained_rows']==60
    assert report['sample_fraction']==1.
    raw=np.log1p(d.price) if manifest['target_transform']=='log1p' else d.price
    assert manifest['target_mean']==pytest.approx(raw.mean())
    assert (output/'assessment/lightgbm.txt').exists()
    prediction=PricePipeline(output).predict('',d.iloc[0].to_dict())
    assert np.isfinite(prediction['price']) and prediction['price']>=0
    import gzip
    raw_model=output/'lightgbm.txt'
    with gzip.open(output/'lightgbm.txt.gz','wb') as f:f.write(raw_model.read_bytes())
    raw_model.unlink()
    compressed=PricePipeline(output).predict('',d.iloc[0].to_dict())
    assert compressed['price']==pytest.approx(prediction['price'])


def test_full_preparation_streams_all_rows_and_deduplicates(tmp_path):
    import zipfile
    from scripts.prepare_full import prepare
    d=frame(20);d['description']=[('Well maintained car '+chr(97+i)+' ')*6 for i in range(20)]
    d.loc[2,'description']=d.loc[0,'description'];d['VIN']='';d['posting_date']=d.date.astype(str);d.loc[19,'price']=0
    source=tmp_path/'source.zip'
    with zipfile.ZipFile(source,'w') as z:z.writestr('vehicles.csv',d.to_csv(index=False))
    output=tmp_path/'full.jsonl';prepare(source,output,workers=1)
    metadata=json.loads(output.with_suffix('.meta.json').read_text());rows=pd.read_json(output,lines=True)
    assert metadata['raw_rows']==20 and metadata['fraction']==1.
    assert len(rows)==18 and rows.row_id.is_unique
    assert sum(metadata['counts'].values())==18
    assert not output.with_suffix('.preparation.sqlite').exists()


def test_minilm_windows_use_cls_sep_and_preserve_long_descriptions():
    from transformers import BertTokenizer
    from tempfile import TemporaryDirectory
    from pricing.text import windows, MAX_TOKENS, MODEL_ID
    with TemporaryDirectory() as directory:
        vocab=Path(directory)/'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncar\n')
        tokenizer=BertTokenizer(vocab_file=str(vocab))
        chunks=windows(tokenizer,'car ' * 700)
        assert MODEL_ID == 'sentence-transformers/all-MiniLM-L6-v2'
        assert len(chunks)>1
        assert all(len(c)<=MAX_TOKENS and c[0]==tokenizer.cls_token_id and c[-1]==tokenizer.sep_token_id for c in chunks)
        assert sum(len(c)-2 for c in chunks)==len(tokenizer.encode('car ' * 700,add_special_tokens=False))


def test_optimizer_checkpoint_resume_matches_uninterrupted(tmp_path,monkeypatch):
    import torch
    from transformers import BertConfig,BertModel
    import pricing.text as text
    torch.set_num_threads(1)
    config=BertConfig(vocab_size=16,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16)
    monkeypatch.setattr(text.AutoModel,'from_pretrained',lambda *a,**k:BertModel(config))
    def inputs(tokenizer,texts,device):
        ids=torch.tensor([[1,int(t)+2,3] for t in texts],device=device)
        return ids,torch.ones_like(ids),torch.arange(len(texts),device=device),len(texts)
    monkeypatch.setattr(text,'batch',inputs)
    kwargs=dict(tokenizer=None,source='tiny-bert',epochs=2,batch_size=2,accumulation=1,device='cpu',checkpoint_steps=1)
    texts=['0','1','2','3'];prices=[2000,4000,8000,16000]
    reference=text.fit_text(texts,prices,**kwargs)
    def interrupt(message,**kwargs):
        if 'seen=2/' in message:raise InterruptedError('simulated interruption after checkpoint')
    with pytest.raises(InterruptedError):
        text.fit_text(texts,prices,checkpoint_dir=tmp_path,log=interrupt,**kwargs)
    restored=text.fit_text(texts,prices,checkpoint_dir=tmp_path,**kwargs)
    for key,value in reference.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)


@pytest.mark.parametrize('target',['log1p','dollars'])
def test_shipped_model_explanations_reconstruct_prediction(target):
    from pricing.inference import PricePipeline
    pipeline=PricePipeline(Path(__file__).resolve().parents[1]/'models/studio-baseline')
    pipeline.manifest['target_transform']=target
    fields={'year':2015,'odometer':95000,'manufacturer':'toyota','model':'corolla'}
    plain=pipeline.predict('',fields)
    explained=pipeline.predict('',fields,explain=True)
    assert plain['price']==explained['price']
    e=explained['explanation']
    assert e['additivity_error']<1e-7
    assert abs(e['base']+sum(r['contribution'] for r in e['features'])-e['raw_prediction'])<1e-7
    assert [r['feature'] for r in e['features']]==pipeline.lgb.feature_name()
    assert all(np.isfinite(r['contribution']) for r in e['features'])
    assert sum(r['gain_pct'] for r in e['features'])==pytest.approx(100)
