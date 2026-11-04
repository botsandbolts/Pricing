"""Independent BGE domain-adaptation shards; no price targets used."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer,AutoModelForMaskedLM,DataCollatorForLanguageModeling,Trainer,TrainingArguments
from datasets import Dataset
from pricing.advert_text import scrub
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args()
if not 0<=a.shard<25:raise ValueError('Invalid shard')
torch.set_num_threads(2)
# Calibration-only descriptions: held-out pricing validation/test never train MLM.
base=pd.read_csv('frozen-baseline/ask/hierarchy-input.csv.gz',dtype={'row_id':str})
allowed=set(base.loc[base.split.eq('calibration'),'row_id'])
texts=[];ids=[];scanned=0
for frame in pd.read_json('training-cache/full-cohort.jsonl',lines=True,chunksize=1000,dtype={'row_id':str}):
    eligible=frame.loc[frame.row_id.isin(allowed)]
    for row in eligible.itertuples():
        if scanned%25==a.shard:
            value=scrub('' if pd.isna(row.text) else str(row.text))
            if value.strip():texts.append(value);ids.append(str(row.row_id))
        scanned+=1
if len(texts)<16:raise ValueError('Too few calibration descriptions')
manifest=json.loads(Path('training-cache/bge-mlm-manifest.json').read_text())
model_id=manifest['model'];revision=manifest['revision']
tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision)
model=AutoModelForMaskedLM.from_pretrained(model_id,revision=revision)
data=Dataset.from_dict({'text':texts})
data=data.map(lambda batch:tokenizer(batch['text'],truncation=True,max_length=256),batched=True,remove_columns=['text'])
split=data.train_test_split(test_size=.1,seed=42)
out=Path('bge-mlm')/f'shard-{a.shard:02d}'
args=TrainingArguments(output_dir=str(out),learning_rate=1e-5,max_steps=300,
 per_device_train_batch_size=8,per_device_eval_batch_size=8,
 seed=42,data_seed=42,use_cpu=True,save_strategy='no',logging_steps=50,
 report_to=[],dataloader_num_workers=0)
trainer=Trainer(model=model,args=args,train_dataset=split['train'],eval_dataset=split['test'],
 data_collator=DataCollatorForLanguageModeling(tokenizer,mlm_probability=.15))
torch.manual_seed(1729);before=trainer.evaluate()
trainer.train()
torch.manual_seed(1729);after=trainer.evaluate()
trainer.save_model(str(out));tokenizer.save_pretrained(str(out))
(out/'adaptation.json').write_text(json.dumps({**manifest,'shard':a.shard,'shards':25,
 'calibration_descriptions':len(texts),'row_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
 'mlm_before':before,'mlm_after':after,'mask_probability':.15,'steps':300,'max_tokens':256,
 'limits':'Independent shard checkpoint, not distributed joint training. MLM improvement does not establish embedding or pricing improvement. Base BGE remains deployed; no weight averaging.'},indent=2))
