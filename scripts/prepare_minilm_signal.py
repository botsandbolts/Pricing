"""Export pretrained MiniLM once, then compute label-free int8 embeddings offline."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    import torch
    from transformers import AutoModel,AutoTokenizer
    from onnxruntime.quantization import quantize_dynamic,QuantType
    torch.set_num_threads(2)
    source='sentence-transformers/all-MiniLM-L6-v2'
    model=AutoModel.from_pretrained(source).eval()
    revision=getattr(model.config,'_commit_hash',None)
    tokenizer=AutoTokenizer.from_pretrained(source,revision=revision)
    tokenizer.backend_tokenizer.save(str(out/'tokenizer.json'))
    class MeanPool(torch.nn.Module):
        def __init__(self,model):super().__init__();self.model=model
        def forward(self,input_ids,attention_mask):
            hidden=self.model(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
            mask=attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            return torch.nn.functional.normalize(pooled,dim=-1)
    dummy=tokenizer(['A clean Toyota Corolla'],return_tensors='pt')
    torch.onnx.export(MeanPool(model),(dummy['input_ids'],dummy['attention_mask']),str(out/'encoder.onnx'),
        input_names=['input_ids','attention_mask'],output_names=['embedding'],
        dynamic_axes={'input_ids':{0:'batch',1:'tokens'},'attention_mask':{0:'batch',1:'tokens'},'embedding':{0:'batch'}},
        opset_version=17,dynamo=False)
    quantize_dynamic(str(out/'encoder.onnx'),str(out/'encoder-int8.onnx'),weight_type=QuantType.QInt8)
    del model,tokenizer
    import gc;gc.collect()
    from pricing.minilm_signal import Encoder
    encoder=Encoder(out);chunks=[];ids=[]
    for d in pd.read_json(a.cohort,lines=True,chunksize=1024):
        chunks.append(encoder.encode(d.text.fillna('').astype(str).tolist()))
        ids.extend(d.row_id.astype(str).tolist())
        print('MINILM EMBEDDED',len(ids),flush=True)
    np.save(out/'embeddings.npy',np.concatenate(chunks))
    (out/'row-ids.json').write_text(json.dumps(ids))
    (out/'provenance.json').write_text(json.dumps({'source':source,'revision':revision,'cohort_sha256':hashlib.file_digest(open(a.cohort,'rb'),'sha256').hexdigest(),'frozen':True,'supervised_backbone_training':False,'max_tokens':128,'description_characters':8000,'head':'Ridge(alpha=100) on log dollars; group OOF within each XGBoost training fold'},indent=2))
if __name__=='__main__':main()
