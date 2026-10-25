"""Use the committed make/model price head; download the pinned public MiniLM encoder."""
import argparse,json
from pathlib import Path
import torch
import pandas as pd
from transformers import AutoTokenizer,AutoModel
from pricing.identity import encode,names,new_head,head_prices


def predict(make,model):
    release=json.loads((Path(__file__).resolve().parents[1]/'models/make-model-price/head.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(release['source'],revision=release['revision'])
    encoder=AutoModel.from_pretrained(release['source'],revision=release['revision']).eval()
    head=new_head(release['dimensions'])
    head.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in release['head'].items()});head.eval()
    text=names(pd.DataFrame({'manufacturer':[make],'model':[model]}))
    x=encode(encoder,tokenizer,text,'cpu')
    return float(head_prices(head,x,release['mean'],release['std'])[0])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('make');p.add_argument('model');a=p.parse_args()
    torch.set_num_threads(4)
    print(json.dumps({'make':a.make,'model':a.model,'predicted_make_model_price':predict(a.make,a.model),
        'scope':'make/model-only historical asking-price signal; not a complete valuation'},indent=2))
