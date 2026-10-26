"""Structural auction transport plus explicitly historical asking models."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from marketbridge.structural import StructuralTransfer
from marketbridge.asking import TransferredAsking

FIELDS=['year','odometer','manufacturer','model','condition','cylinders','fuel','title_status','transmission','drive','type','paint_color','state']

def locate(root,role):
    choices=['models/pricing','models/studio-baseline'] if role=='asking' else ['models/carbids','models/tabular']
    for name in choices:
        if (Path(root)/name/'manifest.json').exists():return Path(root)/name
    raise FileNotFoundError(f'No bundled {role} model')

def normalize(frame,role='asking'):
    source=frame.copy()
    for old,new in {'make':'manufacturer','mileage_miles':'odometer','body_type':'type'}.items():
        if new not in source and old in source:source[new]=source[old]
    out=pd.DataFrame(index=source.index)
    for name in FIELDS:out[name]=source[name] if name in source else None
    for name in FIELDS[2:]:out[name]=out[name].astype('string').str.strip().str.lower().replace({'':pd.NA,'unknown':pd.NA})
    out.paint_color=out.paint_color.replace({'gray':'grey'})
    return out

class MarketComparison:
    def __init__(self,root):
        from pricing.inference import PricePipeline
        root=Path(root);self.structural=StructuralTransfer(root/'models/structural-transfer')
        self.asking=PricePipeline(locate(root,'asking'))
        self.enhanced=TransferredAsking(root/'models/transfer-asking',self.structural) if (root/'models/transfer-asking/manifest.json').exists() else None
        self.ids={'structural':self.structural.model_id,'asking':hashlib.sha256((locate(root,'asking')/'lightgbm.txt.gz').read_bytes()).hexdigest()}
    def predict_frame(self,frame):
        from pricing.tabular import features
        output=self.structural.predict_frame(frame)
        fields=normalize(frame);x=features(fields,self.asking.schema)
        raw=self.asking.lgb.predict(x,num_threads=4)*self.asking.manifest.get('target_std',1)+self.asking.manifest.get('target_mean',0)
        price=np.expm1(np.clip(raw,0,np.log1p(1e7)))
        output['asking_price_usd']=price
        output['asking_reference_date']='2021-05-01'
        if self.enhanced:
            # The asking model remains anchored to its 2021 market, even for a 2022+ auction scenario.
            reference=fields.copy();reference['as_of']='2021-05-01'
            output['asking_with_auction_structure_usd']=self.enhanced.predict_frame(reference)
        output['feature_use']='independent_older_auction_structure; no current retail-sale calibration'
        return output
    def explain(self,fields,role):
        if role=='auction':return self.structural.decompose(fields)
        return self.asking.predict('',normalize(pd.DataFrame([fields])).iloc[0].to_dict(),explain=True)

def run(input_file,output_file,root,chunk_size=5000):
    model=MarketComparison(root);destination=Path(output_file);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.partial');count=invalid=0;missing=0
    reader=pd.read_json(input_file,lines=True,chunksize=chunk_size) if str(input_file).endswith('.jsonl') else pd.read_csv(input_file,chunksize=chunk_size)
    try:
        for chunk in reader:
            result=model.predict_frame(chunk)
            result.to_csv(temporary,index=False,mode='w' if count==0 else 'a',header=count==0)
            invalid+=int(result.transported_auction_price_usd.isna().sum());count+=len(result)
        if not count:raise ValueError('Input is empty')
        temporary.replace(destination)
    except Exception:temporary.unlink(missing_ok=True);raise
    report={'rows':count,'transported_auction_estimates':count-invalid,'unavailable_rows':invalid,'model_ids':model.ids,
        'output_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'role':'Age/mileage/CPI-adjusted auction equivalents; 2021 asking estimates remain a separate target. Retail sales uncalibrated.'}
    destination.with_suffix('.meta.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));a=p.parse_args()
    print(json.dumps(run(a.input,a.output,a.root),indent=2))
