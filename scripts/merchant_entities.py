"""Merchant candidates from named entities; identity remains unverified."""
import json,re,argparse
from pathlib import Path
import pandas as pd,spacy
def extract(frame):
    nlp=spacy.load('en_core_web_sm')
    for row,doc in zip(frame.to_dict('records'),nlp.pipe(frame.text.fillna('').astype(str),batch_size=64)):
        text=str(row.get('text') or '')
        organizations=[{'name':e.text,'start':e.start_char,'end':e.end_char} for e in doc.ents if e.label_=='ORG']
        domains=sorted(set(re.findall(r'(?:https?://|www\.)([a-zA-Z0-9.-]+)',text)))
        yield {'row_id':str(row['row_id']),'organizations':organizations,'domains':domains,'identity_status':'unresolved','warning':'Organization mentions may identify lenders, manufacturers or service providers rather than the seller.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    frame=pd.read_json(a.source,lines=True)
    with open(a.output,'w') as f:
        for r in extract(frame):f.write(json.dumps(r)+'\n')
