"""Offline NER, conservative merchant resolution and semantic/inventory indexes."""
import json,re,hashlib,sqlite3,zlib
from pathlib import Path
import numpy as np,pandas as pd,spacy
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from pricing.merchant_reco import merchant_evidence,vehicle_text,recommend,seed_cars
def main():
    source=Path('training-cache/lemon-mixed/cohort.jsonl')
    frame=pd.read_json(source,lines=True).reset_index(drop=True)
    folder=Path('data/merchant-discovery');folder.mkdir(parents=True,exist_ok=True)
    nlp=spacy.load('en_core_web_sm',disable=['parser','tagger','lemmatizer','attribute_ruler'])
    texts=frame.text.fillna('').astype(str).tolist();entities=[]
    for text,doc in zip(texts,nlp.pipe(texts,batch_size=64)):
        entities.append(merchant_evidence(text,[e.text for e in doc.ents if e.label_=='ORG']))
    # Do not merge a domain carrying conflicting business identities.
    names={}
    for domain,name,status in entities:
        if domain:names.setdefault(domain,set()).add(re.sub(r'[^a-z0-9]','',re.sub(r'\\b(llc|inc|ltd|corporation)\\b','',name.lower())))
    entities=[e if e[0] and len(names[e[0]])==1 else (None,None,'unresolved') for e in entities]
    clean=[(' '.join(str(row.get(k) or '') for k in ['manufacturer','model','type','fuel','transmission'])+'. '+vehicle_text(t,e[1])) for row,t,e in zip(frame.to_dict('records'),texts,entities)]
    encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    encoder.max_seq_length=256
    embeddings=encoder.encode(clean,batch_size=64,normalize_embeddings=True,show_progress_bar=True).astype(np.float32)
    projector=TruncatedSVD(n_components=96,random_state=42)
    semantic=normalize(projector.fit_transform(embeddings)).astype(np.float16)
    del embeddings,encoder,nlp
    # Shared feature/type columns connect inventories across merchants.
    records=[]
    for row in frame.to_dict('records'):
        make=str(row.get('manufacturer') or 'unknown').lower();model=str(row.get('model') or 'unknown').lower();body=str(row.get('type') or 'unknown').lower()
        records.append({'make='+make:1.,'body='+body:1.,'model='+make+'|'+model:1.})
    vectorizer=DictVectorizer();features=vectorizer.fit_transform(records)
    keys=sorted({e[0] for e in entities if e[0]});lookup={key:i for i,key in enumerate(keys)}
    linked=[i for i,e in enumerate(entities) if e[0]]
    inventory=np.zeros((len(frame),32),dtype=np.float32)
    if len(keys)>=3:
        assignment=csr_matrix((np.ones(len(linked)),([lookup[entities[i][0]] for i in linked],linked)),shape=(len(keys),len(frame)))
        counts=assignment@features
        # Dealer size normalization and inverse dealer frequency.
        idf=np.log((1+len(keys))/(1+np.asarray((counts>0).sum(axis=0)).ravel()))+1
        matrix=normalize(counts.multiply(idf),axis=1)
        svd=TruncatedSVD(n_components=min(32,min(matrix.shape)-1),random_state=42).fit(matrix)
        mapped=normalize(features.multiply(idf)@svd.components_.T)
        inventory[:,:mapped.shape[1]]=mapped
    np.save(folder/'semantic.npy',semantic);np.save(folder/'inventory.npy',inventory.astype(np.float16))
    db=folder/'cars.sqlite';db.unlink(missing_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE cars(position INTEGER PRIMARY KEY,payload BLOB NOT NULL)')
        for i,row in enumerate(frame.to_dict('records')):
            safe={k:None if pd.isna(v) else v for k,v in row.items() if k not in ('text','description')}
            safe.update(description=clean[i][:600],merchant_status=entities[i][2],merchant_name=entities[i][1])
            conn.execute('INSERT INTO cars VALUES(?,?)',(i,zlib.compress(json.dumps(safe,default=str).encode())))
    types=[str(x.get('manufacturer'))+'|'+str(x.get('model')) for x in frame.to_dict('records')]
    prices=frame.price.to_numpy(float)
    seeds=seed_cars(semantic);checks={}
    for liked in seeds:
        ratings={i:(4 if i==liked else 1) for i in seeds}
        result=recommend(semantic,inventory,ratings,types,prices)
        assert len(set(result['positions']))==10 and not set(result['positions']).intersection(seeds)
        assert all(a>=b for a,b in zip(result['scores'],result['scores'][1:]))
        checks[str(liked)]={'recommendations':10}
    meta={'version':1,'rows':len(frame),'merchant_candidates':len(keys),'linked_rows':len(linked),'identity_status':'inferred, not verified legal identity','semantic_model':'all-MiniLM-L6-v2 + 96D projection','source_sha256':hashlib.file_digest(open(source,'rb'),'sha256').hexdigest(),'seeds':seeds,'types':types,'prices':prices.tolist(),'mechanical_checks':checks,'relevance_evaluation':'No independent user relevance benchmark; merchant identity precision not yet manually audited'}
    (folder/'manifest.json').write_text(json.dumps(meta))
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string(chr(10).join(["from pathlib import Path","from pricing.merchant_discovery_ui import render","render(Path('.'))"])).run(timeout=120)
    assert not app.exception
    for control in app.get('pills'):
        if str(control.key).startswith('merchant_rating_'):control.set_value(4 if str(control.key).endswith('_'+str(seeds[0])) else 1)
    app.run(timeout=120)
    button=next(b for b in app.button if b.key=='merchant_find')
    assert not button.disabled
    button.click().run(timeout=120)
    assert not app.exception
    assert any('Your shortlist' in m.value for m in app.markdown)
    meta['ui_check']='Five real ratings submitted; shortlist rendered without exceptions'
    (folder/'manifest.json').write_text(json.dumps(meta))
    print(json.dumps({k:v for k,v in meta.items() if k not in ('types','prices')},indent=2))
if __name__=='__main__':main()
