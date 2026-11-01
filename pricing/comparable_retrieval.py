"""Sparse categorical SVD retrieval. Prices never enter similarity."""
import json
import re
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.decomposition import TruncatedSVD

CATEGORIES=('manufacturer','model','type','fuel','transmission','drive','title_status')
WEIGHTS=np.array([1.4,2.0,1.0,.7,.7,.7,.7],dtype=np.float32)

def clean(value):
    v=re.sub(r'\s+',' ',str(value or '').strip().lower())
    return '__missing__' if v in ('','unknown','nan','none') else v

def categories(rows):return [[clean(r.get(k)) for k in CATEGORIES] for r in rows]

def valid(row):
    try:return clean(row.get('manufacturer'))!='__missing__' and clean(row.get('model'))!='__missing__' and 1900<=float(row['year'])<=2030 and 0<=float(row['odometer'])<=2000000
    except (ValueError,TypeError,KeyError):return False

class ComparableIndex:
    def __init__(self,rows,dimensions=32):
        self.rows=[dict(r) for r in rows if valid(r)]
        if len(self.rows)<3:raise ValueError('At least three valid reference vehicles required')
        self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32)
        matrix=self.encoder.fit_transform(categories(self.rows))
        self.column_weights=np.concatenate([np.full(len(c),WEIGHTS[i],dtype=np.float32) for i,c in enumerate(self.encoder.categories_)])
        matrix=matrix.multiply(self.column_weights).tocsr()
        self.dimensions=min(dimensions,matrix.shape[1]-1,len(self.rows)-1)
        self.svd=TruncatedSVD(n_components=self.dimensions,n_iter=5,random_state=42)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            self.vectors=normalize(self.svd.fit_transform(matrix)).astype(np.float32)
        self.years=np.asarray([r['year'] for r in self.rows],dtype=np.float32)
        self.miles=np.asarray([r['odometer'] for r in self.rows],dtype=np.float32)
        self.description_vectors=None
    def attach_descriptions(self,values):
        values=np.asarray(values,dtype=np.float32)
        if values.ndim!=2 or len(values)!=len(self.rows) or not np.isfinite(values).all():raise ValueError('Invalid description index')
        self.description_vectors=normalize(values)
    def search(self,query,k=5,description_vector=None,exclude_id=None):
        if not valid(query):return []
        q=self.encoder.transform(categories([query])).multiply(self.column_weights).tocsr()
        latent=normalize(self.svd.transform(q))[0]
        similarity=np.clip(self.vectors@latent,-1,1)
        year_gap=np.abs(self.years-float(query['year']))
        mileage_gap=np.abs(self.miles-float(query['odometer']))
        models=np.array([SequenceMatcher(None,clean(query['model']),clean(r['model'])).ratio() for r in self.rows])
        # Retain exact/model-name evidence; SVD must not erase rare identities.
        scores=.45*similarity+.30*models-.15*np.minimum(year_gap/5,2)-.10*np.minimum(mileage_gap/100000,2)
        semantic=None
        if description_vector is not None and self.description_vectors is not None:
            v=np.asarray(description_vector,dtype=np.float32).reshape(1,-1)
            if v.shape[1]!=self.description_vectors.shape[1] or not np.isfinite(v).all():raise ValueError('Invalid query description vector')
            semantic=np.clip(self.description_vectors@normalize(v)[0],-1,1)
            has_text=np.array([bool(r.get('description','').strip()) for r in self.rows])
            scores+=np.where(has_text,.10*semantic,0.)
        # Retrieve by SVD first, then apply explicit matching scores and guardrails.
        candidates=np.argsort(-similarity,kind='stable')[:min(500,len(self.rows))]
        matches=[]
        for i in candidates[np.argsort(-scores[candidates],kind='stable')]:
            r=self.rows[i]
            if clean(r['manufacturer'])!=clean(query['manufacturer']) or year_gap[i]>5 or models[i]<.45:continue
            if exclude_id is not None and r.get('id')==exclude_id:continue
            evidence=[f"Same make · {int(year_gap[i])} model years apart",f"Mileage difference: {int(mileage_gap[i]):,} miles",'Same model label' if clean(r['model'])==clean(query['model']) else 'Partial model-name match — check generation and trim']
            for key in ('type','transmission','title_status'):
                a,b=clean(query.get(key)),clean(r.get(key))
                if a!='__missing__' and b!='__missing__':evidence.append(f"{key.replace('_',' ').capitalize()}: {'matches' if a==b else a+' vs '+b}")
            matches.append({'vehicle':r,'score':float(scores[i]),'evidence':evidence,'semantic_similarity':None if semantic is None or not r.get('description') else float(semantic[i])})
            if len(matches)>=k:break
        return matches

def reference_rows(root,market):
    import gzip
    root=Path(root)
    if market=='ask':
        catalog=root/'data/comparable-catalog/ask.json.gz'
        if catalog.exists():
            return json.load(gzip.open(catalog,'rt'))['rows']
        raw=json.loads((root/'data/description-examples.json').read_text())
        return [dict(x['fields'],id=x['id'],description=x['text'],price_kind='Historical asking advert · price omitted',reference_period='April–May 2021') for x in raw['examples']]
    raw=json.load(gzip.open(root/'data/deal-reference.json.gz','rt'))
    return raw['cases']
