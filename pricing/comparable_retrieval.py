"""Independent SVD retrieval geometries with optional price and description features."""
import json
import re
from pathlib import Path
import numpy as np
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.decomposition import TruncatedSVD

CATEGORIES=('manufacturer','model','type','fuel','transmission','drive','title_status')
WEIGHTS=np.array([1.4,2.0,1.0,.7,.7,.7,.7],dtype=np.float32)

def clean(value):
    v=re.sub(r'\s+',' ',str(value or '').strip().lower())
    return '__missing__' if v in ('','unknown','nan','none') else v

def categories(rows):return [[clean(r.get(k)) for k in CATEGORIES] for r in rows]

def valid(row,allow_missing_miles=False):
    try:return clean(row.get('manufacturer'))!='__missing__' and clean(row.get('model'))!='__missing__' and 1900<=float(row['year'])<=2030 and (allow_missing_miles and row.get('odometer') is None or 0<=float(row['odometer'])<=2000000)
    except (ValueError,TypeError,KeyError):return False

class ComparableIndex:
    def __init__(self,rows,dimensions=32,imputer=None,reference_prices=None,description_prices=None):
        source=list(rows)
        kept=[i for i,r in enumerate(source) if valid(r,allow_missing_miles=imputer is not None)]
        self.rows=[dict(source[i]) for i in kept]
        self.reference_prices=np.asarray(reference_prices,dtype=float)[kept] if reference_prices is not None else None
        self.description_prices=np.asarray(description_prices,dtype=float)[kept] if description_prices is not None else None
        self.use_description_price=self.description_prices is not None and bool(np.any(np.isfinite(self.description_prices)&(self.description_prices>0)))
        self.description_log_median=float(np.nanmedian(np.log(self.description_prices[np.isfinite(self.description_prices)&(self.description_prices>0)]))) if self.use_description_price else None
        self.use_price=self.reference_prices is not None and bool(np.any(np.isfinite(self.reference_prices)&(self.reference_prices>0)))
        self.price_support=int(np.sum(np.isfinite(self.reference_prices)&(self.reference_prices>0))) if self.use_price else 0
        self.price_log_median=float(np.median(np.log(self.reference_prices[np.isfinite(self.reference_prices)&(self.reference_prices>0)]))) if self.use_price else None
        if len(self.rows)<3:raise ValueError('At least three valid reference vehicles required')
        self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32)
        self.imputer=imputer
        cats,nums,self.imputed_fields=self._completed(self.rows)
        matrix=self.encoder.fit_transform(cats)
        self.column_weights=np.concatenate([np.full(len(c),WEIGHTS[i],dtype=np.float32) for i,c in enumerate(self.encoder.categories_)])
        matrix=matrix.multiply(self.column_weights).tocsr()
        from scipy.sparse import hstack,csr_matrix
        from sklearn.preprocessing import StandardScaler
        self.numeric_weights=np.array([.35,.35]+([.70,.10] if self.use_price else [])+([.50,.10] if self.use_description_price else []),dtype=np.float32)
        design=self._numeric_design(nums,self.reference_prices,self.description_prices)
        self.numeric_scaler=StandardScaler().fit(design)
        matrix=hstack([matrix,csr_matrix(self.numeric_weights*self.numeric_scaler.transform(design))],format="csr")
        self.dimensions=min(dimensions,matrix.shape[1]-1,len(self.rows)-1)
        self.svd=TruncatedSVD(n_components=self.dimensions,n_iter=5,random_state=42)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            self.vectors=normalize(self.svd.fit_transform(matrix)).astype(np.float32)
        self.years=nums[:,0]
        self.miles=nums[:,1]
        self.description_vectors=None
    def _completed(self,rows):
        if self.imputer is not None:return self.imputer.transform(rows)
        values=np.asarray([[r['year'],r['odometer']] for r in rows],dtype=np.float32)
        return categories(rows),values,[[] for _ in rows]
    def _numeric_design(self,nums,prices=None,description_prices=None):
        parts=[nums[:,0],np.log1p(nums[:,1])]
        if self.use_price:
            values=np.asarray(prices,dtype=float)
            known=np.isfinite(values)&(values>0)
            logged=np.full(len(values),self.price_log_median)
            logged[known]=np.log(values[known])
            parts.extend([logged,(~known).astype(float)])
        if self.use_description_price:
            values=np.asarray(description_prices,dtype=float);known=np.isfinite(values)&(values>0)
            logged=np.full(len(values),self.description_log_median);logged[known]=np.log(values[known])
            parts.extend([logged,(~known).astype(float)])
        return np.column_stack(parts).astype(np.float32)
    def attach_descriptions(self,values):
        values=np.asarray(values,dtype=np.float32)
        if values.ndim!=2 or len(values)!=len(self.rows) or not np.isfinite(values).all():raise ValueError('Invalid description index')
        self.description_vectors=normalize(values)
    def search(self,query,k=5,description_vector=None,exclude_id=None,market=None):
        """FAISS exact cosine retrieval: five neighbours, no reranking."""
        if not valid(query,allow_missing_miles=self.imputer is not None):return []
        import faiss
        from scipy.sparse import hstack,csr_matrix
        faiss.omp_set_num_threads(2)
        cats,num,_=self._completed([query])
        q=self.encoder.transform(cats).multiply(self.column_weights).tocsr()
        query_price=np.array([query.get('_retrieval_price',np.nan)],dtype=float)
        query_description=np.array([query.get('_retrieval_description_price',np.nan)],dtype=float)
        design=self._numeric_design(num,query_price,query_description)
        q=hstack([q,csr_matrix(self.numeric_weights*self.numeric_scaler.transform(design))],format='csr')
        latent=normalize(self.svd.transform(q))[0]
        if np.linalg.norm(latent)<1e-8:return []
        gaps=np.abs(self.years-float(query['year']))
        eligible=[]
        for i,r in enumerate(self.rows):
            if market is not None and r.get('_market')!=market:continue
            if np.linalg.norm(self.vectors[i])<1e-8:continue
            if exclude_id is not None and r.get('id')==exclude_id:continue
            eligible.append(i)
        if not eligible:return []
        semantic=None
        use_text=description_vector is not None and self.description_vectors is not None
        if use_text:
            vector=np.asarray(description_vector,dtype=np.float32).reshape(1,-1)
            if vector.shape[1]!=self.description_vectors.shape[1] or not np.isfinite(vector).all():raise ValueError('Invalid query description vector')
            vector=normalize(vector)[0];semantic=np.clip(self.description_vectors@vector,-1,1)
        # Exact cosine retrieval: unit-normalized SVD embeddings, optionally
        # concatenated with a separately normalized semantic block, then normalized again.
        points=self.vectors[eligible].copy()
        target=latent.copy()
        if use_text:
            points=np.column_stack([np.sqrt(.8)*points,np.sqrt(.2)*self.description_vectors[eligible]])
            target=np.r_[np.sqrt(.8)*target,np.sqrt(.2)*vector]
        points=np.ascontiguousarray(normalize(points),dtype=np.float32)
        target=np.ascontiguousarray(normalize(target.reshape(1,-1)),dtype=np.float32)
        index=faiss.IndexFlatIP(points.shape[1]);index.add(points)
        similarities,ids=index.search(target,min(5,max(1,int(k)),len(eligible)))
        result=[]
        for similarity,position in zip(similarities[0],ids[0]):
            i=eligible[int(position)];r=self.rows[i]
            evidence=[f"Make: {query['manufacturer']} → {r['manufacturer']} · {int(gaps[i])} model years apart",f"Model: {query['model']} → {r['model']}"]
            miles=r.get('odometer')
            if miles is not None and query.get('odometer') is not None:evidence.append(f"Mileage difference: {int(abs(float(miles)-float(query['odometer']))):,} miles")
            else:evidence.append('Mileage not supplied; a retrieval-only estimate was used')
            for key in ('type','transmission','title_status'):
                a,b=clean(query.get(key)),clean(r.get(key))
                if a!='__missing__' and b!='__missing__':evidence.append(f"{key.replace('_',' ').capitalize()}: {'matches' if a==b else a+' vs '+b}")
            if self.imputed_fields[i]:evidence.append('Estimated for retrieval only: '+', '.join(self.imputed_fields[i])+'. These values are not confirmed by the advert.')
            result.append({'vehicle':r,'score':float(similarity),'faiss_cosine_similarity':float(similarity),'cosine_distance':float(1-np.clip(similarity,-1,1)),'evidence':evidence,'semantic_similarity':None if semantic is None or not r.get('description') else float(semantic[i])})
        return result

def reference_rows(root,market):
    import gzip
    root=Path(root)
    if market=='hybrid':
        return [dict(r,_market='ask') for r in reference_rows(root,'ask')]+[dict(r,_market='bid') for r in reference_rows(root,'bid')]
    if market=='ask':
        catalog=root/'data/comparable-catalog/ask.json.gz'
        if catalog.exists():
            return json.load(gzip.open(catalog,'rt'))['rows']
        raw=json.loads((root/'data/description-examples.json').read_text())
        return [dict(x['fields'],id=x['id'],description=x['text'],price_kind='Historical asking advert · price omitted',reference_period='April–May 2021') for x in raw['examples']]
    raw=json.load(gzip.open(root/'data/deal-reference.json.gz','rt'))
    return raw['cases']
