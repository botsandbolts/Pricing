"""Session recommendation from semantic vectors and inventory associations."""
import re
import numpy as np
from urllib.parse import urlparse
BLOCKED=('craigslist.org','facebook.com','google.com','youtube.com','instagram.com','cars.com','autotrader.com','carfax.com')
def merchant_evidence(text,entities):
    candidates=[]
    for name in entities:
        if re.search(r'(?i)motors|auto sales|automotive|dealership|car sales',name):
            if not re.search(r'(?i)finance|credit|bank|warranty',name):candidates.append(name.strip())
    domains=set()
    for url in re.findall(r'(?i)(?:https?://|www\.)[a-z0-9.-]+(?:/[^\s]*)?',text):
        host=urlparse(url if '://' in url else 'https://'+url).hostname or ''
        host=host.lower().removeprefix('www.')
        if host and not any(host==x or host.endswith('.'+x) for x in BLOCKED):domains.add(host)
    if len(domains)==1 and len(set(candidates))==1:
        return next(iter(domains)),candidates[0],'corroborated candidate'
    return None,None,'unresolved'
def vehicle_text(text,seller=None):
    if seller:text=text.replace(seller,' ')
    text=re.sub(r'https?://\S+|www\.\S+|\S+@\S+',' ',text)
    text=re.sub(r'\b[A-HJ-NPR-Z0-9]{17}\b',' ',text,flags=re.I)
    text=re.sub(r'\$[\d,.]+',' ',text)
    # Exclude repetitive contact/finance lines from the vehicle-semantic branch.
    lines=re.split(r'[\n\r]+|(?<=[.!?])\\s+',text)
    text=' '.join(l for l in lines if not re.search(r'(?i)call us|apply for financing|visit our website|credit approval',l))
    return text[:6000]
def similarity(matrix,queries):
    result=np.empty((len(matrix),len(queries)),dtype=np.float32)
    for start in range(0,len(matrix),2048):
        result[start:start+2048]=np.asarray(matrix[start:start+2048],dtype=np.float32)@np.asarray(matrix[queries],dtype=np.float32).T
    return result
def recommend(semantic,inventory,ratings,type_ids,prices,budget=None,k=10):
    ids=np.array(list(ratings),dtype=int);r=np.array(list(ratings.values()),dtype=float)
    if not len(ids) or not np.any(r>=3):raise ValueError('Like at least one car to build a shortlist.')
    sim=similarity(semantic,ids);inv=similarity(inventory,ids)
    good=r>=3;bad=~good
    positive=np.max(sim[:,good]*((r[good]-2.5)/1.5),axis=1)
    negative=np.max(np.maximum(sim[:,bad],0)*((2.5-r[bad])/1.5),axis=1) if bad.any() else np.zeros(len(semantic))
    available=(np.linalg.norm(inventory,axis=1)>0)
    inventory_score=np.max(inv[:,good],axis=1)
    score=np.where(available,.8*positive+.2*inventory_score,positive)-.35*negative
    allowed=np.isfinite(score);allowed[ids]=False
    if budget is not None:allowed&=np.isfinite(prices)&(prices<=budget)
    pool=np.flatnonzero(allowed);pool=pool[np.argsort(-score[pool],kind='stable')[:500]]
    chosen=[];counts={}
    for pos in pool:
        typ=str(type_ids[pos])
        if counts.get(typ,0)>=2:continue
        chosen.append(int(pos));counts[typ]=counts.get(typ,0)+1
        if len(chosen)==k:break
    return {'positions':chosen,'scores':[float(score[i]) for i in chosen],'semantic':[float(positive[i]) for i in chosen],'inventory':[float(inventory_score[i]) for i in chosen],'dislike_penalty':[float(.35*negative[i]) for i in chosen],'ordering':'hybrid preference score descending'}
def seed_cars(vectors,count=5):
    # Farthest-point diversity in semantic space; not preference clusters.
    chosen=[len(vectors)//2];nearest=np.full(len(vectors),np.inf)
    for _ in range(count-1):
        sim=similarity(vectors,[chosen[-1]])[:,0]
        nearest=np.minimum(nearest,1-sim);nearest[chosen]=-np.inf
        chosen.append(int(np.argmax(nearest)))
    return chosen
