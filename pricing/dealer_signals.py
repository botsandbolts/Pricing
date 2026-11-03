"""Original GLiNER candidates: soft sparse identities, no website gating."""
import json,re
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy import sparse

def candidates(entities):
    # Repeated mentions/windows are evidence duplicates, not extra dealer weight.
    found={}
    for e in entities:
        name=str(e.get('name','')).strip()
        score=float(e.get('score',0))
        if not name or not np.isfinite(score):continue
        key=re.sub(r'\s+',' ',name).casefold()
        if key not in found or score>found[key]['score']:
            found[key]={'name':name,'score':float(np.clip(score,0,1)),'label':e.get('label','')}
    return sorted(found.values(),key=lambda x:(-x['score'],x['name']))

@lru_cache(maxsize=1)
def _read(path,revision):
    return json.loads(Path(path).read_text())

def load_dealers(folder):
    p=Path(folder)/'dealers.json'
    return _read(str(p),p.stat().st_mtime_ns) if p.exists() else None

def dealer_matrix(rows):
    keys=sorted({re.sub(r'\s+',' ',e['name']).casefold() for row in rows for e in row})
    lookup={s:i for i,s in enumerate(keys)}
    rr=[];cc=[];vv=[]
    for i,row in enumerate(rows):
        # Divide by sqrt(number of candidates): ambiguous adverts cannot gain
        # unlimited weight. Keep absolute model confidence (not calibrated).
        scale=np.sqrt(max(1,len(row)))
        for e in row:
            rr.append(i);cc.append(lookup[re.sub(r'\s+',' ',e['name']).casefold()]);vv.append(e['score']/scale)
    return sparse.csr_matrix((np.asarray(vv,dtype=np.float32),(rr,cc)),shape=(len(rows),len(keys)))

def dealer_caption(entries):
    if not entries:return 'Dealer: not identified in the advert'
    best=entries[0]
    text=f"Dealer candidate · {best['name']} · NER score {best['score']:.2f}"
    if len(entries)>1:text+=f" · +{len(entries)-1} other candidate(s)"
    return text+' · inferred, unverified'
