"""Auditable, resumable cross-fitting without access to held-out targets."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold


def cross_fit(row_ids,targets,fit,predict,*,folds=3,seed=42,cache=None,fingerprint='',release=None):
    ids=np.asarray(row_ids,dtype=str);y=np.asarray(targets,dtype=float)
    if len(ids)!=len(y) or len(set(ids))!=len(ids) or not np.isfinite(y).all():
        raise ValueError('Unique row IDs and one finite target per row required')
    if not 2<=folds<=len(ids):raise ValueError('folds must be between 2 and row count')
    root=Path(cache) if cache else None
    if root:root.mkdir(parents=True,exist_ok=True)
    stamp=hashlib.sha256(json.dumps({'ids':ids.tolist(),'targets':y.tolist(),'folds':folds,'seed':seed,'config':fingerprint},sort_keys=True).encode()).hexdigest()
    price=np.full(len(y),np.nan);fold_ids=np.full(len(y),-1)
    for fold,(train_idx,held_idx) in enumerate(KFold(folds,shuffle=True,random_state=seed).split(ids)):
        file=root/f'fold-{fold}.npz' if root else None
        if file and file.exists():
            saved=np.load(file,allow_pickle=False)
            if str(saved['fingerprint'])!=stamp or not np.array_equal(saved['held_idx'],held_idx):
                raise ValueError('OOF checkpoint belongs to different data, targets, split or configuration')
            p=saved['price']
        else:
            # The fitting callback receives only the fitting indices and targets.
            # The prediction callback receives held-out indices and no target values.
            model=fit(train_idx,y[train_idx])
            try:p=np.asarray(predict(model,held_idx),dtype=float)
            finally:
                if release:release(model)
                del model
            if p.shape!=(len(held_idx),) or not np.isfinite(p).all() or (p<0).any():
                raise ValueError('Each held-out row must receive one finite nonnegative dollar price')
            if file:np.savez(file,price=p,held_idx=held_idx,fingerprint=stamp)
        if p.shape!=(len(held_idx),) or not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid cached OOF prices')
        price[held_idx]=p;fold_ids[held_idx]=fold
        print(f'OOF fold {fold+1}/{folds}: fit {len(train_idx):,}, predict {len(held_idx):,}',flush=True)
    if not np.isfinite(price).all() or (fold_ids<0).any():raise ValueError('Incomplete OOF predictions')
    if root:
        (root/'audit.json').write_text(json.dumps({'fingerprint':stamp,'rows':len(ids),'folds':folds,
            'rule':'Each predicted row is excluded from its fitting fold, including target scaling.'},indent=2))
    return price,fold_ids
