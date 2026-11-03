"""Group-safe folds within an already retained Lemon training cohort."""
import numpy as np
from sklearn.model_selection import GroupKFold
def fivefold(groups):
    groups=np.asarray(groups)
    if len(np.unique(groups))<5:raise ValueError('Five distinct vehicle groups required')
    splits=list(GroupKFold(5).split(np.arange(len(groups)),groups=groups))
    visits=np.zeros(len(groups),dtype=int)
    for train,held in splits:
        if set(groups[train]) & set(groups[held]):raise ValueError('Vehicle leakage')
        visits[held]+=1
    if not (visits==1).all():raise ValueError('OOF coverage incomplete')
    return splits
