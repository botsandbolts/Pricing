import numpy as np
from pricing.lemon_folds import fivefold
def test_retained_quartile_not_removed_again():
    original=np.arange(100);retained=original[25:]
    folds=fivefold(retained)
    assert len(folds)==5
    seen=[]
    for train,held in folds:
        assert len(train)==60 and len(held)==15
        assert not set(retained[train]) & set(retained[held])
        seen.extend(retained[held])
    assert sorted(seen)==retained.tolist()
def test_repeated_vehicle_never_crosses_fold():
    groups=np.repeat(np.arange(20),3)
    for train,held in fivefold(groups):
        assert not set(groups[train]) & set(groups[held])
