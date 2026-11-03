import pandas as pd
from scripts.dealer_complete_analysis import shared_support
def test_joint_support_excludes_single_dealer_cells():
    records=[]
    for dealer in ['a','b']:
        for i in range(3):records.append(dict(model='x',state='ca',month='2021-05',year_band=400,mileage_band=2,dealer=dealer))
    records.append(dict(model='rare',state='ca',month='2021-05',year_band=400,mileage_band=2,dealer='a'))
    got=shared_support(pd.DataFrame(records))
    assert len(got)==6 and 'rare' not in set(got.model)
