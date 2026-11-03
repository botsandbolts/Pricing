import pytest
from pricing.same_car_markets import comparison
def fixture():
    return {'fields':{'year':2014},'results':{'ask':{'price':15000,'interval_low':12000,'interval_high':18000,'interval_level':.9},'bid':{'price':10000,'analogue':True}}}
def test_same_car_gap_and_separate_ranges():
    rows,gap=comparison(fixture())
    assert gap['Modelled asking–wholesale gap · USD']==5000
    assert gap['Gap / wholesale estimate · %']==50
    assert rows[0]['Historical range lower · USD']==12000
    assert 'analogue' in rows[1]['Coverage / assumptions']
def test_missing_bid_has_no_gap():
    s=fixture();s['results']['bid']={'error':'Generation not observed'}
    rows,gap=comparison(s)
    assert gap is None and rows[1]['Estimate · USD'] is None
    assert rows[1]['Coverage / assumptions']=='Generation not observed'
def test_negative_gap_not_hidden():
    s=fixture();s['results']['ask']['price']=9000
    assert comparison(s)[1]['Modelled asking–wholesale gap · USD']==-1000
def test_invalid_price_not_used():
    s=fixture();s['results']['bid']['price']=float('nan')
    assert comparison(s)[1] is None
