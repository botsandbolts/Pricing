import numpy as np,pandas as pd,pytest
from pricing.lemon_oof import align_oof
def fixture():
    cars=[{'row_id':str(i),'price':100+i} for i in range(5)]
    p=pd.DataFrame({'row_id':[str(i) for i in range(5)],'split':'calibration','fold':range(5),'observed_price':range(100,105),'predicted_price':100.,'residual_log':np.log(np.arange(100,105)/100),'vehicle_group':[str(i) for i in range(5)]})
    return cars,p,{'objective':'survival:aft','folds':5}
def test_exact_alignment_and_sign():
    c,p,e=fixture();pred,r,g=align_oof(c,p.iloc[::-1],e)
    assert np.isclose(r[-1],np.log(104/100)) and g.tolist()==list('01234')
def test_reject_bad_cohort_and_in_sample_predictions():
    c,p,e=fixture();p.loc[0,'fold']=-1
    with pytest.raises(ValueError):align_oof(c,p,e)
    c,p,e=fixture();c[0]['price']=999
    with pytest.raises(ValueError):align_oof(c,p,e)
