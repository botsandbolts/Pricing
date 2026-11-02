from datetime import date
import numpy as np,pandas as pd,pytest
from pricing.mixed_garage import mixed_distance,fit_geometry,transform_geometry,comparison_vectors,AS_OF


def test_mixed_metric_separates_nominal_and_numeric_views():
    a=np.array([1,0,0.],dtype=np.float32)
    b=np.array([0,1,1.],dtype=np.float32)
    assert mixed_distance(a,a,2,1)==pytest.approx(0)
    assert mixed_distance(a,b,2,1)==pytest.approx(1)
    numeric=a.copy();numeric[2]=1
    assert mixed_distance(a,numeric,2,1)==pytest.approx(.5)
    swap=[1,0,2]
    assert mixed_distance(a[swap],b[swap],2,1)==pytest.approx(mixed_distance(a,b,2,1))


class Curve:
    def ratio(self,frame,*args,**kwargs):return np.ones(len(frame))
class Clock:
    def value(self,*args,**kwargs):return 100.,{}
    def reference(self,*args,**kwargs):return 100.,'2021-05'
class Transport:
    curves={'ask':Curve()};clock=Clock()


def test_existing_price_mileage_and_frozen_transform_are_present():
    n=400
    f=pd.DataFrame({'manufacturer':['ford','toyota']*(n//2),'model':['focus','corolla']*(n//2),
                    'year':2010+np.arange(n)%10,'odometer':10000+np.arange(n)*300,
                    'price':5000+np.arange(n)*50,'predicted_text_price':6000+np.arange(n)*30,'date':['2021-05-01']*n})
    geometry,state,prices=fit_geometry(f,Transport())
    assert AS_OF==date(2026,11,1) and state['numeric_dimensions']==7
    assert 'log_mileage' in state['numeric_names'] and 'log_predicted_text_price' in state['numeric_names']
    assert np.isfinite(prices).all()
    np.testing.assert_allclose(geometry,transform_geometry(f,state),atol=1e-6)
    changed=f.iloc[:2].copy();changed.iloc[1]=changed.iloc[0];changed.loc[1,'odometer']=100000
    t=transform_geometry(changed,state)
    assert t[0,state['onehot_dimensions']+1]!=t[1,state['onehot_dimensions']+1]
    assert mixed_distance(t[0],t[1],state['onehot_dimensions'],7)>0
    text_changed=f.iloc[[20,20]].copy().reset_index(drop=True)
    text_changed.loc[1,'predicted_text_price']*=2
    text_geometry=transform_geometry(text_changed,state)
    assert text_geometry[0,state['onehot_dimensions']+5]!=text_geometry[1,state['onehot_dimensions']+5]
    vectors=comparison_vectors(geometry,state)
    np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-6)
    with pytest.raises(ValueError):fit_geometry(f.drop(columns='predicted_text_price'),Transport())
