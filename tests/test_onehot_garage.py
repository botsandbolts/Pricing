import numpy as np
import pandas as pd
import pytest
from pricing.taste_ranking import one_hot_geometry,transform_new_geometry


def reference():
    return pd.DataFrame({'manufacturer':['ford']*6+['toyota']*4+['rare-a','rare-b'],
                         'model':['common']*10+['rare-a','rare-b'],
                         'year':[2010+i for i in range(12)],'odometer':[1000*i for i in range(12)]})


def test_common_categories_are_independent_target_free_indicators():
    frame=reference();g,hot,state=one_hot_geometry(frame,min_count=3,min_fraction=0,return_state=True)
    assert state['categories']=={'manufacturer':['ford','toyota'],'model':['common']}
    assert set(np.unique(hot))<={0,1}
    assert hot[0,:3].tolist()==[1,0,0] and hot[6,:3].tolist()==[0,1,0]
    assert np.array_equal(hot[10],hot[11]) # different rare values share a bucket
    changed=frame.assign(price=np.arange(12)*9999,user='ignored',description='ignored')
    assert np.array_equal(g,one_hot_geometry(changed,min_count=3,min_fraction=0)[0])
    assert g==pytest.approx(transform_new_geometry(frame,state))
    assert not {'stats','prior','smoothing'}.intersection(state)


def test_unseen_categories_use_frozen_rare_bucket_and_numeric_imputation():
    g,_,state=one_hot_geometry(reference(),min_count=3,min_fraction=0,return_state=True)
    future=pd.DataFrame({'manufacturer':['new-brand','rare-a'],'model':['new-model','rare-b'],
                         'year':[None,2015],'odometer':[None,-1]})
    predicted=transform_new_geometry(future,state)
    assert np.isfinite(predicted).all() and predicted.shape==(2,g.shape[1])
    assert np.array_equal(predicted[0,:state['onehot_dimensions']],g[10,:state['onehot_dimensions']])
    assert state['categories']['manufacturer']==['ford','toyota']


def test_threshold_boundaries_constant_fields_and_fraction():
    _,_,state=one_hot_geometry(reference(),min_count=1,min_fraction=.3,return_state=True)
    assert state['threshold']==4 and state['categories']['manufacturer']==['ford','toyota']
    frame=reference().assign(manufacturer='same',model='same')
    g,hot,state=one_hot_geometry(frame,min_count=3,min_fraction=0,return_state=True)
    assert not state['categories'] and hot.shape==(12,0) and g.shape==(12,3)
    with pytest.raises(ValueError):one_hot_geometry(frame,min_count=0)


def test_mileage_changes_numeric_geometry_without_changing_categories():
    frame=reference()
    _,_,state=one_hot_geometry(frame,min_count=3,min_fraction=0,return_state=True)
    cars=pd.DataFrame({'manufacturer':['ford']*3,'model':['common']*3,
                       'year':[2015]*3,'odometer':[1000,100000,None]})
    g=transform_new_geometry(cars,state);offset=state['onehot_dimensions']
    assert np.array_equal(g[0,:offset],g[1,:offset])
    assert g[0,offset+1]!=g[1,offset+1] # log-mileage remains a real clustering feature
    assert g[0,offset+2]==g[1,offset+2]==0 and g[2,offset+2]!=0
    assert state['feature_names'][-3:]==['model_year','log_mileage','mileage_missing']
