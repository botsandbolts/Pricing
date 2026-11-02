import json,sqlite3,zlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pricing.taste_ranking import CATEGORICAL,encode,features,ndcg10,train_rank

def sample():
    rows=[]
    for i in range(400):
        c=i%10
        rows.append(dict(manufacturer='make'+str(c),model='model'+str(c),type='suv' if c%2 else 'sedan',year=2000+c,odometer=10000+c*1000+i,fuel='gas',price=1000+i,cluster=c,user='excluded',description='excluded'))
    return pd.DataFrame(rows)

def test_vehicle_only_native_categories():
    frame=sample();codes,nums,levels=encode(frame);x=features(codes,nums,levels)
    assert all(str(x[c].dtype)=='category' for c in CATEGORICAL)
    assert not {'price','description','cluster','user','qid'}.intersection(x.columns)
    changed=frame.copy();changed.price*=100;changed.user='another';a,b,l=encode(changed)
    assert np.array_equal(a,codes) and np.array_equal(b,nums) and l==levels

def test_ndcg_matches_exponential_gain_definition():
    assert ndcg10([3,2,1,0],[4,3,2,1])==pytest.approx(1)
    assert 0<ndcg10([3,2,1,0],[1,2,3,4])<1
    assert ndcg10([0,0],[2,1]) is None

def test_ranker_learns_without_duplicate_or_user_rows():
    frame=sample();codes,nums,levels=encode(frame);clusters=np.arange(len(frame))%10
    validation=np.random.default_rng(42).random(len(frame))<.2
    result=train_rank(codes,nums,levels,clusters,[1,2,3,4,1,2,3,4,2,3],list(range(10)),validation,rounds=20,prices=frame.price.to_numpy(),adjusted_prices=frame.price.to_numpy()*.8)
    assert result['final_rows']==400 and result['direct_judgments']==10
    assert len(set(result['positions']))==10
    assert not set(result['positions']).intersection(range(10))
    assert 0<=result['ndcg10']<=1
    assert not {'cluster','user','qid'}.intersection(result['feature_importance'])
    assert 'log_advertised_price' in result['feature_importance'] and 'log_date_adjusted_price' in result['feature_importance']
    with pytest.raises(ValueError):train_rank(codes,nums,levels,clusters,[2]*10,list(range(10)),validation,rounds=3)

def test_hdbscan_real_prototypes_and_noise_are_distinguished():
    from scripts.build_taste_garage import density_profiles
    rng=np.random.default_rng(42)
    centres=rng.normal(0,30,(12,5));points=np.concatenate([rng.normal(c,.08,(80,5)) for c in centres])
    clusters,noise,prototypes,attempts,native,memberships=density_profiles(points)
    assert len(prototypes)==len(set(prototypes))==10
    assert set(clusters)==set(range(10))
    assert all(not noise[p] for p in prototypes)
    assert native>=10
    assert memberships.shape==(len(points),10)
    assert np.isfinite(memberships).all() and (memberships>=0).all()
    assert (memberships.sum(axis=1)<=1.0001).all()
    assert all(clusters[p]==i for i,p in enumerate(prototypes))

def test_rating_ui_and_result_staleness(monkeypatch,tmp_path):
    import pricing.taste_garage_ui as ui
    from streamlit.testing.v1 import AppTest
    folder=tmp_path/'data/taste-garage';folder.mkdir(parents=True);(folder/'manifest.json').write_text('{}')
    car=dict(year=2015,manufacturer='Ford',model='Focus',price=10000,observed_date='2021-05-01',odometer=50000)
    meta={'rows':400,'density_clusters':12,'noise_fraction':.1,'profiles':[{'prototype':car,'position':i,'name':f'Profile {i}','cars':40} for i in range(10)]}
    monkeypatch.setattr(ui,'catalogue',lambda *a:(meta,{}));monkeypatch.setattr(ui,'price_view',lambda *a:8000.)
    at=AppTest.from_string(f"from pathlib import Path\nfrom pricing.taste_garage_ui import render\nrender(Path({str(tmp_path)!r}))").run(timeout=30)
    assert not at.exception
    assert next(b for b in at.button if b.label=='Find my ten cars →').disabled
    radios=[r for r in at.get('pills') if r.label.startswith('Your reaction')];assert len(radios)==10
    for r in radios:r.set_value(2)
    at.run();assert not at.exception
    assert next(b for b in at.button if b.label=='Find my ten cars →').disabled


def test_adjusted_price_features_allow_native_categories_and_keep_missing_facts():
    from datetime import date
    from pricing.taste_ranking import adjusted_price_features
    frame=sample();codes,nums,levels=encode(frame)
    class Curve:
        def ratio(self,frame,*args,**kwargs):
            frame.manufacturer.fillna('unknown')
            return np.full(len(frame),.5)
    class Clock:
        def value(self,*args,**kwargs):return 200.,{}
        def reference(self,*args,**kwargs):return 100.,'2021-05'
    class Transport:
        curves={'ask':Curve()};clock=Clock()
    days=np.full(len(frame),np.datetime64('2021-05-01').astype(int))
    nums[0,2]=1
    price=frame.price.to_numpy(dtype=float)
    result=adjusted_price_features(codes,nums,levels,price,days,date(2026,11,1),Transport())
    assert np.isnan(result[0])
    assert result[1:]==pytest.approx(price[1:])
    assert price[0]==1000


def test_leave_one_out_excludes_own_target_and_uses_dense_numeric_geometry():
    from pricing.taste_ranking import leave_one_out_geometry
    frame=sample().iloc[:20].copy();frame['manufacturer']=['unique']+['shared']*19
    prices=frame.price.to_numpy(dtype=float)
    geometry,a=leave_one_out_geometry(frame,prices)
    changed=prices.copy();changed[0]=999999
    _,b=leave_one_out_geometry(frame,changed)
    assert a[0]==pytest.approx(b[0]) # neither category sum nor prior includes this row
    assert geometry.shape==(20,len(CATEGORICAL)+3) and np.isfinite(geometry).all()
    assert a[0,0]==pytest.approx(np.log(prices[1:]).mean())
    assert not np.allclose(a[1:],b[1:])


def test_constant_category_leave_out_artifacts_are_not_amplified_by_scaling():
    from pricing.taste_ranking import leave_one_out_geometry
    frame=sample().iloc[:100].copy()
    for key in CATEGORICAL:frame[key]='constant'
    matrix,_=leave_one_out_geometry(frame,np.linspace(1000,30000,len(frame)))
    assert np.all(matrix[:,:len(CATEGORICAL)]==0)



def test_soft_memberships_train_without_row_expansion():
    frame=sample();codes,nums,levels=encode(frame)
    m=np.full((len(frame),10),.01,dtype=float)
    m[np.arange(len(frame)),np.arange(len(frame))%10]=.7
    valid=np.arange(len(frame))%5==0
    result=train_rank(codes,nums,levels,np.argmax(m,axis=1),[1,2,3,4,1,2,3,4,2,3],list(range(10)),valid,rounds=12,prices=frame.price.to_numpy(),memberships=m)
    assert result['soft_memberships'] and result['final_rows']==len(frame)
    assert result['score_range']>1e-7
    assert len(result['positions'])==10 and 0<=result['ndcg10']<=1
    assert not any('membership' in k for k in result['feature_importance'])

def test_medoid_is_actual_minimum_distance_member():
    from scripts.build_taste_garage import medoid_position
    points=np.array([[0.,0.],[1.,0.],[4.,0.]])
    assert medoid_position(points,[10,20,30])==20


def test_new_listing_transform_is_frozen_and_target_free():
    from pricing.taste_ranking import leave_one_out_geometry,transform_new_geometry
    frame=sample();_,_,state=leave_one_out_geometry(frame,frame.price.to_numpy(),return_state=True)
    new=frame.iloc[:4].copy();a=transform_new_geometry(new,state)
    new.price=99999999
    assert np.array_equal(a,transform_new_geometry(new,state))
    new.manufacturer='unseen';assert np.isfinite(transform_new_geometry(new,state)).all()


def test_frozen_hdbscan_predicts_new_rows_without_refit(monkeypatch):
    import hdbscan
    from pricing.taste_ranking import leave_one_out_geometry,predict_new_profiles
    frame=sample();geometry,_,encoder=leave_one_out_geometry(frame,frame.price.to_numpy(),return_state=True)
    model=hdbscan.HDBSCAN(min_cluster_size=12,min_samples=3,prediction_data=True).fit(geometry)
    k=len(set(model.labels_)-{-1});assert k>=2
    bundle={'encoder':encoder,'hdbscan':model,'macro':np.arange(k)%10,'medoids':geometry[:10]}
    monkeypatch.setattr(model,'fit',lambda *a,**k: (_ for _ in ()).throw(AssertionError('Refit forbidden')))
    m,d=predict_new_profiles(frame.iloc[:3],bundle)
    assert m.shape==d.shape==(3,10)
    assert np.isfinite(m).all() and (m.sum(axis=1)<=1.0001).all()


def test_faiss_l2_matches_direct_calculation_and_order():
    from pricing.taste_ranking import faiss_l2
    q=np.array([[1.,2.,3.],[0.,0.,0.]],dtype=np.float32)
    r=np.array([[3.,4.,5.],[1.,2.,3.],[5.,6.,7.]],dtype=np.float32)
    d,order=faiss_l2(q,r)
    expected=np.sqrt(np.sum((q[:,None,:]-r[None,:,:])**2,axis=2))
    np.testing.assert_allclose(d,expected,atol=1e-5)
    assert order.tolist()==np.argsort(expected,axis=1).tolist()


def test_25_ratings_and_soft_memberships_train_without_expansion():
    frame=sample();codes,nums,levels=encode(frame);n=len(frame)
    labels=np.arange(n)%25;m=np.full((n,25),.001);m[np.arange(n),labels]=.95
    result=train_rank(codes,nums,levels,labels,[1+i%4 for i in range(25)],list(range(25)),np.arange(n)%5==0,rounds=20,prices=frame.price.to_numpy(),memberships=m)
    assert result['direct_judgments']==25 and result['final_rows']==n and len(result['positions'])==10
    assert not set(result['positions']).intersection(range(25))


def test_25_clusters_reuse_saved_hierarchy_without_fit(monkeypatch):
    import hdbscan
    from scripts.build_long_taste import select_long_model
    rng=np.random.default_rng(19)
    centres=rng.normal(0,30,(30,5));x=np.concatenate([rng.normal(c,.05,(35,5)) for c in centres])
    model=hdbscan.HDBSCAN(min_cluster_size=15,min_samples=3,prediction_data=True).fit(x)
    monkeypatch.setattr(model,'fit',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Refit forbidden')))
    selected=select_long_model(model)
    assert len(set(selected.labels_)-{-1})==25
    from hdbscan.flat import membership_vector_flat
    m=membership_vector_flat(selected,x[:4]);assert m.shape==(4,25) and np.isfinite(m).all()


def test_short_long_ui_keeps_ratings_separate(monkeypatch,tmp_path):
    import pricing.taste_garage_ui as ui
    from streamlit.testing.v1 import AppTest
    folder=tmp_path/'data/taste-garage';folder.mkdir(parents=True)
    (folder/'manifest.json').write_text('{}');(folder/'profiles-25.json').write_text('{}')
    car=dict(year=2015,manufacturer='Ford',model='Focus',price=10000,observed_date='2021-05-01',odometer=50000)
    def cat(root,revision,k):return {'rows':400,'density_clusters':k,'noise_fraction':.1,'profiles':[{'prototype':car,'position':i,'name':str(i),'cars':16} for i in range(k)]},{}
    monkeypatch.setattr(ui,'catalogue',cat);monkeypatch.setattr(ui,'price_view',lambda *a:8000.)
    at=AppTest.from_string(f"from pathlib import Path\nfrom pricing.taste_garage_ui import render\nrender(Path({str(tmp_path)!r}))").run()
    [r for r in at.get('pills') if r.label.startswith('Your reaction')][0].set_value(4).run()
    next(b for b in at.button if b.key=='taste_choose_25').click().run()
    assert not at.exception and len([r for r in at.get('pills') if r.label.startswith('Your reaction')])==25
    assert all(r.value is None for r in at.get('pills') if r.label.startswith('Your reaction'))
    next(b for b in at.button if b.key=='taste_choose_10').click().run()
    assert not at.exception and [r for r in at.get('pills') if r.label.startswith('Your reaction')][0].value==4


def test_negative_ratings_penalize_and_noise_is_neutral():
    from pricing.taste_ranking import preference_evidence
    ratings=[4,1]+[2]*8
    m=np.zeros((3,10));m[0,0]=.9;m[1,1]=.9
    e=preference_evidence(m,ratings)
    assert e['signed'][0]>0 and e['signed'][1]<0 and e['signed'][2]==0
    assert e['eligible'].tolist()==[True,False,False]

def test_nearest_disliked_car_is_excluded_even_with_positive_density_evidence():
    from pricing.taste_ranking import preference_evidence
    m=np.zeros((2,10));m[:,0]=1
    distances=np.full((2,10),10.);distances[0,:2]=[2.,.1];distances[1,:2]=[.1,2.]
    e=preference_evidence(m,[4,1]+[2]*8,distances)
    assert e['signed'][0]>0 and not e['eligible'][0]
    assert e['eligible'][1]


def test_cosine_search_and_zero_vector_neutrality():
    from pricing.taste_ranking import faiss_cosine
    q=np.array([[1.,2.],[0.,0.],[-1.,-2.]],dtype=np.float32)
    r=np.array([[2.,4.],[2.,-1.],[-2.,-4.]],dtype=np.float32)
    before=q.copy();d,order=faiss_cosine(q,r)
    qn=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
    rn=r/np.linalg.norm(r,axis=1,keepdims=True)
    np.testing.assert_allclose(d,1-np.clip(qn@rn.T,-1,1),atol=1e-6)
    np.testing.assert_array_equal(q,before)
    assert order[0,0]==0 and order[2,0]==2
    np.testing.assert_allclose(d[1],1)

def test_retired_umap_catalogue_cannot_activate(tmp_path):
    from pricing.taste_ranking import active_catalogue_folder
    path=tmp_path/'data/taste-garage-umap';path.mkdir(parents=True)
    for name in ['manifest.json','features.npz','profiles-25.json','features-25.npz']:(path/name).write_text('old')
    assert active_catalogue_folder(tmp_path)==tmp_path/'data/taste-garage'


def test_reciprocal_standardisation_direction_and_scale():
    from pricing.taste_ranking import reciprocal_rating_scores
    r=np.array([1,2,3,4]*3)
    result=reciprocal_rating_scores(r);inv=1/r
    np.testing.assert_allclose(result,5/12-inv)
    assert np.all(np.diff(result[:4])>0)
    assert result[0]<result[1]<0<result[2]<result[3]
    np.testing.assert_allclose(reciprocal_rating_scores([4]*10),np.full(10,1/6))

def test_reciprocal_evidence_prefers_four_over_one():
    from pricing.taste_ranking import preference_evidence
    ratings=[1,2,3,4,1,2,3,4,1,4]
    e=preference_evidence(np.eye(10),ratings)
    assert e['signed'][3]>e['signed'][2]>e['signed'][1]>e['signed'][0]


def test_rmse_uses_raw_vector_magnitude_without_normalization():
    from pricing.taste_ranking import faiss_rmse
    q=np.array([[1.,2.],[10.,20.]],dtype=np.float32)
    r=np.array([[1.,2.],[2.,4.]],dtype=np.float32)
    d,order=faiss_rmse(q,r)
    np.testing.assert_allclose(d,np.sqrt(np.mean((q[:,None,:]-r[None,:,:])**2,axis=2)),atol=1e-5)
    assert d[0,0]==0 and d[1,0]>10
    assert order[0,0]==0 and order[1,0]==1


def test_presentation_lowest_liked_rmse_first_and_stable_ties():
    from pricing.taste_ranking import closest_liked_first
    distances=np.array([[.4,.9],[.1,.8],[.3,.7],[.1,.6]])
    ordered=closest_liked_first([0,1,2,3],distances,[4,1])
    assert ordered.tolist()==[1,3,2,0]
    assert set(ordered)=={0,1,2,3}
