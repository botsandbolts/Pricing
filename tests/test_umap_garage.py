import numpy as np
import pandas as pd

def test_clustering_embedding_parameters_follow_guide():
    from scripts.build_umap_garage import make_embedding
    p=make_embedding().get_params()
    assert p['n_neighbors']==30 and p['min_dist']==0 and p['n_components']==5 and p['random_state']==42

def test_incomplete_umap_assets_do_not_replace_working_catalogue(tmp_path):
    from pricing.taste_ranking import active_catalogue_folder
    new=tmp_path/'data/taste-garage-umap';new.mkdir(parents=True)
    for f in ['manifest.json','features.npz','profiles-25.json']:(new/f).write_text('pending')
    assert active_catalogue_folder(tmp_path)==tmp_path/'data/taste-garage'
    (new/'features-25.npz').write_text('ready')
    assert active_catalogue_folder(tmp_path)==new

def test_new_listing_uses_umap_for_density_but_original_features_for_faiss(monkeypatch):
    import hdbscan
    from pricing.taste_ranking import leave_one_out_geometry,predict_new_profiles
    frame=pd.DataFrame({'manufacturer':['a','b']*20,'year':np.arange(40)%15+2000,'odometer':np.arange(40)*2000})
    geometry,_,encoder=leave_one_out_geometry(frame,np.arange(40)*100+2000,return_state=True)
    calls=[]
    class FrozenUMAP:
        def transform(self,x):calls.append(('transform',x.shape));return x[:,:5]
        def fit(self,*args):raise AssertionError('New listing must not refit UMAP')
    def membership(model,x):
        calls.append(('density',x.shape));return np.full((len(x),10),.08)
    monkeypatch.setattr(hdbscan,'membership_vector',membership)
    bundle={'encoder':encoder,'umap':FrozenUMAP(),'hdbscan':object(),'macro':np.arange(10),'medoids':geometry[:10]}
    m,d=predict_new_profiles(frame.iloc[:3],bundle)
    assert calls==[('transform',(3,14)),('density',(3,5))]
    assert m.shape==d.shape==(3,10) and np.isfinite(d).all()
