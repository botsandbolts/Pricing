"""Select 25 density profiles from the frozen HDBSCAN linkage tree; never refit."""
import copy,json,hashlib,gc
from pathlib import Path
import numpy as np,joblib
from pricing.taste_ranking import load,records,faiss_l1

def select_long_model(base,count=25):
    from hdbscan.hdbscan_ import _tree_to_labels
    from hdbscan.flat import select_epsilon,re_init
    tree=base.single_linkage_tree_.to_numpy();model=copy.copy(base)
    for size in [base.min_cluster_size,max(30,base.min_cluster_size//2),max(20,base.min_cluster_size//4),20]:
        output=_tree_to_labels(None,tree,min_cluster_size=size,cluster_selection_method='leaf')
        (model.labels_,model.probabilities_,model.cluster_persistence_,model._condensed_tree,model._single_linkage_tree)=output
        model.min_cluster_size=size;model.cluster_selection_method='leaf';model.cluster_selection_epsilon=0.
        try:epsilon=float(select_epsilon(model.condensed_tree_,'leaf',count))
        except (AssertionError,ValueError):continue
        output=_tree_to_labels(None,tree,min_cluster_size=size,cluster_selection_method='leaf',cluster_selection_epsilon=epsilon)
        if len(np.unique(output[0][output[0]>=0]))!=count:continue
        (model.labels_,model.probabilities_,model.cluster_persistence_,model._condensed_tree,model._single_linkage_tree)=output
        model.cluster_selection_epsilon=epsilon;model.generate_prediction_data()
        re_init(model.prediction_data_,model.condensed_tree_,'leaf',cluster_selection_epsilon=epsilon)
        return model
    raise ValueError('Saved HDBSCAN hierarchy cannot support exactly 25 clusters; no substitute groups released')

def l1_medoid(points,positions):
    from scipy.spatial.distance import cdist
    positions=np.asarray(positions);n=len(points)
    if n<=2000:candidates=np.arange(n)
    else:
        rng=np.random.default_rng(42);near=np.argmin(np.abs(points-np.median(points,axis=0)).sum(axis=1))
        candidates=np.unique(np.r_[near,rng.choice(n,255,replace=False)])
    costs=np.zeros(len(candidates))
    for start in range(0,n,2048):costs+=cdist(points[candidates],points[start:start+2048],metric='cityblock').sum(axis=1)
    return int(positions[candidates[np.argmin(costs)]])

def build_long(root,bundle):
    from hdbscan.flat import all_points_membership_vectors_flat
    root=Path(root);meta,a=load(root);model=select_long_model(bundle['hdbscan']);geometry=model._raw_data
    print('Selected exactly 25 clusters from saved hierarchy, no fit call',flush=True)
    m=np.nan_to_num(all_points_membership_vectors_flat(model),nan=0.,posinf=0.,neginf=0.).astype(np.float32)
    m=np.maximum(m,0);m/=np.maximum(1,m.sum(axis=1,keepdims=True))
    assert m.shape==(len(geometry),25) and (m.sum(axis=0)>0).all()
    prototypes=[]
    for i in range(25):
        members=np.flatnonzero(model.labels_==i);prototypes.append(l1_medoid(geometry[members],members))
    d,_=faiss_l1(geometry,geometry[prototypes]);validation=a['validation'].copy();validation[prototypes]=False
    folder=root/'data/taste-garage';file=folder/'features-25.npz'
    np.savez_compressed(file,memberships=m,clusters=np.argmax(m,axis=1).astype(np.int16),noise=model.labels_<0,validation=validation,medoid_l1=d)
    cars=records(root,prototypes)
    profiles=[{'position':p,'cluster':i,'name':f'Profile {i+1}','cars':int((model.labels_==i).sum()),'prototype':cars[i]} for i,p in enumerate(prototypes)]
    long={'version':3,'profile_count':25,'rows':meta['rows'],'source_sha256':meta['source_sha256'],'profiles':profiles,'density_clusters':25,'noise_fraction':float(np.mean(model.labels_<0)),'selection':'25 flat HDBSCAN leaf clusters selected from the saved single-linkage hierarchy; no graph refit or invented clusters','min_cluster_size':int(model.min_cluster_size),'cluster_selection_epsilon':float(model.cluster_selection_epsilon),'representatives':'Real L1 medoids: exact up to 2000 members; deterministic 256-candidate approximation against all members for larger groups','sha256':{file.name:hashlib.sha256(file.read_bytes()).hexdigest()}}
    (folder/'profiles-25.json').write_text(json.dumps(long,indent=2))
    offline=root/'training-cache/taste-incremental';offline.mkdir(parents=True,exist_ok=True)
    joblib.dump({**bundle,'hdbscan':model,'macro':np.arange(25),'medoids':geometry[prototypes],'prediction_method':'flat'},offline/'frozen-profiles-25.joblib',compress=3)
    return long
