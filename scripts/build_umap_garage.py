"""Full-cohort UMAP→HDBSCAN following the UMAP clustering guide."""
from pathlib import Path
import argparse,gc,hashlib,json,time
import joblib,numpy as np
from pricing.taste_ranking import load,records,faiss_l2,predict_new_profiles
from scripts.build_long_taste import select_long_model,l2_medoid

UMAP_PARAMETERS={'n_neighbors':30,'min_dist':0.,'n_components':5,'metric':'euclidean','n_epochs':200,'random_state':42,'low_memory':True,'n_jobs':1}

def make_embedding():
    import umap
    return umap.UMAP(**UMAP_PARAMETERS)

def profile_summary(labels,noise):
    counts=np.bincount(labels)
    return {'largest_profile_share':float(counts.max()/len(labels)),'profile_sizes':counts.tolist(),'native_noise_fraction':float(np.mean(noise))}

def build(root,bundle_path):
    from hdbscan import HDBSCAN
    from hdbscan.flat import all_points_membership_vectors_flat
    import pandas as pd
    started=time.time();root=Path(root);base_meta,base=load(root,catalogue_path=root/'data/taste-garage')
    frozen=joblib.load(bundle_path);original=np.asarray(frozen['hdbscan']._raw_data,dtype=np.float32)
    assert len(original)==base_meta['rows']==frozen['source_rows']
    folder=root/'data/taste-garage-umap';folder.mkdir(parents=True,exist_ok=True)
    offline=root/'training-cache/umap-hdbscan';offline.mkdir(parents=True,exist_ok=True)
    reducer=make_embedding();print(f'UMAP fitting all {len(original):,} original cars: {UMAP_PARAMETERS}',flush=True)
    embedding=reducer.fit_transform(original).astype(np.float64)
    print('UMAP fitted; fitting HDBSCAN in five dimensions',flush=True)
    density=HDBSCAN(min_cluster_size=500,min_samples=10,cluster_selection_method='leaf',metric='euclidean',prediction_data=True,core_dist_n_jobs=2).fit(embedding)
    report={'source_rows':len(original),'baseline_10':profile_summary(base['clusters'],base['noise']),'embedding':UMAP_PARAMETERS,'versions':{}}
    for k in [10,25]:
        print(f'Selecting {k} density profiles',flush=True);model=select_long_model(density,count=k)
        assert len(np.unique(model.labels_[model.labels_>=0]))==k
        memberships=np.nan_to_num(all_points_membership_vectors_flat(model),nan=0.,posinf=0.,neginf=0.).astype(np.float32)
        memberships=np.maximum(memberships,0);memberships/=np.maximum(1,memberships.sum(axis=1,keepdims=True))
        assert memberships.shape==(len(original),k) and (memberships.sum(axis=0)>0).all() and (memberships.sum(axis=1)>0).mean()>.5
        positions=[]
        for j in range(k):
            ids=np.flatnonzero(model.labels_==j)
            # A real medoid in interpretable original features, within its UMAP density cluster.
            positions.append(l2_medoid(original[ids],ids))
        assert len(set(positions))==k
        distances,_=faiss_l2(original,original[positions]);validation=base['validation'].copy();validation[positions]=False
        arrays={'memberships':memberships,'clusters':np.argmax(memberships,axis=1).astype(np.int16),'noise':model.labels_<0,'validation':validation,'medoid_l2':distances}
        if k==10:arrays.update({key:base[key] for key in ['codes','numeric','prices','source_days']})
        file=folder/('features.npz' if k==10 else 'features-25.npz');np.savez_compressed(file,**arrays)
        cars=records(root,positions)
        profiles=[{'position':p,'cluster':j,'name':f'Profile {j+1}','cars':int((model.labels_==j).sum()),'prototype':cars[j]} for j,p in enumerate(positions)]
        meta={'version':4,'profile_count':k,'rows':len(original),'source_sha256':base_meta['source_sha256'],'profiles':profiles,'density_clusters':k,'noise_fraction':float(np.mean(model.labels_<0)),'embedding':dict(UMAP_PARAMETERS),'method_description':'Smoothed leave-one-out category encoding and standardized numerical features → 5D UMAP (30 neighbours, min_dist=0) → HDBSCAN. The 10- and 25-car tests select separate cuts of the density hierarchy. Medoids and FAISS L2 explanations use original scaled features; the embedding is not used as a substitute for interpretable vehicle distance.','representatives':'Real non-noise cars; original-feature L2 medoid objective, exact up to 2000 members and deterministic 256-candidate approximation for larger groups','reference_database':'../taste-garage/cars.sqlite; canonical row positions unchanged','sha256':{file.name:hashlib.sha256(file.read_bytes()).hexdigest()}}
        if k==10:meta['levels']=base_meta['levels']
        (folder/('manifest.json' if k==10 else 'profiles-25.json')).write_text(json.dumps(meta,indent=2))
        bundle={'encoder':frozen['encoder'],'umap':reducer,'hdbscan':model,'macro':np.arange(k),'medoids':original[positions],'source_rows':len(original),'source_sha256':base_meta['source_sha256'],'prediction_method':'flat','embedding_parameters':UMAP_PARAMETERS}
        joblib.dump(bundle,offline/f'frozen-profiles-{k}.joblib',compress=3)
        new_m,new_d=predict_new_profiles(pd.DataFrame(cars[:3]),bundle)
        assert new_m.shape==new_d.shape==(3,k) and np.isfinite(new_m).all() and np.isfinite(new_d).all()
        report['versions'][str(k)]=profile_summary(arrays['clusters'],arrays['noise'])
        print(f'Exported {k} profiles, membership shape {memberships.shape}',flush=True)
        del memberships,arrays,model,bundle;gc.collect()
    report['build_seconds']=time.time()-started
    (folder/'cluster-comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--bundle',required=True);a=p.parse_args();build(a.root,a.bundle)
