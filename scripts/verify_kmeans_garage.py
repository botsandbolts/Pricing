from pathlib import Path
import json
from scripts.taste_ui_check import exercise_taste_ui
from pricing.taste_ranking import load
from unittest.mock import patch
def check_ui(k):
    import streamlit as st
    folder=Path('data/taste-garage-kmeans')
    with patch('pricing.taste_ranking.active_catalogue_folder',return_value=folder),patch('pricing.taste_garage_ui.active_catalogue_folder',return_value=folder):
        return exercise_taste_ui(Path('.'),k,[1+i%4 for i in range(k)])
def main():
    reports={}
    for k in [10,25]:
        folder=Path('data/taste-garage-kmeans')
        meta,a=load(Path('.'),k,catalogue_path=folder)
        assert meta['density_method']=='minibatch-kmeans-soft'
        assert meta['clustering']['scaler'].startswith('sklearn StandardScaler')
        m=a['memberships']
        assert m.shape==(110832,k) and (m>=0).all()
        assert abs(m.sum(axis=1)-1).max()<1e-5
        reports[str(k)]={'passed':True,'method':'MiniBatchKMeans top-three distance softmax; not HDBSCAN','rows':meta['rows'],'ui':check_ui(k)}
    Path('data/taste-garage-kmeans/release-check.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports))
if __name__=='__main__':main()
