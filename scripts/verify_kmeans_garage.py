from pathlib import Path
import json
from scripts.taste_ui_check import exercise_taste_ui
from pricing.taste_ranking import load
def main():
    reports={}
    for k in [10,25]:
        meta,a=load(Path('.'),k)
        assert meta['density_method']=='minibatch-kmeans-soft'
        m=a['memberships']
        assert m.shape==(110832,k) and (m>=0).all()
        assert abs(m.sum(axis=1)-1).max()<1e-5
        reports[str(k)]={'passed':True,'method':'MiniBatchKMeans top-three distance softmax; not HDBSCAN','rows':meta['rows'],'ui':exercise_taste_ui(Path('.'),k,[1+i%4 for i in range(k)])}
    Path('data/taste-garage-mixed/release-check.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports))
if __name__=='__main__':main()
