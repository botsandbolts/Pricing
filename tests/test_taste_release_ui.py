"""Exercise the same UI path used by the full-data release gate."""
import numpy as np
import pytest

@pytest.mark.parametrize('profile_count',[10,25])
def test_release_check_uses_current_pills_and_test_buttons(monkeypatch,tmp_path,profile_count):
    import pricing.taste_garage_ui as ui
    from scripts.taste_ui_check import exercise_taste_ui
    folder=tmp_path/'data/taste-garage';folder.mkdir(parents=True)
    (folder/'manifest.json').write_text('{}');(folder/'profiles-25.json').write_text('{}')
    n=80;car=dict(year=2015,manufacturer='Ford',model='Focus',price=10000,observed_date='2021-05-01',odometer=50000)
    def catalogue(root,revision,k):
        meta={'rows':n,'density_clusters':k,'noise_fraction':.1,'levels':{},'profiles':[{'prototype':car,'position':i,'name':str(i),'cars':3} for i in range(k)]}
        arrays={'codes':np.zeros((n,1)),'numeric':np.zeros((n,3)),'clusters':np.arange(n)%k,'noise':np.zeros(n,dtype=bool),'prices':np.full(n,10000.),'source_days':np.full(n,18748),'validation':np.arange(n)%5==0,'memberships':np.eye(k)[np.arange(n)%k],'medoid_l2':np.ones((n,k))}
        return meta,arrays
    called=[]
    def rank(*args,**kwargs):
        called.append(list(args[4]));k=len(args[4])
        return {'positions':list(range(k,k+10)),'scores':list(range(10)),'ndcg10':.8,'random_ndcg10':.4,'direct_judgments':k}
    monkeypatch.setattr(ui,'catalogue',catalogue);monkeypatch.setattr(ui,'price_view',lambda *a:8000.)
    monkeypatch.setattr(ui,'transport',lambda *a:None);monkeypatch.setattr(ui,'adjusted_price_features',lambda *a:np.full(n,8000.))
    monkeypatch.setattr(ui,'train_rank',rank);monkeypatch.setattr(ui,'records',lambda root,positions:[dict(car) for _ in positions])
    ratings=[1+i%4 for i in range(profile_count)]
    result=exercise_taste_ui(tmp_path,profile_count,ratings)
    assert called==[ratings] and result['ratings_verified']==profile_count and result['recommendations']==10

def test_release_check_rejects_missing_rating_widgets():
    from scripts.taste_ui_check import rating_controls
    class Empty:
        def get(self,kind):return []
    with pytest.raises(AssertionError,match='Expected 25 rating pills; found 0'):rating_controls(Empty(),25)
