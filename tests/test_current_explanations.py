"""Use real installed models: SHAP must reconstruct the same submitted estimates."""
from datetime import date
from pathlib import Path
import numpy as np
import pytest
from pricing.current_explanations import analyse
from pricing.hierarchy_sparse_live import estimate
from pricing.pricing_comparison import selected_receipt
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('side,bayesian',[('ask',False),('bid',False),('ask',True),('bid',True)])
def test_complete_current_graph_suite_reconstructs_dollars_and_native_margin(monkeypatch,side,bayesian):
    from pricing import current_value
    monkeypatch.setattr(current_value,'live_index',lambda:(None,None))
    fields={'year':2015,'manufacturer':'Toyota','model':'Corolla','odometer':95000,'state':'ca','type':'sedan','auction_grade':None}
    text='Full service history. Needs new tires.'
    results,_,_=estimate(ROOT,{**fields,'text':text},date(2026,10,31),True)
    receipt=selected_receipt(results[side],bayesian)
    a=analyse(ROOT,fields,text,date(2026,10,31),bayesian,side,receipt)
    assert a['local']['prediction']==pytest.approx(receipt['price'],rel=1e-5,abs=.05)
    assert a['local']['error']<.01
    assert a['native']['error']<1e-4
    assert np.all(np.isfinite(a['heatmap']['price']))
    assert a['decision']['prediction'][0]==pytest.approx(np.log(receipt['historical_forest_price']),abs=1e-4)
    from pricing import explanation_plots as p
    for fig in [p.waterfall(a['local']),p.beeswarm(a['native']),p.dependence(a['native'],'age'),p.decision(a['decision'],a['comparison']),p.mileage_curve(a['ice'],fields['odometer'],receipt['price']),p.interaction_grid(a['heatmap'])]:
        assert fig.to_json()


def test_timeline_is_fast_uses_saved_anchors_and_accepts_older_receipts(monkeypatch):
    from pricing import current_value
    from pricing.current_tools import timeline_frame
    monkeypatch.setattr(current_value,'live_index',lambda:(None,None))
    fields={'year':2015,'manufacturer':'Toyota','model':'Corolla','odometer':95000,'state':'ca','type':'sedan','auction_grade':None}
    results,transport,_=estimate(ROOT,{**fields,'text':''},date(2026,10,31),True)
    import json
    saved={'signature':json.dumps([fields,'','2026-10-31',True]),'results':results,'date':'2026-10-31'}
    for bayesian in [False,True]:
        frame=timeline_frame(saved,12000,bayesian,transport)
        assert set(frame.Market)=={'Seller asking','Wholesale auction'}
        on_date=frame[frame.Date=='2026-10-31'].set_index('Market')
        for side,label in [('ask','Seller asking'),('bid','Wholesale auction')]:
            assert on_date.loc[label,'USD']==pytest.approx(selected_receipt(results[side],bayesian)['price'],rel=1e-9)
        assert np.isfinite(frame.USD).all()
        assert (frame.Mileage>=0).all()
