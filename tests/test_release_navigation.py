"""Regression checks for the current-model paths which previously lost panels/receipts."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def studio(monkeypatch):
    from pricing import current_value
    monkeypatch.setattr(current_value,'live_index',lambda:(None,'Offline regression uses saved dated index.'))
    return AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()

def test_current_estimate_drivers_sharing_and_bayesian_route(studio):
    at=studio
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric)==2
    for button in at.button:
        if button.label=='Show current model price drivers':
            button.click().run();break
    assert not at.exception
    assert len(at.get('plotly_chart'))>=3
    for button in at.button:
        if button.label.startswith('📣 Share my price'):
            button.click().run();break
    assert not at.exception
    assert at.get('pills')[0].value=='🕵️ Deal Detective'
    assert '⬇ Download image card · PNG' in [x.label for x in at.get('download_button')]
    at.get('pills')[0].set_value('🌳 Hierarchical model').run()
    assert not at.exception
    assert any(m.label=='Bayesian MAE' for m in at.metric)
    assert any('Historical 90% range' in m.value for m in at.markdown)

@pytest.mark.parametrize('page',['📊 Evaluation','🧹 Outlier lab','🧠 The science','📅 Value over time','🧪 Model lab','↔ Two-market garage'])
def test_navigation_is_lazy_and_all_panels_load(studio,page):
    studio.get('pills')[0].set_value(page).run()
    assert not studio.exception
    assert not any(x.label=='Estimate asking + auction prices' for x in studio.button)

def test_regular_waterfall_uses_regular_anchor():
    from pricing.pricing_comparison import selected_receipt
    from pricing.forest_time import adjustment_chart
    original={'price':120.,'forest_price':100.,'historical_corrected_price':60.,'historical_forest_price':50.,'vehicle_factor':1.5,'date':'2026-11-01'}
    chosen=selected_receipt(original,bayesian=False)
    chart=adjustment_chart(chosen)
    assert chart.data[0].y[0]==50.
    assert sum(chart.data[0].y[:3])==pytest.approx(100.)
    assert original['price']==120.
