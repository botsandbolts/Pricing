"""Regression checks for the current-model paths which previously lost panels/receipts."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def studio(monkeypatch):
    # Keep pricing/graphs real while isolating the automatic text generator.
    from pricing import bart_summary
    class OverviewEngine:
        def encode(self,text):return text.split()
        def decode(self,tokens):return ' '.join(tokens)
        def generate(self,text,forbid_numbers=False):
            return 'A vehicle overview based on the supplied advert and historical models.'
    monkeypatch.setattr(bart_summary,'SummaryEngine',OverviewEngine)
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
    at.session_state['_navigate_to']='🌳 Hierarchical model'
    at.run()
    assert not at.exception
    assert at.get('pills')[0].value=='🚗 Price my car'
    assert at.session_state['hierarchy_sparse_result']['inference_mode']=='hierarchical-only-v1'
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


def test_timeline_stays_visible_after_assumption_changes_and_rerun(studio):
    at=studio
    at.button[0].click().run()
    at.get('pills')[0].set_value('📅 Value over time').run()
    assert not at.exception
    assert len(at.get('plotly_chart'))==1
    at.slider[0].set_value(12000).run()
    assert not at.exception
    assert len(at.get('plotly_chart'))==1
    at.run()
    assert not at.exception
    assert len(at.get('plotly_chart'))==1


def test_vehicle_and_description_survive_navigation(studio):
    at=studio
    for field in at.text_input:
        if field.label=='Make':field.set_value('Ford')
        if field.label=='Model / trim':field.set_value('Focus')
    at.text_area[0].set_value('Service history and new tires.')
    at.button[0].click().run()
    assert not at.exception
    at.get('pills')[0].set_value('📊 Evaluation').run()
    at.get('pills')[0].set_value('🚗 Price my car').run()
    assert not at.exception
    assert next(x.value for x in at.text_input if x.label=='Make')=='Ford'
    assert next(x.value for x in at.text_input if x.label=='Model / trim')=='Focus'
    assert at.text_area[0].value=='Service history and new tires.'
    assert len(at.metric)==2


def test_parked_car_timeline_shortcut_saves_the_valuation_before_navigation(studio):
    at=studio
    at.session_state['_after_estimate_panel']='📅 Value over time'
    at.button[0].click().run()
    assert not at.exception
    assert at.get('pills')[0].value=='📅 Value over time'
    assert len(at.get('plotly_chart'))==1
    assert at.session_state['last_displayed_price'][1]['prediction_kind']=='date_adjusted_scenario'

def test_inspector_prices_own_car_before_five_neighbour_search(studio):
    at=studio
    at.get('pills')[0].set_value('🔎 Listing Inspector').run(timeout=60)
    assert not at.exception
    at.text_area[0].set_value('Full service history. Needs new tires.')
    next(b for b in at.button if b.label=='Price my car & find neighbours').click().run(timeout=90)
    assert not at.exception
    assert any(m.label=='Seller’s asking estimate · USD' for m in at.metric)
    assert any(m.label=='Wholesale auction estimate · USD' for m in at.metric)
    assert any('10 neighbours · exact FAISS cosine search' in c.value for c in at.caption)
    assert all(t.label!='Advert checks & questions' for t in at.tabs)
