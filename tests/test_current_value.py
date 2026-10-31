"""Current-date serving, explanation parity and explicit extrapolation contracts."""
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from marketbridge.time_engine import TimeTransport
from pricing.current_value import DateAdjustedAsking, adjustment_chart
from pricing.text_attribution import explain_text, signature

ROOT = Path(__file__).resolve().parents[1]
CAR = {'year':2015, 'manufacturer':'toyota', 'model':'corolla', 'type':'sedan',
       'odometer':95000, 'transmission':'automatic', 'state':'ca'}


class Anchor:
    manifest = {'pipeline':'fixture', 'feature_names':[], 'text_feature':True}
    text_model = True
    def __init__(self):self.calls = 0
    def predict(self, text, fields, explain=False):
        self.calls += 1
        assert fields == CAR
        return {'price':20000+100*len(text.split()), 'predicted_text_price':10000,
                'range':[18000,22000]}


def test_adjustment_uses_actual_vehicle_age_and_separate_market_multiplier():
    t = TimeTransport(ROOT);base = Anchor()
    wrapper = DateAdjustedAsking(base, t, date(2026,10,30), as_of=date(2026,10,30))
    original = base.predict('Clean car', CAR)
    adjusted = wrapper.adjust(original, CAR)
    assert base.calls == 1  # Reuse the existing prediction; no duplicate transformer call.
    r = adjusted['time_adjustment']
    assert r['age'] > 11
    assert r['anchor_age'] < 7
    assert r['market']['kind'] == 'forecast_monthly_index'
    assert r['market']['horizon'] == 1
    assert adjusted['price'] == pytest.approx(original['price']*r['age_factor']*r['market_factor'])
    assert adjusted['price'] != original['price']
    assert original['price'] == 20200 and original['range'] == [18000,22000]
    assert adjusted['range'] is None
    assert adjusted['valuation_date'] == '2026-10-30'
    assert r['target_coverage_validated'] is False


def test_phrase_explanations_reconstruct_the_date_adjusted_receipt():
    t = TimeTransport(ROOT);base = Anchor()
    wrapper = DateAdjustedAsking(base,t,'2026-10-30',as_of='2026-10-30')
    p = explain_text(lambda text:wrapper.predict(text,CAR)['price'], 'Clean car')
    displayed = wrapper.predict('Clean car',CAR)
    assert p['prediction'] == pytest.approx(displayed['price'])
    assert p['base']+sum(p['values']) == pytest.approx(displayed['price'])
    factor = displayed['time_adjustment']['age_factor']*displayed['time_adjustment']['market_factor']
    np.testing.assert_allclose(p['values'], [100*factor,100*factor])
    new_wrapper = DateAdjustedAsking(base,t,'2026-10-30',as_of='2026-10-30')
    args = [wrapper.manifest,CAR,'Clean car','Quick look',wrapper.explanation_identity,displayed['price']]
    other = [new_wrapper.manifest,CAR,'Clean car','Quick look',new_wrapper.explanation_identity,displayed['price']]
    assert signature(*args) == signature(*other)
    tomorrow = DateAdjustedAsking(base,t,'2026-10-31',as_of='2026-10-31')
    assert tomorrow.manifest != wrapper.manifest


def test_stale_clock_and_new_generation_do_not_fall_back_to_historical_prices():
    t = TimeTransport(ROOT);base = Anchor()
    with pytest.raises(ValueError,match='six months'):
        DateAdjustedAsking(base,t,'2028-10-30',as_of='2028-10-30').adjust({'price':20000},CAR)
    with pytest.raises(ValueError,match='generation-analogue'):
        t.estimate(base,'',{**CAR,'year':2020},'2026-10-30','bid',as_of='2026-10-30')
    with pytest.raises(ValueError,match='Only the historical asking'):
        t.estimate(base,'',CAR,'2026-10-30','bid',as_of='2026-10-30',anchor_price=20000)


def test_time_waterfall_preserves_exact_adjusted_total():
    w = DateAdjustedAsking(Anchor(),TimeTransport(ROOT),'2026-10-30',as_of='2026-10-30')
    r = w.predict('Clean car',CAR)['time_adjustment']
    fig = adjustment_chart({'ask':r})
    assert sum(fig.data[0].y) == pytest.approx(r['price'])
    assert list(fig.data[0].measure) == ['absolute','relative','relative','total']
    pair = adjustment_chart({'ask':r,'bid':{**r,'side':'bid'}})
    assert len(pair.data) == 2
    assert pair.data[0].yaxis != pair.data[1].yaxis
    assert pair.data[0].totals.marker.color != pair.data[1].totals.marker.color


def test_today_control_advances_without_a_frozen_date_widget(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import pricing.current_value as current
    monkeypatch.setattr(current,'today',lambda:date(2026,10,30))
    app = AppTest.from_string('''
import streamlit as st
from pricing.current_value import clock_controls
day,allow=clock_controls('fixture')
st.text(str(day))
''').run()
    assert not app.exception
    assert app.text[0].value == '2026-10-30'
    assert app.checkbox[0].value is True
    app.radio[0].set_value('Today · follows the calendar').run()
    assert app.text[0].value == '2026-10-30'
    monkeypatch.setattr(current,'today',lambda:date(2026,10,31))
    app.run()
    assert app.text[0].value == '2026-10-31'
    monkeypatch.setattr(current,'today',lambda:date(2026,11,1))
    app.run()
    assert app.text[0].value == '2026-11-01'
    app.radio[0].set_value('Choose a date').run()
    app.date_input[0].set_value(date(2026,9,12)).run()
    assert app.text[0].value == '2026-09-12'


def test_current_share_caption_and_card_carry_the_actual_date():
    from pricing.deal_tools import caption, card_png, APP_URL
    from PIL import Image
    import io
    text = caption(CAR,15000,reference_date='2026-10-30',time_adjusted=True)
    assert '2026-10-30' in text and 'date-adjusted' in text and 'May 2021' not in text
    png = card_png(CAR,15000,APP_URL,reference_date='2026-10-30',time_adjusted=True)
    assert Image.open(io.BytesIO(png)).size == (1200,630)


def test_month_end_calendar_and_october_nowcast():
    from pricing.current_value import month_end
    assert month_end(date(2026,10,30)) == date(2026,10,31)
    assert month_end(date(2026,11,1)) == date(2026,11,30)
    assert month_end(date(2028,2,4)) == date(2028,2,29)
    t=TimeTransport(ROOT)
    class Bid:
        manifest={'feature_names':[]}
        def predict(self,text,fields):return {'price':20000.}
    r=t.estimate(Bid(),'',{**CAR,'year':2018},'2026-10-31','bid',True,as_of='2026-10-30')
    assert r['date']=='2026-10-31' and r['price']>0
    assert r['generation_analogue'] is True
    assert r['market']['month']=='2026-10'
    assert r['market']['method']=='damped_holt'
    assert r['market']['kind']=='forecast_monthly_index'


def test_forest_time_uses_source_age_and_single_auction_reference():
    import pandas as pd
    from pricing.forest_time import adjust
    from marketbridge.time_engine import age_on
    t=TimeTransport(ROOT)
    row=pd.Series({**CAR,'source_date':'2015-06-01','age':age_on(2015,'2015-06-01'),'base_price':20000.,'corrected_price':22000.})
    r=adjust(t,row,'bid','2026-10-30',95000,as_of='2026-10-30')
    level,_=t.clock.value('2026-10-30',as_of='2026-10-30')
    assert r['market_factor']==pytest.approx(level/t.clock.data.loc['2015-01'])
    assert r['price']==pytest.approx(22000*r['vehicle_factor']*r['market_factor'])
    assert r['forest_price']==pytest.approx(20000*r['vehicle_factor']*r['market_factor'])
    assert r['source_age']<1 and r['age']>11
    same=adjust(t,row,'ask','2015-06-01',95000,as_of='2026-10-30')
    assert same['price']==pytest.approx(22000.)
    november=adjust(t,row,'bid','2026-11-01',95000,as_of='2026-11-01')
    assert november['market']['horizon']==2
    with pytest.raises(ValueError,match='six months'):
        adjust(t,row,'bid','2028-01-01',95000,as_of='2026-10-30')


def test_forest_time_waterfall_keeps_dates_as_category_labels():
    from pricing.forest_time import adjustment_chart
    fig=adjustment_chart({'historical_corrected_price':20000.,'vehicle_factor':.7,'price':15000.,'date':'2026-11-15'})
    assert fig.layout.xaxis.type=='category'
    assert sum(fig.data[0].y)==pytest.approx(15000.)
    assert len(fig.data[0].x)==4
