from pricing.bart_summary import build_valuation_story

def test_history_and_time_backtests_are_part_of_source_story():
    from pathlib import Path
    from marketbridge.time_engine import TimeTransport,today
    from pricing.valuation_story import extend_report
    root=Path(__file__).parents[1]
    car={'year':2015,'manufacturer':'Toyota','model':'Corolla','odometer':95000}
    class Anchor:
        manifest={'feature_names':['age']}
        def predict(self,text,fields):return {'price':12892}
    transport=TimeTransport(root)
    receipt=transport.estimate(Anchor(),'',car,today(),'ask',as_of=today())
    report=extend_report({'valuation_date':str(today())},root,car,{}, {'ask':receipt},transport,None,None)
    assert report['historical_scenarios'][-1]['price_usd']==round(receipt['price'],2)
    assert all(r['mileage']==95000 for r in report['historical_scenarios'])
    story=build_valuation_story(car,'Small dent.',report)
    for phrase in ['2021-05-01','reconstructed same-car','not observed sales','depreciation evidence','Market clock backtests']:
        assert phrase in story
