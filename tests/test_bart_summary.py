from pricing.vehicle_narrative import narrative
from pricing.bart_summary import summarize_story,CHUNK_TOKENS
from pricing.bart_summary import build_valuation_story

def test_valuation_story_includes_both_targets_time_and_limitations():
    r={'estimated_price_usd':8237,'historical_anchor_usd':12892,'age_multiplier':.623,'market_multiplier':1.026,'vehicle_age_years':11.83,'market_index_status':{'kind':'forecast_monthly_index','month':'2026-10','horizon':1,'method':'damped_holt'},'pooled_depreciation':True}
    story=build_valuation_story({'year':2015,'manufacturer':'Toyota','model':'Corolla'},'Small dent.',{'valuation_date':'2026-10-30','market_estimates':{'ask':r,'bid':{**r,'estimated_price_usd':5760}},'market_source':'BLS/FRED','latest_observed_month':'2026-09'})
    for text in ['8,237.00','5,760.00','asking price','auction sale price','2026-10-30','Small dent','not dealer profit','forecast 1 months','pooled']:assert text in story
    assert 'estimated_price_usd' not in story

def test_missing_auction_never_creates_a_price():
    story=build_valuation_story({},'',{'unavailable_markets':['bid']})
    assert 'bid estimate is unavailable' in story

def test_qualitative_generation_requests_numeric_constraint():
    engine=Engine()
    calls=[]
    def generate(text,forbid_numbers=False):
        calls.append(forbid_numbers)
        return 'A car valuation based on historical data.'
    engine.generate=generate
    result=summarize_story('The estimate is 8237 dollars.',engine,qualitative=True)
    assert calls==[True] and result['status']=='ready'

class Engine:
    def __init__(self):self.calls=[]
    def encode(self,text):return text.split()
    def decode(self,tokens):return ' '.join(tokens)
    def generate(self,text):
        self.calls.append(text)
        return ' '.join(text.split()[-8:])

def test_full_story_and_price_target_exclusion():
    facts,text=narrative(dict(manufacturer='Ford',model='Focus',year=2014,odometer=95000,price=1234567),'New tires.')
    assert 'Ford Focus' in facts and '95,000' in facts and '1234567' not in facts
    engine=Engine();r=summarize_story(facts+' '+text,engine)
    assert engine.calls==[facts+' '+text] and r['status']=='ready'

def test_tail_and_reduce_not_silent_truncation():
    story='word '*(CHUNK_TOKENS+5)+'TAIL DEFECT DISCLOSED'
    engine=Engine();r=summarize_story(story,engine)
    assert r['chunks']==2 and len(engine.calls)==3
    assert 'TAIL DEFECT DISCLOSED' in engine.calls[1] and 'TAIL DEFECT DISCLOSED' in r['summary']

def test_invented_numbers_withheld():
    engine=Engine();engine.generate=lambda text:'It has 12345 miles.'
    assert summarize_story('Vehicle has 95000 miles.',engine)['status']=='review_required'

def test_contacts_and_empty():
    _,text=narrative({},'<b>New brakes.</b> https://example.com a@b.com 212-555-1234')
    assert '<b>' not in text and 'a@b.com' not in text and '212-555-1234' not in text
    assert summarize_story('',Engine())['status']=='empty'
