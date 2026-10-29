from pricing.vehicle_narrative import narrative
from pricing.bart_summary import summarize_story,CHUNK_TOKENS

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
