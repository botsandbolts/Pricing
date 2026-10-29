import json
from pricing.vehicle_summary import (vehicle_story,summarize_vehicle,summarize_vehicles,SummaryConfig,input_fingerprint)


class FakeEngine:
    name='test'
    def __init__(self,answer='The seller reports a serviced Ford Focus.'):
        self.answer=answer;self.requests=[]
    def tokenize(self,text):return list(text)
    def detokenize(self,tokens):return ''.join(tokens)
    def complete_batch(self,requests,max_new_tokens):
        self.requests.extend(requests)
        return [self.answer]*len(requests)


def test_story_is_narrative_and_price_targets_never_enter():
    fields=dict(manufacturer='Ford',model='Focus',year=2014.,odometer=95000.,fuel='gas',price=1234567,predicted_text_price=999999,residual=555555)
    story=vehicle_story(fields,'Runs well. Price $12345. Call 212-555-1234. VIN 1HGCM82633A004352.')
    assert 'The make is Ford.' in story and 'The mileage in miles is 95000.' in story
    assert all(value not in story for value in ['1234567','999999','555555','12345','212-555-1234','1HGCM82633A004352'])
    assert 'unverified' in story


def test_batch_and_live_share_the_same_function_and_prompt():
    f={'manufacturer':'Ford','model':'Focus'};text='Service history reported.'
    e=FakeEngine();single=summarize_vehicle(f,text,e)
    e2=FakeEngine();batch=summarize_vehicles([{'fields':f,'description':text}],e2)[0]
    assert single==batch and e.requests==e2.requests
    assert single['status']=='ready'


def test_long_description_tail_is_processed_and_reduced():
    engine=FakeEngine()
    result=summarize_vehicle({'manufacturer':'Ford'},'A'*200+'TAIL_DEFECT',engine,SummaryConfig(source_chunk_tokens=100))
    prompts=' '.join(m['content'] for request in engine.requests for m in request)
    assert 'TAIL_DEFECT' in prompts and result['source_chunks']>1
    assert len(engine.requests)>result['source_chunks']  # reduction happened


def test_invented_numeric_claim_is_withheld_not_silently_displayed():
    result=summarize_vehicle({'year':2014,'odometer':95000},'Service history',FakeEngine('This 2020 car has 40000 miles.'))
    assert result['status']=='review_required' and result['summary']==''
    assert result['candidate_summary'] and set(result['unsupported_numbers'])=={'2020','40000'}


def test_invented_transmission_is_withheld_even_if_all_numbers_exist():
    result=summarize_vehicle({'year':2017,'cylinders':'6 cylinders','transmission':'other'},
                             'Listed condition is good',FakeEngine('2017 car with 6-speed manual transmission.'))
    assert result['status']=='review_required' and result['summary']==''
    assert result['conflicting_specs'] and not result['unsupported_numbers']


def test_sales_boilerplate_removed_from_generated_paragraph():
    from pricing.vehicle_summary import concise_summary
    text='This Ford has service history. Financing is available. Seller reports worn tires. It has a clean title. Call the stock number.'
    result=concise_summary(text)
    assert result=='This Ford has service history. Seller reports worn tires. It has a clean title.'


def test_changed_fields_and_description_change_identity():
    a=vehicle_story({'manufacturer':'Ford','year':2014},'Good condition')
    b=vehicle_story({'manufacturer':'Ford','year':2015},'Good condition')
    assert input_fingerprint(a)!=input_fingerprint(b)
