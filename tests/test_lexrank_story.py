import numpy as np
from pricing.lexrank_story import summarize,lexrank,narrative


def test_defect_and_negation_survive_sales_noise():
    fields=dict(manufacturer='Ford',model='Focus',year=2014,odometer=95000)
    text='Financing available. New tires installed. Air conditioning does not work. Call now. Great car.'
    r=summarize(fields,text)
    assert 'Ford Focus' in r['facts'] and '95,000' in r['facts']
    assert 'Air conditioning does not work.' in r['sentences']
    assert 'New tires installed.' in r['sentences']
    assert not any('Call now' in s or 'Financing' in s for s in r['sentences'])
    assert all(s in text for s in r['sentences'])


def test_no_targets_or_downloads_and_deterministic():
    f=dict(manufacturer='Honda',price=1234567,predicted_text_price=7654321,condition='unknown')
    a=summarize(f,'Runs well. Runs well. Check-engine light on.')
    assert a==summarize(f,'Runs well. Runs well. Check-engine light on.')
    assert '1234567' not in a['story'] and '7654321' not in a['story']
    assert a['sentences'].count('Runs well.')==1
    assert summarize({},'')['sentences']==[]


def test_rank_normalized_and_story_tail_available():
    rank=lexrank(['Engine serviced.','Engine needs repair.','New tires.'])
    assert np.isclose(rank.sum(),1) and np.isfinite(rank).all()
    r=summarize({},'Nice. '*100+'Flood damage disclosed.')
    assert 'Flood damage disclosed.' in r['sentences']
    assert 'Flood damage disclosed.' in r['story']


def test_contacts_redacted_and_html_removed():
    _,text=narrative({},'<b>New brakes.</b> https://example.com a@b.com 212-555-1234')
    assert '<b>' not in text and 'a@b.com' not in text and '212-555-1234' not in text
