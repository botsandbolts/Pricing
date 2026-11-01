import numpy as np
import pytest
from pricing.comparable_retrieval import ComparableIndex,reference_rows
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sample():
    return [dict(id=str(i),manufacturer='ford' if i<20 else 'toyota',model='focus' if i<20 else 'corolla',year=2010+i%5,odometer=i*1000,transmission='automatic',price=i*10000,description='Maintained vehicle') for i in range(30)]

def test_prices_excluded_and_identity_guardrails():
    rows=sample();index=ComparableIndex(rows)
    q=rows[3];a=index.search(q,exclude_id=q['id'])
    assert a and all(x['vehicle']['manufacturer']=='ford' and x['vehicle']['id']!='3' for x in a)
    for r in rows:r['price']=1
    b=ComparableIndex(rows).search(q,exclude_id=q['id'])
    assert [x['score'] for x in a]==pytest.approx([x['score'] for x in b])
    assert index.search(dict(q,manufacturer='unseen'))==[]
    assert index.search(dict(q,year=2030))==[]
    assert index.search(dict(q,odometer=float('nan')))==[]

def test_semantics_and_unknown_categories():
    rows=sample();index=ComparableIndex(rows,8)
    index.attach_descriptions(np.ones((30,4)))
    assert index.search(dict(rows[0],fuel='unseen'),description_vector=np.ones(4))
    with pytest.raises(ValueError):index.attach_descriptions(np.ones((2,4)))
    with pytest.raises(ValueError):index.search(rows[0],description_vector=np.ones(5))

def test_real_reference_bounds():
    for side in ('ask','bid'):
        index=ComparableIndex(reference_rows(ROOT,side),32)
        for q in index.rows[::200]:
            for match in index.search(q,exclude_id=q['id']):
                assert abs(q['year']-match['vehicle']['year'])<=5
                assert q['manufacturer'].lower()==match['vehicle']['manufacturer'].lower()

def test_inspector_form_and_reruns():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_string("from pathlib import Path\nfrom pricing.listing_inspector import render\nrender(Path('.'))").run(timeout=30)
    assert not at.exception
    at.text_area[0].set_value('Manual transmission. Full service history.')
    next(x for x in at.selectbox if x.label=='Transmission').select('automatic');at.button[0].click().run(timeout=30)
    assert not at.exception
    assert any('Check transmission' in x.value for x in at.subheader)
    at.radio[0].set_value('Historical auction sales').run(timeout=30)
    assert not at.exception
    at.number_input[-1].set_value(500.).run(timeout=30)
    assert not at.exception
