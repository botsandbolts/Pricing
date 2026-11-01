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
    assert a and all(x['vehicle']['id']!='3' for x in a)
    for r in rows:r['price']=1
    b=ComparableIndex(rows).search(q,exclude_id=q['id'])
    assert [x['score'] for x in a]==pytest.approx([x['score'] for x in b])
    assert len(index.search(dict(q,manufacturer='unseen')))==5
    assert len(index.search(dict(q,year=2030)))==5
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
                assert match['vehicle']['id']!=q['id']
                assert np.isfinite(match['score'])

def test_inspector_form_and_reruns():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_string("from pathlib import Path\nfrom pricing.listing_inspector import render\nrender(Path('.'))").run(timeout=30)
    assert not at.exception
    at.text_area[0].set_value('Manual transmission. Full service history.')
    next(x for x in at.selectbox if x.label=='Transmission').select('automatic');at.button[0].click().run(timeout=30)
    assert not at.exception
    assert not any('flag' in x.value.lower() for x in at.info)
    assert all(x.label!='Advert checks & questions' for x in at.tabs)
    at.radio[0].set_value('Historical auction sales').run(timeout=30)
    assert not at.exception
    at.number_input[-1].set_value(500.).run(timeout=30)
    assert not at.exception

def test_faiss_is_exact_cosine_and_returns_five_directly():
    from scipy.sparse import csr_matrix,hstack
    from sklearn.preprocessing import normalize
    index=ComparableIndex(sample(),8);query=sample()[3]
    cats,nums,_=index._completed([query])
    matrix=hstack([index.encoder.transform(cats).multiply(index.column_weights),csr_matrix(.2*index.numeric_scaler.transform(nums))],format='csr')
    vector=normalize(index.svd.transform(matrix))[0]
    expected=sorted((float(index.vectors[i]@vector) for i,r in enumerate(index.rows)),reverse=True)[:5]
    matches=index.search(query,k=500)
    assert len(matches)==5
    assert [m['faiss_cosine_similarity'] for m in matches]==pytest.approx(expected,abs=1e-6)
    assert all(m['cosine_distance']==pytest.approx(1-m['score'],abs=1e-6) for m in matches)


def test_mixed_imputation_preserves_observed_facts_and_portable_schedule():
    from pricing.retrieval_imputation import RetrievalImputer,export,restore
    rows=[dict(id=str(i),manufacturer='ford',model='focus',year=2015,odometer=50000+i*100,
               fuel='gas' if i%3 else None,type='sedan',transmission='automatic' if i%2 else 'manual') for i in range(100)]
    fitted=RetrievalImputer().fit(rows);portable=restore(export(fitted))
    query=[dict(rows[0],odometer=None,transmission=None)]
    a=fitted.transform(query);b=portable.transform(query)
    assert a[0]==b[0] and np.allclose(a[1],b[1])
    assert np.isfinite(a[1]).all()
    assert a[0][0][0]=='ford' and a[0][0][1]=='focus'
    assert 'odometer' in a[2][0] and 'transmission' in a[2][0]
    index=ComparableIndex(rows,8,imputer=fitted)
    for match in index.search(dict(rows[1],odometer=50100)):
        original=next(r for r in rows if r['id']==match['vehicle']['id'])
        assert match['vehicle']==original


def test_cross_brand_search_has_no_compatibility_filter():
    rows=sample()
    index=ComparableIndex(rows,8)
    query=dict(rows[0],manufacturer='volkswagen',model='golf')
    matches=index.search(query)
    assert len(matches)==5
    assert all(m['vehicle']['manufacturer']!='volkswagen' for m in matches)
