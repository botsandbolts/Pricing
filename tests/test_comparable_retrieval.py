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

def test_inspector_form_and_reruns(monkeypatch):
    from pricing import current_value
    monkeypatch.setattr(current_value,"live_index",lambda:(None,"Offline test snapshot"))
    from pricing import inspector_estimate
    from pricing.release_contract import release_identity
    monkeypatch.setattr(inspector_estimate,'quote',lambda root,fields,notes,date,allow:dict(fields=fields,notes=notes,date=str(date),model_version=release_identity(root),results={'ask':{'price':11000,'forest_price':10000,'historical_forest_price':9000,'interval_low':8000,'interval_high':14000},'bid':{'price':6000,'forest_price':5500,'historical_forest_price':5000,'interval_low':4000,'interval_high':8000}}))
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_string("from pathlib import Path\nfrom pricing.listing_inspector import render\nrender(Path('.'))").run(timeout=30)
    assert not at.exception
    at.text_area[0].set_value('Manual transmission. Full service history.')
    next(x for x in at.selectbox if x.label=='Transmission').select('automatic');at.button[0].click().run(timeout=30)
    assert not at.exception
    assert not any('flag' in x.value.lower() for x in at.info)
    assert all(x.label!='Advert checks & questions' for x in at.tabs)
    next(x for x in at.radio if x.label=='Matching approach').set_value('Bid · auction records').run(timeout=30)
    assert not at.exception
    at.number_input[-1].set_value(500.).run(timeout=30)
    assert not at.exception

def test_faiss_is_exact_cosine_and_returns_five_directly():
    from scipy.sparse import csr_matrix,hstack
    from sklearn.preprocessing import normalize
    index=ComparableIndex(sample(),8);query=sample()[3]
    cats,nums,_=index._completed([query])
    matrix=hstack([index.encoder.transform(cats).multiply(index.column_weights),csr_matrix(index.numeric_weights*index.numeric_scaler.transform(index._numeric_design(nums)))],format='csr')
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


def test_log_price_is_in_svd_and_changes_query_representation():
    rows=sample();values=np.array([10000+i*2000 for i in range(len(rows))],dtype=float)
    index=ComparableIndex(rows,16,reference_prices=values)
    assert index.use_price and index.numeric_scaler.n_features_in_==4
    q=dict(rows[4],_retrieval_price=15000)
    a=index.search(q)
    b=index.search(dict(q,_retrieval_price=60000))
    assert [m['score'] for m in a]!=[m['score'] for m in b]
    assert np.isfinite(index.vectors).all()


def test_hybrid_geometry_ignores_price_and_all_description_features():
    rows=[dict(r,_market='ask' if i%2 else 'bid',description='Some text',price=10000+i) for i,r in enumerate(sample())]
    a=ComparableIndex(rows,8)
    changed=[dict(r,price=999999,description='Completely different text',description_price=888888) for r in rows]
    b=ComparableIndex(changed,8)
    assert not a.use_price and not a.use_description_price and a.description_vectors is None
    assert np.allclose(a.vectors,b.vectors)
    q=sample()[0]
    assert len(a.search(q,market='ask'))==5 and len(a.search(q,market='bid'))==5
    assert all(x['vehicle']['_market']=='ask' for x in a.search(q,market='ask'))

def test_missing_auction_description_price_is_chained_imputed():
    from pricing.description_price_imputation import DescriptionPriceImputer
    rows=[dict(r,id=str(i),manufacturer='ford',model='focus') for i,r in enumerate(sample()*4)]
    price=np.array([10000+i*50 for i in range(len(rows))],dtype=float)
    signal=price*1.1;signal[60:]=np.nan
    model=DescriptionPriceImputer().fit(rows,price,signal)
    values,imputed=model.transform(rows,price)
    assert np.isfinite(values).all() and imputed[60:].all()
    assert not imputed[:60].any()
    assert np.isnan(signal[60:]).all()
    restored=DescriptionPriceImputer.from_data(model.data())
    assert restored.transform(rows[-3:],price[-3:])[0]==pytest.approx(values[-3:])
