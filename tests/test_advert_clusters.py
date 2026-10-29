import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from pricing.advert_clusters import (CATEGORIES, categorical_tokens, identity_tokens, token_documents, clean_description,
    classify, distinctive_terms, encode, load_bundle, numeric_block)


def small_bundle():
    records = pd.DataFrame([{'text':'excellent clean serviced family sedan', 'manufacturer':'toyota'},
                            {'text':'diesel towing work truck commercial', 'manufacturer':'rare-brand', 'model':'xj 5.0'}])
    words=TfidfVectorizer().fit(records.text)
    docs=token_documents(records,words)
    vectorizer=TfidfVectorizer(analyzer=identity_tokens,min_df=1).fit(docs)
    return dict(vectorizer=vectorizer, word_vectorizer=words,
                tfidf_weight=.85, numeric_weight=.15,
                numeric_knots={'year':np.array([1990,2020]), 'odometer':np.array([0,200000])},
                numeric_quantiles={'year':np.array([0.,1.]), 'odometer':np.array([0.,1.])})


def test_rare_categories_share_tfidf_numeric_ordering_and_block_weights():
    bundle = small_bundle()
    records = pd.DataFrame([{'text':'excellent clean serviced family sedan', 'manufacturer':'toyota', 'year':2005, 'odometer':100000},
                            {'text':'excellent clean serviced family sedan', 'manufacturer':'rare-brand', 'year':2005, 'odometer':100000}])
    features, text, usable = encode(bundle, records)
    assert usable.all()
    n = text.shape[1]
    np.testing.assert_allclose(features[0,:n].sum(), .85, atol=1e-6)
    np.testing.assert_allclose(features[0,-4:].sum(), .15, atol=1e-6)
    assert features.shape[1] == n+4  # No separate one-hot feature block.
    rare=bundle['vectorizer'].vocabulary_['cat:manufacturer=rare-brand']
    assert text[1,rare] > 0 and text[0,rare] == 0
    assert 'cat:model=xj%205.0' in bundle['vectorizer'].vocabulary_
    np.testing.assert_allclose(np.asarray(features.sum(axis=1)).ravel(), 1.)
    numeric = numeric_block(bundle, pd.DataFrame({'year':[1990,2005,2020,np.nan], 'odometer':[0,100000,200000,-1]})).toarray()
    np.testing.assert_allclose(numeric[:3,0], [0,.5,1])
    assert not numeric[-1].any()


def test_all_category_fields_namespaced_and_prices_excluded():
    row={key:'rare value 5' for key in CATEGORIES}
    tokens=categorical_tokens(pd.DataFrame([row]))[0]
    assert len(tokens)==len(CATEGORIES) and len(set(tokens))==len(CATEGORIES)
    assert all('rare%20value%205' in token for token in tokens)
    bundle=small_bundle()
    base={'text':'excellent clean serviced family sedan','manufacturer':'toyota','year':2015,'odometer':80000}
    first,_,_=encode(bundle,pd.DataFrame([base]))
    second,_,_=encode(bundle,pd.DataFrame([{**base,'price':999999,'predicted_text_price':1,'residual':1000}]))
    assert (first!=second).nnz==0


def test_missing_blocks_and_invalid_numbers_are_finite():
    bundle = small_bundle()
    features, _, valid = encode(bundle, pd.DataFrame([{'text':'excellent clean serviced family sedan', 'year':9999, 'odometer':-4}]))
    assert np.isfinite(features.data).all() and valid.all()
    assert features[0,-4:].nnz == 0
    assert float(features.sum()) == pytest.approx(1.)


def test_keywords_are_distinctive_and_names_unique():
    text = ['diesel towing work truck']*12 + ['family clean serviced sedan']*12
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    matrix = vectorizer.fit_transform(text)
    groups = distinctive_terms(matrix, np.array([0]*12+[1]*12), vectorizer.get_feature_names_out())
    assert groups['0']['name'] != groups['1']['name']
    assert 'diesel' in ' '.join(groups['0']['keywords'])
    assert 'family' in ' '.join(groups['1']['keywords'])


def test_short_or_unknown_text_rejected_without_embedding():
    bundle = small_bundle()
    assert classify(bundle, '', {})['cluster_id'] == -1
    assert classify(bundle, 'zzzz qqqq xxxx vvvv bbbb nnnn', {})['cluster_id'] == -1
    assert '$' not in clean_description('Price $15000 phone 212-555-1234 https://site.test')


def test_category_only_input_can_assign_but_density_noise_is_not_forced(monkeypatch):
    import hdbscan
    bundle=small_bundle()
    class Projection:
        def transform(self,features):
            return np.zeros((features.shape[0],2))
    bundle.update(reducer=Projection(),clusterer=object(),min_strength=0.,
                  clusters={'0':{'name':'Rare model group','keywords':['model: xj 5.0'],'size':20}})
    monkeypatch.setattr(hdbscan,'approximate_predict',lambda *a:(np.array([0]),np.array([.05])))
    result=classify(bundle,'',{'manufacturer':'rare-brand','model':'xj 5.0'})
    assert result['cluster_id']==0 and result['strength']==.05
    monkeypatch.setattr(hdbscan,'approximate_predict',lambda *a:(np.array([-1]),np.array([0.])))
    assert classify(bundle,'',{'manufacturer':'rare-brand','model':'xj 5.0'})['cluster_id']==-1


def test_installed_bundle_frozen_prediction_and_integrity(tmp_path):
    folder = Path('models/advert-clusters')
    if not (folder/'manifest.json').exists():
        pytest.skip('Run trainer to test installed bundle')
    bundle = load_bundle(folder)
    examples = json.loads(Path('data/description-examples.json').read_text())['examples']
    first = classify(bundle, examples[0]['text'], examples[0]['fields'])
    assert np.isfinite(first['strength']) and 0 <= first['strength'] <= 1
    assert first == classify(bundle, examples[0]['text'], examples[0]['fields'])
    assert classify(bundle, '', {})['name'] == 'Unclassified advert'
    (tmp_path/'manifest.json').write_bytes((folder/'manifest.json').read_bytes())
    (tmp_path/'model.joblib').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        load_bundle(tmp_path)
