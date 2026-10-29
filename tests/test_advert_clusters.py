import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from pricing.advert_clusters import (CATEGORIES, category_frame, clean_description,
    classify, distinctive_terms, encode, load_bundle, numeric_block)


def small_bundle():
    records = pd.DataFrame([{'text':'excellent clean serviced family sedan', 'manufacturer':'toyota'},
                            {'text':'diesel towing work truck commercial', 'manufacturer':'ford'}])
    encoder = OneHotEncoder(categories=[['toyota','ford']] + [['__none__']]*(len(CATEGORIES)-1), handle_unknown='ignore')
    encoder.fit(category_frame(records))
    return dict(vectorizer=TfidfVectorizer().fit(records.text), encoder=encoder,
                text_weight=.65, category_weight=.20, numeric_weight=.15,
                numeric_knots={'year':np.array([1990,2020]), 'odometer':np.array([0,200000])},
                numeric_quantiles={'year':np.array([0.,1.]), 'odometer':np.array([0.,1.])})


def test_rare_categories_are_ignored_numeric_ordering_and_block_weights():
    bundle = small_bundle()
    records = pd.DataFrame([{'text':'excellent clean serviced family sedan', 'manufacturer':'toyota', 'year':2005, 'odometer':100000},
                            {'text':'excellent clean serviced family sedan', 'manufacturer':'rare-brand', 'year':2005, 'odometer':100000}])
    features, text, usable = encode(bundle, records)
    assert usable.all()
    n = text.shape[1]
    np.testing.assert_allclose(features[0,:n].sum(), .65, atol=1e-6)
    np.testing.assert_allclose(features[0,-4:].sum(), .15, atol=1e-6)
    assert features[1,n:-4].nnz == 0
    np.testing.assert_allclose(np.asarray(features.sum(axis=1)).ravel(), 1.)
    numeric = numeric_block(bundle, pd.DataFrame({'year':[1990,2005,2020,np.nan], 'odometer':[0,100000,200000,-1]})).toarray()
    np.testing.assert_allclose(numeric[:3,0], [0,.5,1])
    assert not numeric[-1].any()


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
