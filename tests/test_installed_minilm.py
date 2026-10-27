"""Deployment checks against the real Kaggle assessment weights; no retraining."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pricing.transformer_storage import backbone_directory, sha256

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'models/minilm-assessment'


@pytest.fixture(scope='module')
def pipeline():
    from pricing.inference import PricePipeline
    return PricePipeline(BUNDLE)


def test_package_checksums_and_honest_training_scope():
    m = json.loads((BUNDLE / 'manifest.json').read_text())
    for name, digest in m['files'].items():
        assert sha256(BUNDLE / name) == digest, name
    assert m['full_data_refit_complete'] is False
    assert m['description_fitting_rows'] == 148290
    assert m['trained_rows'] == 109609
    assert m['band80_log_radius'] is None


def test_native_category_schema_matches_installed_model(pipeline):
    from pricing.tabular import CATEGORICAL
    assert [pipeline.schema[c] for c in CATEGORICAL] == pipeline.lgb.pandas_categorical
    assert pipeline.lgb.feature_name() == pipeline.manifest['feature_names']
    assert pipeline.lgb.num_trees() == 999


def test_description_is_dynamic_and_independent_of_form_fields(pipeline):
    fields = dict(year=2015, odometer=95000, manufacturer='toyota', model='corolla', condition='good')
    good = 'Well maintained Toyota Corolla with service records, clean title and no known mechanical faults.'
    damaged = 'Needs engine replacement. Does not run. Salvage title and severe accident damage.'
    a = pipeline.predict(good, fields, explain=True)
    b = pipeline.predict(damaged, fields)
    c = pipeline.predict(good, {**fields, 'year':2010, 'odometer':190000})
    assert a['predicted_text_price'] > b['predicted_text_price']
    assert a['price'] != b['price']
    assert a['predicted_text_price'] == c['predicted_text_price']
    assert a['price'] != c['price']
    assert a['range'] is None
    assert a['explanation']['target'] == 'dollars'
    assert a['explanation']['additivity_error'] < 1e-6


def test_all_six_graphs_use_actual_installed_prediction(pipeline):
    from pricing.visual_explanations import analyse, load_reference
    from pricing import explanation_plots as plots
    fields = dict(year=2015, odometer=95000, manufacturer='toyota', model='corolla', condition='good')
    result = pipeline.predict('Well maintained sedan with service records.', fields)
    reference, _ = load_reference()
    a = analyse(pipeline, fields, result['predicted_text_price'], reference, pairs=4)
    assert a['conditional_text'] is True
    assert a['unit'] == 'USD'
    assert np.isclose(a['local']['prediction'], result['price'])
    assert a['local']['error'] < 1e-6
    assert a['native']['error'] < 1e-6
    figures = [plots.waterfall(a['local']), plots.beeswarm(a['native']),
               plots.dependence(a['native'], 'odometer'), plots.decision(a['decision'], a['comparison']),
               plots.mileage_curve(a['ice'], fields['odometer'], result['price']),
               plots.interaction_grid(a['heatmap'])]
    assert all(len(fig.to_json()) > 100 for fig in figures)


def test_multipart_roundtrip_and_partial_cache_repair(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    config = b'{"model_type":"bert"}'
    data = b'test weights' * 100
    compressed = gzip.compress(data)
    (source / 'config.json').write_bytes(config)
    (source / 'part00').write_bytes(compressed)
    digest = lambda b: hashlib.sha256(b).hexdigest()
    spec = dict(bytes=len(data), sha256=digest(data), config_sha256=digest(config),
                compressed_sha256=digest(compressed),
                parts=[dict(file='part00', bytes=len(compressed), sha256=digest(compressed))])
    (source / 'weights.json').write_text(json.dumps(spec))
    cache = tmp_path / 'cache'
    target = backbone_directory(source, cache)
    assert (target / 'model.safetensors').read_bytes() == data
    (target / 'config.json').unlink()
    assert (backbone_directory(source, cache) / 'config.json').read_bytes() == config
    (target / 'model.safetensors').unlink()
    (source / 'part00').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        backbone_directory(source, cache)
