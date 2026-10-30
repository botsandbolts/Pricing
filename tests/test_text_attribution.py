"""Explanation maths, bounded inference, safe rendering and session UI contracts."""
import json
import numpy as np
import pytest

from pricing.text_attribution import (phrase_groups, explain_text, signature,
                                      highlight_html, contribution_chart)


def test_groups_preserve_complete_text_and_word_order():
    text = 'No accident history.\nFull service records; inspection welcome.'
    groups = phrase_groups(text, 4)
    assert len(groups) == 3
    assert ''.join(groups) == text
    assert phrase_groups('', 8) == []
    assert phrase_groups('No accident history. Full service records.', 8) == ['No accident history. ', 'Full service records.']


def test_exact_shapley_splits_interactions_and_uses_real_blank_baseline():
    calls = []
    def score(text):
        calls.append(text)
        return 1000 + 100*('clean' in text.split()) - 30*('dent' in text.split()) + 80*('clean' in text.split() and 'dent' in text.split())
    p = explain_text(score, 'clean dent', limit=2)
    assert p['exact']
    assert p['base'] == 1000
    assert p['prediction'] == 1150
    np.testing.assert_allclose(p['values'], [140, 10])
    assert p['error'] == 0
    assert len(calls) == len(set(calls)) == 4
    assert '' in calls
    np.testing.assert_allclose(p['se'], 0)


@pytest.mark.parametrize('limit,pairs,budget', [(8, 2, 30), (12, 4, 90)])
def test_sampled_nonlinear_paths_reconstruct_in_usd_and_respect_cpu_budget(limit, pairs, budget):
    text = ' '.join(f'word{i}' for i in range(30))
    calls, progress = [], []
    def score(s):
        calls.append(s)
        return float(np.exp(5 + len(s.split())/30) + (700 if 'word1' in s.split() and 'word20' in s.split() else 0))
    p = explain_text(score, text, limit=limit, pairs=pairs,
                     progress=lambda done, total: progress.append((done, total)))
    assert not p['exact']
    assert p['evaluations'] == len(calls) <= budget
    assert len(set(calls)) == len(calls)
    assert p['base'] + sum(p['values']) == pytest.approx(p['prediction'])
    assert progress[-1][0] == progress[-1][1] == len(calls)
    assert all(v >= 0 for v in p['se'])
    again = explain_text(score, text, limit=limit, pairs=pairs)
    assert again['values'] == p['values']


def test_sampled_additive_word_effects_are_exact_without_surrogate():
    text = 'a b c d e f g h'
    weights = dict(zip(text.split(), [1, -2, 3, -4, 5, -6, 7, -8]))
    p = explain_text(lambda s: 100 + sum(weights[w] for w in s.split()), text)
    np.testing.assert_allclose(p['values'], list(weights.values()))
    np.testing.assert_allclose(p['se'], 0)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1])
def test_bad_predictions_never_become_plausible_explanations(bad):
    with pytest.raises(ValueError, match='invalid dollar'):
        explain_text(lambda _: bad, 'clean car')


def test_empty_description_and_unbounded_settings_are_rejected():
    with pytest.raises(ValueError, match='description'):
        explain_text(lambda _: 10, '')
    with pytest.raises(ValueError):
        explain_text(lambda _: 10, 'car', limit=999)
    with pytest.raises(ValueError):
        explain_text(lambda _: 10, 'car', pairs=999)


def test_signature_invalidates_every_relevant_change():
    args = [{'model': 'v1'}, {'odometer': 90}, 'car', 'Quick look', 1, 1000]
    original = signature(*args)
    for index, value in enumerate([{'model': 'v2'}, {'odometer': 95}, 'dent', 'Closer look', 2, 1001]):
        changed = args.copy()
        changed[index] = value
        assert signature(*changed) != original


def test_html_is_escaped_and_collapsed_chart_keeps_total_credit():
    p = explain_text(lambda s: 500 + len(s), '<script>alert(1)</script> a b c d e f g h', limit=8)
    rendered = highlight_html(p)
    assert '<script>' not in rendered
    assert '&lt;script&gt;' in rendered
    chart = contribution_chart(p)
    assert len(chart.data[0].x) == 7
    assert sum(chart.data[0].x) == pytest.approx(p['difference'])
    assert len(contribution_chart(p, True).data[0].x) == 8
    assert 'USD' in chart.layout.xaxis.title.text
    json.dumps(p, allow_nan=False)


UI_APP = '''
import streamlit as st
from pricing.text_attribution_ui import render_text_attribution
class FakePipeline:
    manifest = {'pipeline':'test-text-model'}
    text_model = True
    def __init__(self): self.calls = 0
    def predict(self, text, fields, explain=False):
        assert fields == {'odometer':95000}
        assert explain is False
        self.calls += 1
        return {'price':1000 + 10*len(text.split())}
if 'fake' not in st.session_state: st.session_state.fake = FakePipeline()
notes = st.text_input('Description', 'Service history available. Small dent on passenger door. Inspection welcome.')
render_text_attribution({'price':1000+10*len(notes.split())}, {'odometer':95000}, notes, st.session_state.fake)
'''


def test_ui_is_on_demand_caches_only_session_and_hides_stale_result():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string(UI_APP).run()
    assert not app.exception
    assert app.session_state.fake.calls == 0
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.fake.calls <= 30
    assert [m.label for m in app.metric] == ['Same car · blank advert', 'With your description', 'Description’s net effect']
    calls = app.session_state.fake.calls
    app.run()
    assert app.session_state.fake.calls == calls
    app.text_input[0].set_value('Different car description').run()
    assert not app.metric
    assert app.session_state.fake.calls == calls
    other = AppTest.from_string(UI_APP).run()
    assert other.session_state.fake.calls == 0
    assert not other.metric


def test_tabular_mode_never_offers_irrelevant_text_analysis():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
from types import SimpleNamespace
from pricing.text_attribution_ui import render_text_attribution
render_text_attribution({'price':500}, {}, 'Nice car', SimpleNamespace(text_model=None))
''').run()
    assert not app.exception
    assert not app.button
    assert 'vehicle facts only' in app.info[0].value
