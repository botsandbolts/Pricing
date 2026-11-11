"""Regressions from the November 11 Streamlit app log.

Persisted browser/session selections may be serialized as strings. Invalid
choices must never crash a rerun before the valuation form appears.
"""
from pricing.description_examples import selected_example
from pricing.quantized_bge_runtime import VerifiedInt8BGE


def test_example_selection_accepts_integer_or_persisted_numeric_string():
    examples = [{"label": "first"}, {"label": "second"}]
    assert selected_example(examples, 1) is examples[0]
    assert selected_example(examples, "1") is examples[0]
    assert selected_example(examples, " 2 ") is examples[1]


def test_example_selection_rejects_invalid_and_stale_indices():
    examples = [{"label": "first"}]
    for choice in [None, 0, "0", -1, "2", "car title", "", True, 1.5]:
        assert selected_example(examples, choice) is None


def test_onnx_and_native_dimensionality_contract_share_current_api():
    assert VerifiedInt8BGE.get_embedding_dimension(None) == 384
    assert VerifiedInt8BGE.get_sentence_embedding_dimension(None) == 384
