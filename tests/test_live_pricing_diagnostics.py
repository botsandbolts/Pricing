"""Guard the real Streamlit valuation path against silent stalls."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_inference_has_stage_checkpoints_and_failure_categories():
    ui=(ROOT/"pricing/neural_experimental_ui.py").read_text()
    ast.parse(ui)
    stages=(
        "form_submitted","ask_checkpoint_start","ask_checkpoint_complete",
        "ask_bge_encode_start","ask_bge_encode_complete",
        "ask_native_forward_start","ask_native_forward_complete",
        "bid_checkpoint_start","bid_checkpoint_complete",
        "bid_native_forward_start","bid_native_forward_complete","receipt_saved",
    )
    for stage in stages:
        assert f'_valuation_probe("{stage}"' in ui,stage
    assert "car_price_studio.valuation" in ui
    assert "results[\"ask\"]=model.predict_prepared(fields,prepared)" in ui
    assert "results[\"bid\"]=model.predict(bid_fields)" in ui
    assert "results[\"ask_embedding\"]=prepared[\"embedding\"]" in ui
    assert "Your inputs are preserved and Bid can still run" in ui
    assert "never record vehicle text or VINs" in ui


def test_production_python_smoke_uses_real_serving_weights_and_real_bge():
    script=(ROOT/"scripts/native_pricing_runtime_smoke.py").read_text()
    workflow=(ROOT/".github/workflows/native-pricing-runtime-qa.yml").read_text()
    ast.parse(script)
    assert "ask=load_ask(ROOT)" in script
    assert "bid=load_bid(ROOT)" in script
    assert "ask.prepare_text(DESCRIPTION)" in script
    assert "bid.predict(AUCTION)" in script
    assert "both_native_prices_verified" in script
    assert "python-version: '3.14'" in workflow
    assert "python -m scripts.native_pricing_runtime_smoke" in workflow
