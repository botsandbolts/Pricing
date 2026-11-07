"""Experimental-panel ask/bid contract tests and fail-closed UI checks."""
from datetime import date
from pathlib import Path
import numpy as np
import pytest
import torch

from pricing.neural_ask_embedding_live import available as ask_ready
from pricing.neural_bid_blend_live import (
    available as bid_ready,raw_numeric,checked_price,_auction_date)
from scripts.raw_embedding_ask_train import EmbeddingOnlyDeepCross


def test_uninstalled_models_never_fabricate_ask_or_auction_prices(tmp_path):
    assert ask_ready(tmp_path) is False
    assert bid_ready(tmp_path) is False


def test_bid_2014_2015_only_not_a_current_bid():
    sample={"year":2012,"odometer":95000,"auction_date":"2015-01-15",
            "auction_grade":"3"}
    vec=raw_numeric(sample)
    assert vec.shape==(5,) and np.isfinite(vec).all()
    assert vec[0]==2012 and vec[1]>2
    assert vec[4]==3
    with pytest.raises(ValueError):
        raw_numeric({**sample,"auction_date":"2026-11-06"})
    with pytest.raises(ValueError):
        raw_numeric({**sample,"odometer":-1})
    with pytest.raises(ValueError):
        raw_numeric({**sample,"year":2030})
    assert _auction_date("2014-10-05")==date(2014,10,5)


def test_bid_price_sanity():
    assert checked_price(18000,"bid")==18000.
    with pytest.raises(ValueError):
        checked_price(float("nan"),"bid")
    with pytest.raises(ValueError):
        checked_price(-200,"bid")


def test_entire_embedding_ask_model_not_ridge_head():
    model=EmbeddingOnlyDeepCross([6]*11,7,384)
    assert model.projector[1].in_features==384
    assert model.projector[1].out_features==96
    numeric=torch.ones((2,7))
    cats=torch.ones((2,11),dtype=torch.long)
    embeddings=torch.randn((2,384))
    model.eval()
    with torch.inference_mode():
        output=model(numeric,cats,embeddings)
    assert output.shape==(2,2) and torch.isfinite(output).all()
    from pricing.neural_ask_embedding_live import FOLDER,FILES
    assert "weights.pt" in FILES
    assert "encoder-int8.onnx" not in FILES
    assert "tokenizer.json" not in FILES
    assert FOLDER.endswith("ask-bge")


def test_main_pricing_uses_both_verified_neural_engines():
    ui=Path("pricing/neural_experimental_ui.py").read_text()
    app=Path("app.py").read_text()
    assert "if _workspace=='🚗 Price my car':" in app
    assert "render_two_market_pricing(Path(__file__).parent,main=True)" in app
    assert "from pricing.neural_experimental_ui import render" in app
    assert "neural_ask_embedding_live" in ui
    assert "neural_bid_blend_live" in ui
    assert "Compare complete neural estimates" in ui
    assert "No MiniLM Ridge" in ui
    assert "st.form" in ui and "st.tabs" in ui
    assert "public homepage" in ui.lower() or "main price calculator" in ui.lower()
    assert "Ridge" not in ui.split("def _ask(root):",1)[1].split("def _bid(root):",1)[0].split("st.caption(",1)[-1] or "No MiniLM Ridge" in ui


def test_description_workflow_uses_entire_deepcross_price_not_ridge_head():
    code=Path("pricing/neural_ask_embedding_live.py").read_text()
    assert "EmbeddingOnlyDeepCross" in code
    assert "prepare_text" in code and "predict_prepared" in code
    assert "text_price_heads" in code
    assert "head_price(" not in code
    assert "predict_text_price" not in code
    assert "Ridge(" not in code
    assert "price_network" not in Path("pricing/neural_experimental_ui.py").read_text()
