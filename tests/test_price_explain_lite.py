"""Memory-light price explainability: truthful arithmetic, native cache reuse and one-row XGBoost."""
import ast
import sys
from pathlib import Path

import numpy as np
import pytest

from pricing import price_explain_lite as lite


def test_source_is_lightweight_and_never_uses_process_or_captum():
    source=Path("pricing/price_explain_lite.py").read_text()
    module=ast.parse(source)
    for node in module.body:
        if isinstance(node,ast.Import):
            assert all(item.name in {"json","math","hashlib"} for item in node.names)
        if isinstance(node,ast.ImportFrom):
            raise AssertionError("No eager third-party imports")
    assert "subprocess" not in source
    assert "torch.load(" not in source
    assert "from captum" not in source
    assert "import shap" not in source
    assert "pred_contribs=True" in source
    assert "one_scenario(" in source


def test_explain_clock_exact_accounting_without_model_reexecution():
    receipt={"anchor_price_reference_usd":10000.,"age_factor":.85,
             "market_factor":1.2,"price":10200.}
    x=lite.explain_clock(receipt,10000.)
    assert x["anchor"]==10000.
    assert x["age_effect"]==pytest.approx(-1500.)
    assert x["market_effect"]==pytest.approx(1700.)
    assert sum(x[k] for k in ("anchor","age_effect","market_effect"))==pytest.approx(10200.)
    with pytest.raises(ValueError,match="anchor"):
        lite.explain_clock(receipt,11500.)
    with pytest.raises(ValueError,match="reconcile"):
        lite.explain_clock({**receipt,"price":20000.},10000.)


def test_bid_component_math_exactly_matches_native_model_receipt():
    r=lite._bid_ensemble({
        "deepfm_price":12000.,"xgboost_lemon_price":8000.,
        "price":11000.})
    assert r["weight"]==.75 and r["combined"]==11000.
    with pytest.raises(ValueError,match="reconcile"):
        lite._bid_ensemble({
            "deepfm_price":12000.,"xgboost_lemon_price":8000.,
            "price":11500.})


def test_bounded_noncausal_comparable_car_choices():
    example={"odometer":94000.,"title_status":"clean","auction_grade":"3"}
    assert lite._change(example,"10,000 fewer miles","ask")["odometer"]==84000.
    assert lite._change(example,"10,000 more miles","bid")["odometer"]==104000.
    assert lite._change(example,"Compare a rebuilt title","ask")["title_status"]=="rebuilt"
    assert lite._change(example,"Compare pre-sale grade 4","bid")["auction_grade"]=="4"
    with pytest.raises(ValueError,match="Unsupported"):
        lite._change(example,"Compare pre-sale grade 4","ask")


class FakeAsk:
    def __init__(self): self.predictions=0;self.encodes=0
    def prepare_text(self,text):
        self.encodes+=1
        return {"embedding":"prepared"}
    def predict_prepared(self,car,prepared):
        self.predictions+=1
        assert prepared["embedding"]=="prepared"
        return {"price":float(11000.+float(car["odometer"])*.05)}


class FakeBid:
    def __init__(self):self.predictions=0
    def predict(self,car):
        self.predictions+=1
        return {"price":float(8000.+float(car["odometer"])*.04)}


def test_one_real_ask_prediction_uses_existing_prepared_bge(monkeypatch):
    from pricing import neural_ask_embedding_live as module
    instance=FakeAsk()
    monkeypatch.setattr(module,"load",lambda root:instance)
    saved={"ask":{"price":15500.,"model":"native Ask"},
           "ask_prepared":{"embedding":"prepared"},
           "notes":"Excellent vehicle",
           "fields":{"odometer":90000.,"title_status":"clean"}}
    result=lite.one_scenario(".",
          saved,"ask","10,000 more miles")
    assert result["comparison_usd"]==16000.
    assert result["difference_usd"]==500.
    assert instance.encodes==0
    assert instance.predictions==1


def test_one_real_bid_prediction_uses_existing_cached_model(monkeypatch):
    from pricing import neural_bid_blend_live as module
    instance=FakeBid()
    monkeypatch.setattr(module,"load",lambda root:instance)
    saved={"bid":{"price":11600.,"model":"native Bid"},
           "bid_fields":{"odometer":90000.,"auction_grade":"3"}}
    result=lite.one_scenario(".",
          saved,"bid","10,000 fewer miles")
    assert result["comparison_usd"]==11200.
    assert result["difference_usd"]==-400.
    assert instance.predictions==1


def test_native_single_row_tree_contributions_reconcile_without_shap_library(monkeypatch):
    from pricing import hierarchy_sparse_live,neural_bid_blend_live
    names=["odometer","auction_grade"]
    class FakeMatrix:
        feature_names=names
    class Booster:
        def predict(self,row,pred_contribs=False,output_margin=False):
            if pred_contribs:
                return np.array([[.18,-.08,9.]],dtype=float)
            if output_margin:return np.array([9.10],dtype=float)
            raise AssertionError("Unexpected prediction path")
    class Original:
        xgb=Booster()
        xgb_schema={}
        def features(self,fields):
            return None,None,np.array([0.,5.,10.,.0,3.]),[]
    class Model:
        original=Original()
    monkeypatch.setattr(neural_bid_blend_live,"load",lambda root:Model())
    monkeypatch.setattr(hierarchy_sparse_live,"design_matrix",
                        lambda *a,**k:(FakeMatrix(),[]))
    price=float(np.exp(9.10))
    saved={"bid":{"price":price,"xgboost_lemon_price":price},
           "bid_fields":{"year":2010,"odometer":90000.,
                         "auction_grade":"3"}}
    out=lite.native_bid_tree_drivers(".",saved)
    assert out["tree_price_usd"]==pytest.approx(price,rel=1e-6)
    assert abs(out["log_reconciliation"])<1e-6
    assert [r["Feature"] for r in out["top_drivers"]]==["odometer","auction_grade"]
    saved["bid"]["xgboost_lemon_price"]=price+1000
    with pytest.raises(ValueError,match="not computed on the saved"):
        lite.native_bid_tree_drivers(".",saved)


def test_pricing_page_uses_same_cached_models_not_subprocess():
    page=Path("pricing/neural_experimental_ui.py").read_text()
    assert "render_price_lite(root,saved,adjusted," in page
    assert "results[\"ask_prepared\"]=prepared" in page
    assert "results[\"bid_fields\"]=bid_fields" in page
    assert "model.predict_prepared(fields,prepared)" in page
    assert "isolated_price" not in page
    assert "neural_isolated" not in page
    assert 'if not main and "ask" in saved and ask_ready:' in page
    assert "if adjusted and not main:" in page
    app=Path("app.py").read_text()
    assert "render_two_market_pricing(Path(__file__).parent,main=True)" in app
    assert "beta_neural" not in app
