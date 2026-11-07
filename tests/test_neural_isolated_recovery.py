"""New native Ask/Bid models, tiny explanation UI and memory-isolated workers.

Covers real production routing, no eager Torch import, bounded inference,
graceful OOM/timeout, no text telemetry, and accurate model sensitivity deltas.
"""
import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pricing import neural_isolated_serving as proxy
from pricing import neural_isolated_worker as worker

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT/path).read_text()


def car():
    return {
      "year":2015,"manufacturer":"Toyota","model":"Camry",
      "odometer":90000.,"state":"ca","type":"sedan",
      "condition":"good","title_status":"clean",
      "auction_date":"2015-01-15","auction_grade":"3",
    }


def test_all_new_files_compile_without_running_model_weights():
    for name in (
        "app.py","pricing/neural_experimental_ui.py",
        "pricing/neural_price_lite_ui.py",
        "pricing/neural_isolated_serving.py",
        "pricing/neural_isolated_worker.py"):
        ast.parse(read(name),filename=name)


def test_main_app_routes_to_new_models_and_not_legacy_or_beta():
    app=read("app.py")
    new=app.index("if _workspace=='🚗 Price my car':")
    legacy=app.rindex("if _workspace=='🚗 Price my car':")
    assert new<legacy
    assert "render_two_market_pricing(Path(__file__).parent,main=True)" in app[new:legacy]
    assert "st.stop()" in app[new:legacy]
    assert "beta_neural" not in app
    ui=read("pricing/neural_experimental_ui.py")
    assert 'isolated_price(root,"ask",fields,notes)' in ui
    assert 'isolated_price(root,"bid",bid_fields)' in ui
    assert "render_price_lite(root,saved,adjusted)" in ui
    assert "render_price_explanations(" not in ui
    assert "render_time_evidence(transport,adjusted)" in ui
    assert "if adjusted and not main:" in ui


def test_streamlit_parent_never_imports_heavy_explainer_or_torch():
    for name in ("pricing/neural_isolated_serving.py",
                 "pricing/neural_price_lite_ui.py"):
        src=read(name)
        assert "import torch" not in src
        assert "from captum" not in src
        assert "import shap" not in src
        assert "import sentence_transformers" not in src
    assert "captum>=" not in read("requirements.txt")
    assert "from pricing.neural_price_xai_ui import" not in read("app.py")


def fake_response(market="ask",operation="predict"):
    result={"ok":True,"market":market,"operation":operation}
    if operation=="predict":
        result["prediction"]={"market":market,"price":12345.,"model":"native_test"}
    else:
        result["sensitivity"]={
            "market":market,"baseline_usd":12345.,
            "changes":[{"scenario":"Mileage -10000 mi",
                        "historical_price_usd":12000.,
                        "difference_usd":-345.}]}
    return result


def _returncode(code,reply=None):
    stdout=("CPS_NATIVE_RESULT:"+json.dumps(reply)+"\n") if reply else ""
    return SimpleNamespace(returncode=code,stdout=stdout,stderr="")


def test_isolated_ipc_success_and_cpu_thread_limits(monkeypatch):
    args=[]
    def stub(*a,**kw):
        args.append((a,kw))
        return _returncode(0,fake_response())
    monkeypatch.setattr(proxy.subprocess,"run",stub)
    result=proxy.run(ROOT,"ask",car(),"Clean title.")
    assert result["price"]==12345.
    assert len(args)==1
    call,kw=args[0]
    assert call[0]==[sys.executable,"-m","pricing.neural_isolated_worker"]
    assert kw["env"]["OMP_NUM_THREADS"]=="2"
    assert kw["env"]["OPENBLAS_NUM_THREADS"]=="1"
    assert kw["env"]["TOKENIZERS_PARALLELISM"]=="false"
    assert kw["timeout"]<=120.


def test_oom_kills_worker_not_public_webpage(monkeypatch):
    monkeypatch.setattr(proxy.subprocess,"run",
                        lambda *a,**k:_returncode(-9))
    with pytest.raises(proxy.IsolatedModelError,match="memory allowance"):
        proxy.run(ROOT,"ask",car(),"")
    monkeypatch.setattr(proxy.subprocess,"run",
                        lambda *a,**k:_returncode(137))
    with pytest.raises(proxy.IsolatedModelError,match="memory allowance"):
        proxy.run(ROOT,"bid",car(),"")


def test_worker_timeout_and_invalid_receipts_fail_closed(monkeypatch):
    def timeout(*a,**k):
        raise subprocess.TimeoutExpired("fake",1)
    monkeypatch.setattr(proxy.subprocess,"run",timeout)
    with pytest.raises(proxy.IsolatedModelError,match="too long"):
        proxy.run(ROOT,"ask",car(),"")
    monkeypatch.setattr(proxy.subprocess,"run",
                        lambda *a,**k:_returncode(0,None))
    with pytest.raises(proxy.IsolatedModelError,match="valid receipt"):
        proxy.run(ROOT,"ask",car(),"")
    monkeypatch.setattr(proxy.subprocess,"run",
                        lambda *a,**k:_returncode(0,fake_response("bid")))
    with pytest.raises(proxy.IsolatedModelError,match="wrong market"):
        proxy.run(ROOT,"ask",car(),"")


class FakeAsking:
    def __init__(self):
        self.prepare_calls=0
        self.predict_calls=0
    def prepare_text(self,s):
        self.prepare_calls+=1
        return {"word_count":len(str(s).split())}
    def predict_prepared(self,fields,prepared):
        self.predict_calls+=1
        value=12000.+.1*float(fields["odometer"])+(
              -1500 if fields["title_status"]!="clean" else 0)
        return {"market":"ask","price":value,"model":"fake original BGE"}


class FakeBidding:
    def __init__(self):self.predict_calls=0
    def predict(self,fields):
        self.predict_calls+=1
        grade=float(fields["auction_grade"])
        value=8500.+.08*float(fields["odometer"])+500*grade
        return {"market":"bid","price":value,
                "deepfm_price":value+300,"xgboost_lemon_price":value-900}


@pytest.mark.parametrize("market,model_class",[
    ("ask",FakeAsking),("bid",FakeBidding)])
def test_worker_reuses_native_engine_and_bounds_explanation_cost(
    monkeypatch,market,model_class):
    monkeypatch.setitem(sys.modules,"torch",
                        SimpleNamespace(set_num_threads=lambda n:None))
    model=model_class()
    monkeypatch.setattr(worker,"_model",lambda root,side:model)
    fields=car()
    predict=worker.evaluate(ROOT,{
        "market":market,"fields":fields,
        "operation":"predict","description":"New tires."})
    original=predict["prediction"]["price"]
    study=worker.evaluate(ROOT,{
        "market":market,"fields":fields,"expected":original,
        "operation":"sensitivity","description":"New tires."})
    output=study["sensitivity"]
    assert output["calls"]<=4
    assert len(output["changes"])<=3
    assert output["causal"] is False
    assert all(isinstance(item["difference_usd"],float)
               for item in output["changes"])
    if market=="ask":
        assert model.prepare_calls==2  # once per worker request, not per probe
    else:
        assert all("Auction grade" in row["scenario"] or "Mileage" in row["scenario"]
                   for row in output["changes"])
    with pytest.raises(ValueError,match="Original estimate changed"):
        worker.evaluate(ROOT,{
            "market":market,"fields":fields,"expected":original+700,
            "operation":"sensitivity","description":"New tires."})


def test_bid_model_parts_are_required_to_add_up():
    from pricing.neural_price_lite_ui import _money,_signed
    assert _money(12345)=="$12,345"
    assert _signed(-120)=="−$120"
    ui=read("pricing/neural_price_lite_ui.py")
    assert 'abs(combined-published)>max(.5,published*.0001)' in ui
    assert "one-at-a-time" in ui
    assert "Captum" in ui  # marked EXCLUDED, not imported


def test_parallel_campaign_visitors_cannot_spawn_extra_native_workers(monkeypatch):
    attempts=[]
    class Busy:
        def acquire(self,timeout):
            assert timeout<=15.
            return False
        def release(self):
            raise AssertionError("Never release an unacquired slot")
    monkeypatch.setattr(proxy,"_WORKER_SLOT",Busy())
    monkeypatch.setattr(proxy.subprocess,"run",
                        lambda *a,**kw:attempts.append((a,kw)))
    with pytest.raises(proxy.IsolatedModelError,match="busy serving another visitor"):
        proxy.run(ROOT,"ask",car(),"Full description")
    assert attempts==[]
