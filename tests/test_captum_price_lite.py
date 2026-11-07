"""Low-RAM real Captum IG, exact native price parity and public page safeguards."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from pricing import captum_price_lite as lite

ROOT=Path(__file__).resolve().parents[1]


class TinyAsk(nn.Module):
    def forward(self,n,c,t):
        # Genuine dollar function under the same exp(log-price) output link.
        dollars=13000.+200*n[:,0]+300*n[:,2]+0*c[:,0].float()+0*t[:,0]
        log=torch.log(dollars)
        return torch.stack((log,log+.02),dim=-1)


class TinyBid(nn.Module):
    def forward(self,n,c):
        return torch.log(8000.+100*n[:,0]+50*n[:,2]+0*c[:,0].float())


class FakeAskOriginal:
    def __init__(self):
        self.price_network=TinyAsk().eval()
        self.encoder_calls=0
    def encode_inputs(self,fields,prepared):
        assert prepared["embedding"].shape==(1,384)
        assert prepared["raw"]=="Runs perfectly."
        n=np.zeros((1,7),dtype=np.float32)
        n[0,0]=2.0
        n[0,2]=3.0
        return n,np.ones((1,11),dtype=np.int64),prepared["embedding"]
    def prepare_text(self,notes):
        self.encoder_calls+=1
        raise AssertionError("Captum MUST NOT call BGE again")


class FakeBidOriginal:
    def features(self,fields):
        n=np.zeros((1,5),dtype=np.float32)
        n[0,0]=2.
        n[0,2]=4.
        return n,np.ones((1,10),dtype=np.int64),np.ones(5),[]


def ask_saved():
    return {"ask":{"price":14300.,"model":"real tiny test"},
            "ask_embedding":np.ones((1,384),dtype=np.float32),
            "fields":{"year":2012,"odometer":80000.},
            "notes":"Runs perfectly.","date":"2026-11-07"}


def bid_saved():
    price=8400.
    return {"bid":{"price":price,"deepfm_price":price,
                   "neural_weight":.9,"model":"tiny DeepFM"},
            "bid_fields":{"year":2012,"odometer":80000.,
                          "auction_grade":"3"},
            "date":"2026-11-07"}


def test_real_ask_ig_24_steps_single_batches_no_bge_encoder(monkeypatch):
    pytest.importorskip("captum")
    original=FakeAskOriginal()
    monkeypatch.setattr(lite,"_model",lambda *a:original)
    r=lite.explain(ROOT,ask_saved(),"ask",steps=24)
    assert r["market"]=="ask" and r["batch_size"]==1
    assert r["steps"]==24
    assert r["neural_usd"]==pytest.approx(14300.,abs=.01)
    assert r["anchor_usd"]==pytest.approx(13000.,abs=.01)
    assert r["reconciliation_error_usd"]<.1
    assert r["adequate"]
    assert original.encoder_calls==0
    assert len(r["items"])==7
    assert sum(item["dollars"] for item in r["items"])==pytest.approx(1300.,abs=.1)


def test_real_bid_ig_attributes_weighted_neural_only(monkeypatch):
    pytest.importorskip("captum")
    original=FakeBidOriginal()
    model=SimpleNamespace(model=TinyBid().eval(),original=original,alpha=.9)
    monkeypatch.setattr(lite,"_model",lambda *a:model)
    r=lite.explain(ROOT,bid_saved(),"bid",steps=24,bid_champion=True)
    assert r["neural_usd"]==pytest.approx(.9*8400.,abs=.01)
    assert r["anchor_usd"]==pytest.approx(.9*8000.,abs=.01)
    assert r["reconciliation_error_usd"]<.1
    assert r["adequate"]
    assert len(r["items"])==5
    assert "Only the weighted DeepFM" in r["scope"]


def test_mismatch_refuses_fabricated_model_attribution(monkeypatch):
    pytest.importorskip("captum")
    monkeypatch.setattr(lite,"_model",lambda *a:FakeAskOriginal())
    bad=ask_saved()
    bad["ask"]["price"]+=1000
    with pytest.raises(ValueError,match="does not reproduce"):
        lite.explain(ROOT,bad,"ask",steps=24)


def test_steps_budget_and_busy_attribution_lane(monkeypatch):
    for forbidden in (8,16,100):
        with pytest.raises(ValueError,match="exactly 24"):
            lite.explain(ROOT,ask_saved(),"ask",steps=forbidden)
    class Busy:
        def acquire(self,blocking=False):
            assert blocking is False
            return False
        def release(self):
            raise AssertionError("Never release an unacquired semaphore")
    monkeypatch.setattr(lite,"_LANE",Busy())
    with pytest.raises(RuntimeError,match="busy"):
        lite.explain(ROOT,ask_saved(),"ask",steps=24)


def test_source_has_no_eager_captum_model_load_or_extra_encoder():
    src=(ROOT/"pricing/captum_price_lite.py").read_text()
    tree=ast.parse(src)
    assert "from captum.attr import IntegratedGradients" in src
    assert "internal_batch_size=1" in src
    assert "INTEGRATION_STEPS=24" in src
    assert "def explain(root,saved,side,steps=INTEGRATION_STEPS" in src
    assert "steps=INTEGRATION_STEPS" in src
    assert "st.select_slider(" not in src
    assert "24 Gauss–Legendre integration steps" in src
    assert "n_steps=steps" in src
    assert "method=\"gausslegendre\"" in src
    assert "return_convergence_delta=True" in src
    assert "torch.enable_grad()" in src
    assert "model.prepare_text(" not in src
    assert "torch.load(" not in src
    assert "subprocess" not in src
    for stmt in tree.body:
        if isinstance(stmt,ast.ImportFrom):
            assert stmt.module!="captum.attr", "Captum may be imported only after a click"


def test_public_homepage_has_real_weights_saved_embedding_and_lazy_button():
    src=(ROOT/"pricing/neural_experimental_ui.py").read_text()
    assert 'results["ask_embedding"]=prepared["embedding"]' in src
    assert 'results["bid_fields"]=bid_fields' in src
    assert "render_captum_lite(root,saved," in src
    assert "model.predict_prepared(fields,prepared)" in src
    assert "if st.button(\"Draw mileage sensitivity\"" in src
    assert 'if st.button("Show historical-to-selected-date evidence"' in src
    assert "load_champion_ask(root) if ask_champion" in src
    assert "load_champion_bid(root) if bid_champion" in src
    assert "subprocess.run(" not in src
    assert "from pricing.captum_price_lite import render" in src
    assert "captum==0.9.0" in (ROOT/"requirements.txt").read_text()


def test_flagship_captum_controls_are_visible_and_reachable_after_price():
    # Prevent a "deployed but invisible" regression: a successful real quote
    # must show its Captum form, never an extra collapsed accordion.
    page=(ROOT/"pricing/neural_experimental_ui.py").read_text()
    captum=(ROOT/"pricing/captum_price_lite.py").read_text()
    assert "Your AI explanation is included" in page  # before quote
    assert 'if main:' in page
    assert "### 🔍 04 / Why THAT price? · Your AI price detective" in page
    assert page.index('saved=st.session_state.get("experimental_combined_receipt")') < page.index(
        "render_captum_lite(root,saved,")
    assert "with st.container(border=True):" in captum
    assert 'with st.expander("🧠 Captum' not in captum
    assert "🔬 Explain THIS price · Captum" in captum
    assert "st.select_slider(" not in captum
    assert "24 Gauss–Legendre integration steps" in captum
    assert 'internal_batch_size=1' in captum  # unchanged memory gate
    assert 'from captum.attr import IntegratedGradients' in captum


def test_pre_quote_teaser_is_present_in_browser_release_contract():
    smoke=(ROOT/"scripts/browser_smoke_main_pricing.py").read_text()
    assert "Your AI explanation is included" in smoke
    assert "Missing public-page message" in smoke


def test_render_after_saved_price_displays_unhidden_captum_form(monkeypatch):
    """Exercise the actual post-valuation presentation without neural loading."""
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    seen=[]
    fake=SimpleNamespace(
        markdown=lambda value,**kwargs:seen.append(("markdown",value)),
        caption=lambda value,**kwargs:seen.append(("caption",value)),
        container=lambda **kwargs:nullcontext(),
        form=lambda *args,**kwargs:nullcontext(),
        radio=lambda *args,**kwargs:"ask",
        form_submit_button=lambda label,**kwargs:seen.append(("button",label)) or False,
        session_state={},
    )
    monkeypatch.setitem(sys.modules,"streamlit",fake)
    lite.render(ROOT,ask_saved())
    assert any(kind=="markdown" and "Why THAT price?" in value
               for kind,value in seen)
    assert any(kind=="button" and "Explain THIS price" in value
               for kind,value in seen)
    assert any(kind=="caption" and "24 Gauss–Legendre" in value
               for kind,value in seen)
    assert not any(kind=="caption" and "hidden" in value.lower()
                   for kind,value in seen)
    # No real model, Captum import, or attribution call before pressing the button.


def test_captum_failure_reports_parity_reason_instead_of_silent_warning(monkeypatch):
    """User sees a specific safe category; traceback goes to server logging."""
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    seen=[]
    fake=SimpleNamespace(
        markdown=lambda value,**kwargs:None,
        caption=lambda value,**kwargs:seen.append(("caption",value)),
        warning=lambda value,**kwargs:seen.append(("warning",value)),
        container=lambda **kwargs:nullcontext(),
        form=lambda *args,**kwargs:nullcontext(),
        spinner=lambda *args,**kwargs:nullcontext(),
        radio=lambda *args,**kwargs:"ask",
        form_submit_button=lambda label,**kwargs:True,
        session_state={},
    )
    monkeypatch.setitem(sys.modules,"streamlit",fake)
    def fail(*args,**kwargs):
        raise ValueError("Captum's model does not reproduce the saved price")
    monkeypatch.setattr(lite,"explain",fail)
    lite.render(ROOT,ask_saved())
    assert any(t=="warning" and "Model parity check failed" in msg
               for t,msg in seen)
    assert any(t=="caption" and "run a new quote" in msg
               for t,msg in seen)
    assert "captum_lite_result" not in fake.session_state
