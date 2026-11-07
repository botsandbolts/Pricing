"""Primary public Ask/Bid pricing launch: routability, historical dates and safety.

These tests are deliberately offline, deterministic and do not load large
pretrained language-model checkpoints. Native holdout parity is tested in the
separate model release workflow; browser UX gets its own campaign QA workflow.
"""
import ast
from datetime import date
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


def read(name):
    return (ROOT/name).read_text()


def test_every_changed_public_python_module_parses():
    for name in ("app.py","pricing/navigation.py","pricing/studio_design.py",
                 "pricing/current_value.py","pricing/neural_experimental_ui.py",
                 "pricing/neural_champion_live.py"):
        ast.parse(read(name),filename=name)


def test_neural_beta_is_gated_and_earlier_public_calculator_restored():
    src=read("app.py")
    beta=src.index("if _workspace=='🚗 Price my car' and st.query_params.get('beta_neural')=='1':")
    next_panel=src.index("if _workspace=='🏪 Merchant lens':",beta)
    beta_section=src[beta:next_panel]
    assert "render_two_market_pricing(Path(__file__).parent,main=True)" in beta_section
    assert "render_privacy()" in beta_section
    assert "st.stop()" in beta_section
    restored=src.index("if _workspace=='🚗 Price my car':",next_panel)
    assert restored>next_panel
    assert "get_pipeline(str(path)).predict(notes,fields,explain=False)" in src
    assert "render_light_xai(saved,dated_receipts," in src
    assert "from pricing.bart_summary import" not in src
    assert "Fine-tune the details" not in src
    assert "scan_description" not in src

def test_separate_experimental_page_removed_but_legacy_links_redirect():
    src=read("pricing/navigation.py")
    module=ast.parse(src)
    page_node=next(x for x in module.body if isinstance(x,ast.Assign)
                   and any(isinstance(y,ast.Name) and y.id=="PAGES"
                           for y in x.targets))
    pages=ast.literal_eval(page_node.value)
    assert pages[0]=="🚗 Price my car"
    assert "🧬 Experimental models" not in pages
    assert len(pages)==len(set(pages))
    assert "'🧬 Experimental models'" in src
    assert "pending=PAGES[0]" in src
    assert "seller Ask + wholesale Bid" in src


def test_main_branding_science_business_and_no_synthetic_price():
    src=read("pricing/neural_experimental_ui.py")
    assert "def render(root,*,main=False)" in src
    assert "Zero crystal balls" in src
    assert "Car Price Studio main price calculator" in src
    assert "MADE IN GERMANY" in src
    assert "Ask" in src and "Bid" in src
    assert "CVaR95" in src and "RMSE" in src and "MAE" in src
    assert "Date-adjusted scenario, not an observed transaction" in src
    assert "not a guaranteed dealer profit" in src
    assert "main" in src and "compact=main" in src
    assert "champion_ask_available" in src and "champion_bid_available" in src
    assert "ask_champion or ask_available(root)" in src
    assert "bid_champion or bid_available(root)" in src
    assert "BART" not in src or "no scan/BART step" in src
    assert "Fine-tune the details" not in src
    assert "scan description" not in src.lower()


class NullContext:
    def __enter__(self):return self
    def __exit__(self,*args):return False


class FakeStreamlit:
    def __init__(self,mode,picked=None):
        self.mode=mode
        self.picked=picked
        self.inputs=[]
        self.options=[]
        self.markdowns=[]

    def container(self,**kwargs):return NullContext()
    def expander(self,*args,**kwargs):return NullContext()
    def markdown(self,s):self.markdowns.append(s)
    def caption(self,s):pass
    def radio(self,label,values,**kwargs):
        self.options.append((label,values))
        assert self.mode in values
        return self.mode
    def date_input(self,label,value,**kwargs):
        self.inputs.append((label,value,kwargs))
        return self.picked
    def checkbox(self,*args,**kwargs):return kwargs.get("value",False)


@pytest.mark.parametrize("mode,picked,expected",[
    ("Today",None,date(2026,11,7)),
    ("Historical date",date(2018,8,12),date(2018,8,12)),
    ("End of this month",None,date(2026,11,30)),
])
def test_historical_date_selector_is_prominent_and_correct(
    monkeypatch,mode,picked,expected):
    from pricing import current_value
    fake=FakeStreamlit(mode,picked)
    monkeypatch.setattr(current_value,"st",fake)
    monkeypatch.setattr(current_value,"today",lambda:date(2026,11,7))
    selected,allow=current_value.clock_controls(prefix="campaign",prominent=True)
    assert selected==expected and allow is True
    assert any("Price it today" in x for x in fake.markdowns)
    assert fake.options[0][0]=="Valuation date"
    if mode=="Historical date":
        assert len(fake.inputs)==1
        label,value,kwargs=fake.inputs[0]
        assert "Historical" in label
        assert kwargs["min_value"]==date(2014,1,1)
        assert kwargs["max_value"]==date(2026,11,7)
    else:
        assert not fake.inputs


def test_market_transport_never_uses_a_future_model_year_as_past_car():
    from marketbridge.time_engine import age_on
    assert age_on(2020,date(2018,8,12))<0
    src=read("marketbridge/time_engine.py")
    assert "if not 0<=age<=30:" in src


def test_no_auto_trained_model_or_fake_release_winner():
    src=read("pricing/neural_champion_live.py")
    assert '"release_candidate_verified"' in src
    assert '_sha(folder/"weights.pt")' in src
    assert "if not ask_available(root):" in src
    assert "if not bid_available(root):" in src
    assert "original_load(root)" in src
