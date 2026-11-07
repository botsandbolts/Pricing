"""Regression checks for the public low-memory recovery after the XAI incident."""
import ast
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def code(path):
    return (ROOT/path).read_text()


def test_python_modules_parse_after_rollback():
    for path in ("app.py","pricing/neural_experimental_ui.py",
                 "pricing/lightweight_price_xai_ui.py","pricing/navigation.py"):
        ast.parse(code(path),filename=path)


def test_public_price_homepage_is_not_neural_memory_heavy():
    app=code("app.py")
    # The resource-intensive BGE panel remains opt-in beta, not on public visits.
    assert "st.query_params.get('beta_neural')=='1'" in app
    assert "if _workspace=='🚗 Price my car':" in app
    assert app.index("st.query_params.get('beta_neural')")<app.index(
        "if _workspace=='🚗 Price my car':")
    assert "The new neural price laboratory is paused" not in app or "previous" in app
    assert "render_light_xai(saved,dated_receipts," in app
    assert "get_pipeline(str(path)).predict(notes,fields,explain=False)" in app
    assert "get_pipeline(str(path)).predict(notes,fields,explain=True)" not in app
    # Neither the expensive old explanation suite nor the new XAI lab is
    # run as part of the public main-page price receipt.
    legacy=app[app.index("if _workspace=='🚗 Price my car':"):]
    assert "render_visual_explanations(" not in legacy
    assert "render_text_attribution(" not in legacy
    assert "render_explanation(" not in legacy
    assert "from pricing.bart_summary import render_story" not in legacy


def test_public_xai_gated_and_no_gradients_dependency_at_boot():
    ui=code("pricing/neural_experimental_ui.py")
    assert "render_price_explanations(" not in ui
    assert "The advanced Why this price? lab is temporarily offline" in ui
    assert "captum>=" not in code("requirements.txt")
    app=code("app.py")
    assert "'pricing.neural_price_xai_ui'" not in app


def test_factor_bridge_never_invents_dollars():
    from pricing.lightweight_price_xai_ui import dollar_bridge
    source={"anchor_price_reference_usd":10000.,"age_factor":.8,
            "market_factor":1.2,"price":9600.}
    summary=dollar_bridge(source)
    assert summary["vehicle_age_effect_usd"]==pytest.approx(-2000.)
    assert summary["market_index_effect_usd"]==pytest.approx(1600.)
    assert sum(summary[k] for k in
               ("historical_anchor_usd","vehicle_age_effect_usd",
                "market_index_effect_usd"))==pytest.approx(9600.)
    with pytest.raises(ValueError,match="reconcile"):
        dollar_bridge({**source,"price":10000.})
    with pytest.raises(ValueError):
        dollar_bridge({**source,"market_factor":float("nan")})


def test_one_scenario_one_actual_call_no_surrogate():
    from pricing.lightweight_price_xai_ui import single_scenario
    requests=[]
    def native(row):
        requests.append(dict(row))
        return 10000.+.1*(row["odometer"]-10000.)
    result=single_scenario(native,10000.,{"odometer":10000.,
                     "manufacturer":"Toyota"},11000)
    assert len(requests)==1
    assert requests[0]["manufacturer"]=="Toyota"
    assert result["historical_scenario_usd"]==10100.
    assert result["difference_usd"]==100.
    with pytest.raises(ValueError):
        single_scenario(native,10000.,{"odometer":10000},2_100_000)
    assert len(requests)==1


def test_explanation_has_no_eager_model_imports():
    src=code("pricing/lightweight_price_xai_ui.py")
    module=ast.parse(src)
    module_imports=[n for n in module.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not module_imports
    assert "from captum" not in src
    assert "import torch" not in src
    assert "import plotly" not in src
    assert "import sentence_transformers" not in src
    assert "st.button(" in src
    assert "explain=False" in src
