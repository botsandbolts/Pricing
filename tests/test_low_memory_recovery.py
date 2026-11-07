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


def test_public_homepage_has_full_models_without_eager_weights():
    app=code("app.py")
    assert "st.query_params.get('beta_neural')" not in app
    start=app.index("if _workspace=='🚗 Price my car':")
    end=app.index("if _workspace=='🏪 Merchant lens':",start)
    assert "render_two_market_pricing(Path(__file__).parent,main=True)" in app[start:end]
    assert "st.stop()" in app[start:end]
    ui=code("pricing/neural_experimental_ui.py")
    assert 'isolated_price(root,"ask",fields,notes)' in ui
    assert 'isolated_price(root,"bid",bid_fields)' in ui
    assert "render_price_lite(root,saved,adjusted)" in ui
    assert "render_price_explanations(" not in ui
    assert "if adjusted and not main:" in ui
    assert "captum>=" not in code("requirements.txt")
    assert "'pricing.neural_price_xai_ui'" not in app


def test_new_lite_explainer_has_no_eager_model_imports():
    code0=code("pricing/neural_price_lite_ui.py")
    node=ast.parse(code0)
    assert all(not isinstance(n,(ast.Import,ast.ImportFrom))
               for n in node.body)
    assert "from captum" not in code0
    assert "import torch" not in code0
    assert "import plotly" not in code0
    assert "import sentence_transformers" not in code0
    assert "st.button(" in code0


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
