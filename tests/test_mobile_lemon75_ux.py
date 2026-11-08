"""Regression guard: one-tap mobile calculator and truthful Lemon75 metrics."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def src(path):
    return (ROOT/path).read_text(encoding="utf-8")


def test_current_ask_lives_in_verified_frozen_artifact():
    data=json.loads(src("models/neural-ray-lemon75-ask/manifest.json"))
    metrics=data["filtered_validation"]
    assert data["architecture"]=="deepcross"
    assert data["filtered_validation_rows"]==23751
    assert data["filtered_validation_is_conditional"] is True
    assert metrics["rows"]==23751
    assert abs(metrics["CVaR95_absolute_USD"]-8959.618612064656)<0.01
    assert abs(metrics["MAE_USD"]-1825.1853595216633)<0.01
    assert abs(metrics["RMSE_USD"]-2895.6036351809685)<0.01
    assert data["original_test_scored"] is False


def test_stale_ask_model_test_score_not_shown_as_current():
    s=src("pricing/neural_experimental_ui.py")
    assert "Lemon-cleaned 75% historical VALIDATION" in s
    assert 'metric=audit["filtered_validation"]' in s
    assert 'x.metric("CVaR95 · worst 5%",_usd(metric["CVaR95_absolute_USD"]))' in s
    assert 'y.metric("RMSE",_usd(metric["RMSE_USD"]))' in s
    assert 'z.metric("MAE",_usd(metric["MAE_USD"]))' in s
    assert "historical TEST accuracy" not in s
    assert "$19,985" not in s and "$19,092" not in s
    assert "no matching Lemon-filtered original TEST" in s.lower() or \
           "No matching Lemon-filtered original TEST" in s
    assert "retrospective" in s.lower()


def test_mobile_banner_has_one_tap_query_route():
    design=src("pricing/studio_design.py")
    lemon=src("pricing/ray_lemon75_ask_ui.py")
    navigation=src("pricing/navigation.py")
    ui=src("pricing/neural_experimental_ui.py")
    assert 'class="garage-home-cta" href="?focus=pricing"' in design
    assert 'class="ray-lemon-link" href="?focus=pricing"' in lemon
    assert "if st.query_params.get('focus') == 'pricing':" in design
    assert "if st.query_params.get('focus') == 'pricing':" in navigation
    assert "return PAGES[0]" in navigation
    assert 'fastlane=main and st.query_params.get("focus")=="pricing"' in ui
    assert "if not fastlane:" in ui


def test_touch_only_never_waits_for_hover():
    css=src("assets/garage.css")
    assert css.count("@media (hover:hover) and (pointer:fine)")>=2
    assert "touch-action:manipulation" in css
    assert "(pointer:coarse), (hover:none)" in css
    ui=src("pricing/neural_experimental_ui.py")
    assert 'key=prefix+"_calculate_single_tap"' in ui
    assert "Request received" in ui
    assert "Calculation complete" in ui


def test_changed_public_modules_syntax():
    for name in ("pricing/studio_design.py","pricing/navigation.py",
                 "pricing/neural_experimental_ui.py","pricing/ray_lemon75_ask_ui.py",
                 "pricing/cvar95_experiment_science.py"):
        ast.parse(src(name),filename=name)
