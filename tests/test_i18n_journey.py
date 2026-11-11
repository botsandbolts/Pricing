"""Integration regression: real app routes render translated entrypoints.

The test intentionally does not load pricing weights or produce a synthetic
quote. It checks the user-facing layout that previously escaped browser smoke.
"""
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def run_market(market):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=75)
    at.query_params["market"] = market
    at.query_params["lang"] = "nl"
    at.run()
    assert not at.exception
    return at


def all_rendered_text(at):
    strings = []
    for item in at.markdown:
        strings.append(str(item.value))
    for item in at.caption:
        strings.append(str(item.value))
    return "\n".join(strings)


def test_real_dutch_gateway_cannot_regress_to_english():
    at = run_market("us")
    copy = all_rendered_text(at)
    assert "Verenigde Staten" in copy
    assert "Amerikaanse vraagprijzen" in copy
    assert "Europese vraagprijsramingen" in copy
    assert "Jouw auto. Onze data." in copy
    assert "INTERNATIONAL EDITION" not in copy
    assert "American seller Ask" not in copy
    assert "European advertised asking prices in euros" not in copy
    assert "The original tools and models stay" not in copy


def test_real_eu_route_translates_header_and_core_input_labels():
    at = run_market("eu")
    copy = all_rendered_text(at)
    assert "Verenigde Staten" in copy
    assert "Vraagprijsramingen" in copy or "vraagprijsramingen" in copy
    assert "From Hamburg to Rome" not in copy
    assert "EUROPEAN ADVERTISED ASK" not in copy
    labels = [x.label for x in at.selectbox] + [x.label for x in at.text_input] + [x.label for x in at.number_input]
    assert "Land · 8" in labels
    assert "Merk" in labels
    assert "Jaar eerste toelating" in labels
    assert "Kilometerstand (km)" in labels


def test_dutch_european_science_panel_is_accessible_without_nested_expanders():
    at = run_market("eu")
    nav = next(x for x in at.selectbox if x.key == "eu_workspace_selector_v2")
    nav.set_value("🔬 The science").run()
    assert not at.exception
    research = next(x for x in at.checkbox if x.key == "cps_eu_research_english_🔬 The science")
    assert "Engels" in research.label or "Engelse" in research.label
    research.set_value(True).run()
    assert not at.exception
    assert next(x for x in at.selectbox if x.key == "eu_workspace_selector_v2").value == "🔬 The science"
