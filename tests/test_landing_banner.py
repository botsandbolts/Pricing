"""Trust/branding and functional anchor regressions for homepage welcome."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from pricing import studio_design

ROOT=Path(__file__).resolve().parents[1]


def test_compact_hero_has_one_cta_and_factual_localized_copy(monkeypatch):
    from pricing import i18n
    for code,marker in (
        ("en-US","Your ride. Our numbers."),
        ("en-GB","Your car. Our data."),
        ("de","Dein Auto. Unsere Daten."),
        ("fr","Votre voiture. Nos données."),
        ("ru","Два континента."),
    ):
        monkeypatch.setattr(i18n.st,"session_state",{"cps_locale":code})
        html=studio_design.landing_banner_html()
        assert '<section class="garage-home-quick"' in html
        assert 'id="garage-home-title"' in html
        assert 'href="#studio-calculator"' in html
        assert html.count('href="#studio-calculator"') == 1
        assert "github.com/botsandbolts/Pricing" in html
        assert "INTERNATIONAL EDITION" not in html
        assert "100% gdpr" not in html.lower()
        assert "<script" not in html
        assert "onclick=" not in html
        assert marker in html or "Два континента" in html or code=="ru"


def test_homepage_only_not_every_workspace(monkeypatch):
    for page,expected in [
       ("🚗 Price my car",True),
       ("🚦 Pick your garage",False),
       ("🧠 The science",False),
       ("🔎 Listing Inspector",False),
    ]:
        monkeypatch.setattr(studio_design,"st",SimpleNamespace(
            session_state={"landing_studio_workspace":page},
            query_params={}))
        assert studio_design.show_landing_banner() is expected


def test_homepage_defaults_to_banner_except_shared_garage_link(monkeypatch):
    monkeypatch.setattr(studio_design,"st",SimpleNamespace(
        session_state={},query_params={}))
    assert studio_design.show_landing_banner()
    monkeypatch.setattr(studio_design,"st",SimpleNamespace(
        session_state={},query_params={"garage":"1"}))
    assert not studio_design.show_landing_banner()
    monkeypatch.setattr(studio_design,"st",SimpleNamespace(
        session_state={},query_params={"focus":"pricing"}))
    assert studio_design.show_landing_banner()


def test_real_calculator_has_matching_anchor_and_no_duplicate_main_billboard():
    source=(ROOT/"pricing/neural_experimental_ui.py").read_text()
    assert 'id="studio-calculator" tabindex="-1"' in source
    assert "if main:" in source
    assert "st.markdown(CSS,unsafe_allow_html=True)" in source
    assert "if main:" in source[source.index("def render(root"):source.index("def render(root")+1350]


def test_css_has_mobile_reflow_keyboard_focus_and_reduced_motion():
    css=(ROOT/"assets/garage.css").read_text()
    for rule in (
       ".garage-home-hero{", ".garage-home-cta:focus-visible",
       ".garage-home-privacy summary:focus-visible",
       "@media(max-width:680px)", "grid-template-columns:1fr",
       "@media(prefers-reduced-motion:reduce)",
       ".garage-home-art svg"
    ):
        assert rule in css


def test_no_remote_image_dependency_in_hero():
    html=studio_design.landing_banner_html()
    assert "<img src=\"http" not in html
    assert "<script" not in html
    assert "onerror=" not in html
    assert "onclick=" not in html


def test_bid_banner_matches_live_release_and_is_never_fake():
    from pricing.bid_launch_announcement import banner_html
    html=banner_html()
    if not html:
        pytest.skip("Model weights not installed in this checkout")
    assert '<section' in html
    assert 'CVaR95' in html
    assert 'github.com/botsandbolts/Pricing/actions/runs/' in html
    assert '<script' not in html
    assert 'onclick=' not in html
    assert 'onerror=' not in html


def test_release_metrics_are_collapsed_before_the_calculator():
    source=(ROOT/'app.py').read_text()
    assert "if st.session_state.get('landing_studio_workspace') == '🚗 Price my car':" in source
    assert source.count('from pricing.bid_launch_announcement import banner_html as _bid_launch_banner') == 1
    assert 'with st.expander(localized_panel("🧪 Model lab"), expanded=False):' in source
    assert 'st.markdown(_banner, unsafe_allow_html=True)' in source
    assert '_RELEASE_ANNOUNCEMENT_HTML' not in source
    assert 'href="#studio-calculator"' in studio_design.landing_banner_html()
    assert source.index('render_hero()',source.index('prepare_navigation_state()')) < source.index('_workspace=render_navigation()')
