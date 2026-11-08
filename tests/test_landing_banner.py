"""Trust/branding and functional anchor regressions for homepage welcome."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from pricing import studio_design

ROOT=Path(__file__).resolve().parents[1]


def test_hero_is_present_with_real_calculator_anchor_and_honest_marketing():
    html=studio_design.landing_banner_html()
    assert '<section class="garage-home-hero"' in html
    assert 'id="garage-home-title"' in html
    assert 'href="#studio-calculator"' in html
    assert 'aria-label="Go to the car pricing calculator"' in html
    assert "MADE IN GERMANY" in html
    assert "I KNOW WHAT I'VE GOT" in html
    assert "Price my car — let's roll" in html
    assert "No registration" in html
    assert "No ads or paywall" in html
    assert "Independent project" in html
    assert "MIT open source" in html
    assert "Google Analytics require consent" in html
    assert "Rejecting analytics doesn't disable the calculator" in html
    assert "Hamburg Commissioner for Data Protection" in html
    assert "Regulatory oversight is not an endorsement, audit, or certification" in html
    assert "<svg " in html
    assert "github.com/botsandbolts/Pricing" in html
    assert "https://datenschutz-hamburg.de/" in html
    assert "certified by" not in html.lower()
    assert "100% gdpr" not in html.lower()


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
