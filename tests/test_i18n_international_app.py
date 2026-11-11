"""Real-app smoke checks for new European language choices.

The tests run the production Streamlit routes without altering price weights.
"""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def open_app(code, market):
    at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=75)
    at.query_params["lang"]=code
    at.query_params["market"]=market
    at.run()
    assert not at.exception
    return at


def text_of(at):
    return "\n".join(str(x.value) for x in (*at.markdown,*at.caption,*at.info))


@pytest.mark.parametrize("language,marker,market",[
    ("uk","Незалежна оцінка авто","us"),
    ("tr","İki kıta. Tek garaj.","eu"),
    ("nb","To kontinenter. Én garasje.","eu"),
    ("sr-Cyrl","Независна процена цене","us"),
    ("sr-Latn","Nezavisna procena cene","eu"),
    ("ca","Dos continents. Un garatge.","eu"),
])
def test_international_language_market_gateway(language,marker,market):
    at=open_app(language,market)
    visible=text_of(at)
    assert marker in visible
    assert "INTERNATIONAL EDITION" not in visible
    assert "American seller Ask" not in visible


@pytest.mark.parametrize("language,expected",[
    ("uk","Країна · 8"),
    ("tr","Ülke · 8"),
    ("nb","Land · 8"),
    ("sr-Cyrl","Држава · 8"),
    ("sr-Latn","Država · 8"),
    ("ca","País · 8"),
])
def test_international_eu_country_ui_keeps_canonical_market(language,expected):
    at=open_app(language,"eu")
    labels=[x.label for x in at.selectbox]
    assert expected in labels
    assert at.session_state.get("pricing_market_v1","eu") == "eu"
