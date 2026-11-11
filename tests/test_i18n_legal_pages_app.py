"""Actual legal routes should render the selected language before source English."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("kind,expected",[
    ("privacy","Privacy en cookies"),
    ("imprint","Colofon en juridische informatie"),
])
def test_dutch_legal_route_keeps_reference_and_local_copy(kind,expected):
    at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=30)
    at.query_params["legal"]=kind
    at.query_params["lang"]="nl"
    at.run()
    assert not at.exception
    visible="\n".join(str(x.value) for x in at.markdown)
    assert expected in visible
    assert "amitsteinberg74@gmail.com" in visible
    assert "volledige Engelse" in visible
    assert "German and EU laws apply" in visible if kind=="imprint" else "GDPR Article 6(1)(a)" in visible


def test_polish_privacy_notice_is_explained_before_english_original():
    at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=30)
    at.query_params["legal"]="privacy"
    at.query_params["lang"]="pl"
    at.run()
    assert not at.exception
    visible="\n".join(str(x.value) for x in at.markdown)
    assert "Prywatność i pliki cookie" in visible
    assert "90 dni" in visible
    assert "Turso" in visible
    assert "GDPR Article 6(1)(f)" in visible
