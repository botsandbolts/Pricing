"""Regression: the real app's focused banner route cannot land on the bulletin."""
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[1]


def source(path):
    return (ROOT/path).read_text()


def test_focused_route_moves_reliability_warning_below_main_calculator():
    app=source("app.py")
    ast.parse(app)
    assert '_RELIABILITY_NOTICE_HTML = """' in app
    assert 'if st.query_params.get("focus") != "pricing":' in app
    calc=app.index("render_two_market_pricing(Path(__file__).parent,main=True)")
    warning_after=app.index('st.markdown(_RELIABILITY_NOTICE_HTML, unsafe_allow_html=True)',calc)
    assert warning_after>calc
    assert app.count('st.markdown(_RELIABILITY_NOTICE_HTML, unsafe_allow_html=True)')==2


def test_banner_focus_route_shows_real_vehicle_form_before_date_or_models():
    ui=source("pricing/neural_experimental_ui.py")
    ast.parse(ui)
    anchor=ui.index('fastlane=main and st.query_params.get("focus")=="pricing"')
    initial=ui.index('    if fastlane:\n        # Direct banner navigation:',anchor)
    first_form=ui.index('fields,notes,go=_vehicle(',initial)
    clock=ui.index('    valuation_date,allow_analogue=clock_controls(',initial)
    model_description=ui.index('    a,b=st.columns(2)',clock)
    assert initial<first_form<clock<model_description
    assert ui.count("    if not fastlane:\n        fields,notes,go=_vehicle(")==1
    assert "Original auction date (2014–2015)" in ui
    assert "original" in ui
