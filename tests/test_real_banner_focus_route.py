"""Source-level guards for a permanent, single-route pricing flow."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]

def source(path):
    return (ROOT/path).read_text()

def test_banners_persist_across_hash_navigation_and_calculation():
    app=source("app.py")
    design=source("pricing/studio_design.py")
    ui=source("pricing/neural_experimental_ui.py")
    nav=source("pricing/navigation.py")
    for name,s in (("app.py",app),("pricing/studio_design.py",design),
                   ("pricing/neural_experimental_ui.py",ui),("pricing/navigation.py",nav)):
        ast.parse(s,filename=name)
    assert app.count('st.markdown(_RELEASE_ANNOUNCEMENT_HTML, unsafe_allow_html=True)')==1
    assert 'if st.query_params.get("focus") != "pricing":' not in app
    assert 'href="#studio-calculator"' in design
    assert 'id="studio-calculator"' in ui
    assert 'fastlane' not in ui
    assert 'progress_notice.empty()' in ui

def test_pricing_example_selector_precedes_exactly_one_main_form():
    ui=source("pricing/neural_experimental_ui.py")
    start=ui.index("def render(root,*,main=False):")
    sample=ui.index('key="experimental_example"',start)
    form=ui.index('fields,notes,go=_vehicle(',start)
    submit=ui.index('    if go:',form)
    assert start<sample<form<submit
    assert ui[start:].count('fields,notes,go=_vehicle(')==1
    assert "on_change=_apply_example" in ui
    assert "One click received" in ui
    assert 'st.success("✅ Calculation complete.' in ui
