from streamlit.testing.v1 import AppTest
app=AppTest.from_string("from pathlib import Path\nfrom pricing.dealer_positioning_ui import render\nrender(Path('.'))").run()
assert not app.exception,app.exception
assert len(app.metric)==3
assert app.warning and app.error and app.info
for frame in app.dataframe:
    assert 'Residual association (%)' not in frame.value.columns
assert any('withheld' in w.value for w in app.warning)
print('Merchant lens loads; unsupported named residual percentages withheld.')
