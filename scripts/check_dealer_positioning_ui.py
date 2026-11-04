from pathlib import Path
from streamlit.testing.v1 import AppTest
app=AppTest.from_string("from pathlib import Path\nfrom pricing.dealer_positioning_ui import render\nrender(Path('.'))").run()
assert not app.exception,app.exception
if (Path('reports/dealer-overlap/completed-analysis.json').exists()):
    import json
    report=json.loads(Path('reports/dealer-overlap/completed-analysis.json').read_text())
    if report.get('version',0)>=3:assert len(app.metric)==1
assert app.warning and app.info
for frame in app.dataframe:
    assert 'Residual association (%)' not in frame.value.columns
assert any('No unsupported named percentages' in x.value or 'end-to-end' in x.value for x in app.info) or any('completed model-level analysis' in x.value for x in app.subheader)
print('Merchant lens loads; unsupported named residual percentages withheld.')
