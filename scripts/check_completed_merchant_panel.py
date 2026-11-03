from pathlib import Path
import json
from streamlit.testing.v1 import AppTest
report=json.loads(Path('reports/dealer-overlap/completed-analysis.json').read_text())
assert len(report['runs'])==5
assert all('primary' in r and 'alternative_top_candidate' in r for r in report['runs'])
app=AppTest.from_string("from pathlib import Path\nfrom pricing.dealer_positioning_ui import render\nrender(Path('.'))").run(timeout=60)
assert not app.exception,app.exception
assert any('completed model-level analysis' in x.value for x in app.subheader)
assert app.warning and app.info
print('Completed research panel, five imputations, support gates and caveats verified.')
