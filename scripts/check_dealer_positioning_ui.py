from pathlib import Path
from streamlit.testing.v1 import AppTest
app=AppTest.from_string("from pathlib import Path\nfrom pricing.dealer_positioning_ui import render\nrender(Path('.'))").run()
assert not app.exception,app.exception
assert len(app.dataframe)==1
assert len(app.dataframe[0].value)==22
assert len(app.metric)==3
assert app.warning and app.error and app.info
print('Merchant lens rendered 22 name groups, metrics and uncertainty/service notices.')
