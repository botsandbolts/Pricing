import json
from pathlib import Path
from pricing.svd_discovery import build
folder=Path('data/merchant-discovery-minilm')
report=build(folder)
(folder/'shrunk-svd.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
