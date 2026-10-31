"""Read checksum-verified original-cohort research artifacts without retraining."""
import gzip,io,json
from pathlib import Path
import pandas as pd
from pricing.suite_storage import read_model_file


def load(root,side,family='forest-original'):
    if family not in {'forest-original'}:raise ValueError('Unknown experiment')
    if side not in {'ask','bid'}:raise ValueError('Unknown market')
    folder=Path(root)/'models'/family/side
    if not (folder/'evaluation.json').exists():return None
    bundle=json.loads(gzip.decompress(read_model_file(folder/'model.json.gz')))
    report=json.loads((folder/'evaluation.json').read_text())
    rows=pd.read_csv(io.BytesIO(read_model_file(folder/'examples.csv.gz')),compression='gzip')
    return bundle,report,rows
