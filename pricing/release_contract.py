"""Small release identity helper, independent of cached serving modules."""
import hashlib
from pathlib import Path

def release_identity(root):
    folder=Path(root)/'models/xgboost-hierarchy'
    paths=[folder/'release.json',folder/'ask/features.json',folder/'ask/evaluation.json',folder/'bid/evaluation.json',folder/'ask/minilm-heads.json']
    digest=hashlib.sha256()
    for path in paths:
        if path.exists():digest.update(path.read_bytes())
    return digest.hexdigest()
