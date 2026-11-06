"""Reuse the identical 18-arm neural trainer with augmented MiniLM price features."""
import argparse
import json
from pathlib import Path
from scripts.neural_ask_train import ARCHS,LOSSES,COHORTS,run as original_train

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--arch",choices=ARCHS,required=True)
    p.add_argument("--loss",choices=LOSSES,required=True)
    p.add_argument("--cohort",choices=COHORTS,required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--epochs",type=int,default=35)
    p.add_argument("--patience",type=int,default=7)
    p.add_argument("--batch-size",type=int,default=768)
    a=p.parse_args()
    manifest=json.loads(Path(a.data).with_suffix(".json").read_text())
    if (manifest.get("variant")!="minilm-2-oob-price" or
        manifest.get("minilm_internal_stopper_labels_used") is not False or
        manifest.get("minilm_validation_test_labels_used") is not False or
        manifest.get("minilm_original_text_svd_retained") is not True):
        raise ValueError("Unverified MiniLM data or fold policies")
    original_train(a)
    path=Path(a.output)/"result.json"
    result=json.loads(path.read_text())
    result.update(feature_variant="minilm-2-oob-price",minilm_2_oob=True,
                  minilm_oob_sha256=manifest["minilm_oob_sha256"],
                  minilm_encoder_revision=manifest["minilm_encoder_revision"],
                  miniLM_original_text_svd_retained=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
