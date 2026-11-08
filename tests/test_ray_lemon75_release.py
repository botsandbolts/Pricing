"""Fast fail-closed production gate checks; full real-car parity runs in CI."""
from pathlib import Path

import pytest

from pricing.ray_lemon75_ask_live import (
    FOLDER, REQUIRED, STUDY_RUN, STUDY_ARTIFACT, manifest, available)
from scripts.release_ray_lemon75_ask import (
    TRAIN_ROWS, LEMON_TRAIN, VALIDATION_ROWS, RETAINED_VALIDATION,
    sha_ids, HOLDOUT_SHA)


def test_source_identity_and_75_percent_partition():
    assert STUDY_RUN == 42485146978
    assert STUDY_ARTIFACT == 13621540013
    assert TRAIN_ROWS == 147777
    assert LEMON_TRAIN == 110832
    assert TRAIN_ROWS - LEMON_TRAIN == 36945
    assert RETAINED_VALIDATION == 23751
    assert VALIDATION_ROWS - RETAINED_VALIDATION == 7917
    assert len(HOLDOUT_SHA)==64
    assert len(REQUIRED)==4
    assert FOLDER == "models/neural-ray-lemon75-ask"


def test_release_unavailable_without_artifacts(tmp_path):
    assert manifest(tmp_path) is None
    assert available(tmp_path) is False


def test_source_id_fingerprint_order_independent():
    assert sha_ids(["b", "a", "c"]) == sha_ids(["c", "b", "a"])


def test_fake_manifest_does_not_enable_new_model(tmp_path):
    root=tmp_path/FOLDER
    root.mkdir(parents=True)
    for name in REQUIRED:
        (root/name).write_bytes(b'unsafe')
    assert manifest(tmp_path) is None
    assert available(tmp_path) is False
