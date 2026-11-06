"""Guard against leakage, invalid Lemon matching, and fabricated benchmark scores."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.neural_ask_prepare import lemon_mask, numeric_features, phase_from_source
from scripts.neural_ask_report import _scores, paired_interval, reference


def sample():
    return pd.DataFrame({
        "row_id": ["train-a", "train-b", "train-c", "val-x", "test-z"],
        "phase": ["train", "train", "train", "validation", "test"],
        "year": [2014, 2015, 2018, 2016, 2020],
        "odometer": [50000, 45000, 12000, 65000, 6000],
        "text": ["good condition", "roadworthy", "minor repairs", "clean", "excellent"]})


def evidence(tmp_path, ids=None):
    ids = ids or ["train-a", "train-c", "val-x", "test-z"]
    mapping = {"train-a": "calibration", "train-b": "calibration",
               "train-c": "calibration", "val-x": "validation", "test-z": "test"}
    path = tmp_path / "frozen.csv"
    pd.DataFrame({"row_id": ids, "split": [mapping[i] for i in ids]}).to_csv(path, index=False)
    return path


def test_lemon_removes_train_only(tmp_path):
    frame = sample()
    phase = phase_from_source(frame)
    assert phase.tolist() == frame.phase.tolist()
    mask = lemon_mask(frame, evidence(tmp_path))
    assert mask.tolist() == [True, False, True, True, True]
    assert all(mask[frame.phase != "train"])


def test_lemon_never_removes_test(tmp_path):
    with pytest.raises(ValueError, match="holdout"):
        lemon_mask(sample(), evidence(tmp_path, ["train-a", "val-x"]))


def test_lemon_rejects_phase_mismatch(tmp_path):
    source = evidence(tmp_path)
    df = pd.read_csv(source)
    df.loc[df.row_id == "val-x", "split"] = "test"
    df.to_csv(source, index=False)
    with pytest.raises(ValueError, match="canonical"):
        lemon_mask(sample(), source)


def test_numeric_label_free_and_finite():
    frame = sample()
    before = numeric_features(frame)
    frame["price"] = [10000, 15000, 30000, 23000, 50000]
    after = numeric_features(frame)
    assert before.shape == (5, 7)
    np.testing.assert_allclose(before, after)
    assert np.isfinite(before).all()


def test_rejects_split_reordering():
    frame = sample()
    frame.loc[4, "phase"] = "train"
    with pytest.raises(ValueError):
        phase_from_source(frame)


def test_reference_only_holdouts(tmp_path):
    path = tmp_path / "xgb.csv"
    pd.DataFrame({
        "row_id": ["a", "b", "c"],
        "split": ["calibration", "validation", "test"],
        "observed_price": [10000., 12000., 15000.],
        "predicted_price": [10500., 12500., 16000.]}).to_csv(path, index=False)
    result = reference(path)
    assert set(result.index) == {"b", "c"}
    assert result.loc["c", "predicted_price"] == 16000.


def test_paired_errors_not_unpaired():
    a = np.array([2000., 9000., 10000.])
    b = np.array([2100., 9000., 11000.])
    score = _scores(a, b)
    assert score["mae_usd"] == pytest.approx(1100 / 3)
    same = paired_interval(np.abs(a - b), np.abs(a - b))
    assert same == [0., 0.]


def test_duplicated_baseline_identity_rejected(tmp_path):
    path = tmp_path / "dup.csv"
    pd.DataFrame({
        "row_id": ["b", "b"], "split": ["validation", "test"],
        "observed_price": [1, 2], "predicted_price": [2, 3]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        reference(path)
