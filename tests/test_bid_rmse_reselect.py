"""USD RMSE selection safety / tail-loss and leakage isolation tests."""
import numpy as np
import pytest
from scripts.bid_rmse_reselect import score,paired_rmse_interval,specs,TRIAL_COUNT


def test_rmse_is_actual_square_usd_error_and_punishes_catastrophic_miss():
    y=np.array([10000.,10000.,10000.,10000.])
    p=np.array([10000.,10000.,10000.,0.1])
    s=score(y,p)
    assert abs(s["mae_usd"]-(9999.9/4))<1e-6
    assert abs(s["rmse_usd"]-(9999.9/2))<1e-6
    assert s["rmse_usd"]>s["mae_usd"]
    assert s["absolute_error_p99_usd"]>9500.
    assert s["fraction_error_over_5000_usd"]==.25
    assert s["fraction_error_over_10000_usd"]==0


def test_rmse_can_change_ranking_from_mae_and_is_not_log_rmse():
    y=np.array([1000.,1000.,1000.,1000.])
    a=np.array([1000.,1000.,1000.,700.])
    b=np.array([910.,910.,910.,910.])
    assert score(y,a)["mae_usd"]<score(y,b)["mae_usd"]
    assert score(y,a)["rmse_usd"]>score(y,b)["rmse_usd"]
    assert score(y,b)["rmse_usd"]==90.


def test_rmse_paired_bootstrap_has_correct_sign():
    actual=np.full(100,10000.)
    better=np.full(100,9990.)
    worse=np.full(100,9000.)
    interval=paired_rmse_interval(actual,better,worse,replications=45)
    assert len(interval)==2
    assert interval[0]<-900 and interval[1]<-900


def test_catastrophic_metrics_and_bad_predictions():
    s=score([1000,25000,200000],[10000,1000,100000])
    assert s["absolute_error_p99_9_usd"]>=s["absolute_error_p99_usd"]
    assert s["absolute_error_p99_usd"]>=s["absolute_error_p95_usd"]
    assert s["fraction_error_over_20000_usd"]==2./3.
    with pytest.raises(ValueError):
        score([1000.],[float("nan")])
    with pytest.raises(ValueError):
        score([1000.],[-10.])
    with pytest.raises(ValueError):
        score([1000.,2000.],[900.])


def test_registered_studies_full_vs_lemon_and_all_trials():
    assert set(TRIAL_COUNT)=={"ft29","fm10","fm26"}
    for s,n in TRIAL_COUNT.items():
        assert len(specs(s))==n
    assert {v[2] for v in specs("fm26").values()}=={"full","lemon"}
    assert {v[2] for v in specs("fm10").values()}=={"full","lemon"}
    with pytest.raises(ValueError):
        specs("unknown")


def test_science_panel_cites_peer_reviewed_research_without_training():
    from pricing.error_metrics_science import ACADEMIC_PAPERS
    assert len(ACADEMIC_PAPERS)>=3
    for p in ACADEMIC_PAPERS:
        assert p["doi"].startswith("https://doi.org/10.")
        assert p["authors"] and p["journal"] and p["message"]
    from pathlib import Path
    method=Path("pricing/error_metrics_science.py").read_text()
    assert "st.latex" in method and "trained" in method
    assert "render" in method


def test_pending_rmse_reports_are_not_misrepresented_as_completed(tmp_path):
    import json
    from pricing.error_metrics_science import complete_reports
    assert complete_reports(tmp_path)==[]
    folder=tmp_path/"reports"/"bid-rmse"/"fm10"
    folder.mkdir(parents=True)
    (folder/"comparison.json").write_text(json.dumps({
        "status":"running","study":"fm10",
        "primary_selection_metric":"original_holdout_validation_RMSE_USD",
        "trained_objectives_unchanged":True,
        "original_test_used_only_after_val_selection":True}))
    assert complete_reports(tmp_path)==[]
