"""Preflight contracts for all four 50-trial neural searches."""
import json
import subprocess
import sys

import numpy as np
import pytest

from scripts.neural_200_sweep import GROUPS, N_TRIALS, spec, defaults
from scripts.cvar95_historical import risk


def test_exactly_200_deterministic_trial_configurations():
    assert GROUPS == ("ask_hparams","bid_hparams","ask_structure","bid_structure")
    assert N_TRIALS == 50
    for group in GROUPS:
        configs=[spec(group,i) for i in range(50)]
        assert len(configs)==50
        assert len({json.dumps(x,sort_keys=True) for x in configs})==50
        assert configs==[spec(group,i) for i in range(50)]


def test_param_groups_change_hyperparameters_not_architectures():
    for i in range(50):
        ask=spec("ask_hparams",i)
        assert 0.0001<ask["lr"]<.003 and ask["batch"]>=384
        assert 0<=ask["noise"]<=.03 and ask["epochs"]>=20
        bid=spec("bid_hparams",i)
        assert .0001<bid["lr"]<.008 and bid["batch"]>=512
        assert 0<=bid["tail_lambda"]<=.4
    assert defaults("ask_structure")["lr"]==.001
    assert defaults("bid_structure")["lr"]==.0025


def test_structural_trials_change_real_graph_dimensions():
    for i in range(50):
        ask=spec("ask_structure",i)
        assert ask["cross_layers"]>=1 and ask["rank"]>=16
        assert ask["experts"]>=1 and ask["project_layers"]>=1
        bid=spec("bid_structure",i)
        assert bid["rank"] in (16,32)
        assert bid["width1"]>=64 and bid["depth"]>=1


def test_never_accept_unregistered_group_or_index():
    for group,idx in (("ask",-1),("bid",0),("ask_hparams",50),("ask_structure",1.2)):
        with pytest.raises(ValueError):
            spec(group,idx)


def test_cvar_tail_metric_is_original_dollar_top_five_percent():
    y=np.full(100,1000.,dtype=float)
    p=y.copy()
    p[-5:]=[1000.,1100.,1200.,1300.,1400.]
    s=risk(y,p)
    assert s["worst_five_pct_rows"]==5
    assert np.isclose(s["CVaR95_absolute_USD"],2000.)


def test_dry_run_outputs_registered_configuration(tmp_path):
    result=subprocess.run([sys.executable,"-m","scripts.neural_200_sweep",
        "--group","bid_structure","--trial","49","--dry-run"],
        capture_output=True,text=True,check=True)
    record=json.loads(result.stdout)
    assert record["trial"]==49
    assert "rank" in record["config"]
