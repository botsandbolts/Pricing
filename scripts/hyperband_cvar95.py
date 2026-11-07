"""Artifact-synchronized Optuna TPE + multi-bracket Hyperband for Ask and Bid.

Six batches, four independent Ask and four independent Bid candidates each.
Only original TRAIN is used for epoch pruning. The original validation split
is reserved for one final CVaR95 score per surviving configuration.
No test rows are evaluated, and no serving checkpoint is modified.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

MARKETS = ("ask", "bid")
SLOTS = 4
STAGES = 6
SEED = 20261107
ROW_COUNTS = {"ask": 31668, "bid": 82380}


def _spaces(market):
    """Finite categorical spaces plus two log-scaled continuous parameters."""
    common = {
        "lr": ("log", .00025, .0028 if market == "ask" else .007),
        "wd": ("log", .0001, .13),
    }
    if market == "ask":
        hp = {
            **common,
            "batch": [384, 512, 768, 1024, 1536],
            "noise": [0., .004, .008, .014, .024],
            "clip": [.35, .7, 1., 1.5, 2.5],
            "patience": [4, 6, 8, 10],
            "epochs": [28, 35, 42],
            "optimizer": ["adamw", "adam"],
            "schedule": ["cosine", "constant"],
        }
        arch = {
            "projection": [48, 64, 96, 128, 192],
            "project_layers": [1, 2, 3],
            "proj_dropout": [0., .05, .08, .15, .23],
            "cross_layers": [1, 2, 3, 4, 5],
            "rank": [16, 24, 32, 48, 64],
            "experts": [1, 2, 3],
            "width": [128, 192, 256, 384],
            "blocks": [1, 2, 3, 4],
            "dropout": [.08, .12, .2, .3],
            "activation": ["gelu", "silu"],
        }
    else:
        hp = {
            **common,
            "batch": [512, 1024, 1536, 2048, 4096],
            "beta": [.1, .17, .25, .4, .65],
            "tail_lambda": [0., .05, .1, .2, .35],
            "clip": [.5, 1., 2.],
            "patience": [3, 5, 7, 9],
            "epochs": [25, 30, 36],
        }
        arch = {
            "rank": [16, 32],
            "width1": [64, 96, 128, 192, 256],
            "width2": [32, 48, 64, 96, 128],
            "depth": [1, 2, 3, 4],
            "dropout": [.05, .12, .16, .24, .32],
            "activation": ["gelu", "silu"],
            "pairgate": [False, True],
        }
    return {**{"hp_" + k: v for k, v in hp.items()},
            **{"arch_" + k: v for k, v in arch.items()}}


def _distributions(market):
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    out = {}
    for name, choices in _spaces(market).items():
        if isinstance(choices, tuple) and choices[0] == "log":
            out[name] = FloatDistribution(choices[1], choices[2], log=True)
        else:
            out[name] = CategoricalDistribution(choices)
    return out


def _suggest(trial, market):
    parameters = {}
    for name, choices in _spaces(market).items():
        if isinstance(choices, tuple) and choices[0] == "log":
            parameters[name] = trial.suggest_float(name, choices[1], choices[2], log=True)
        else:
            parameters[name] = trial.suggest_categorical(name, choices)
    return parameters


def _config(params):
    return {
        "hyperparameters": {k[3:]: v for k, v in params.items() if k.startswith("hp_")},
        "structure": {k[5:]: v for k, v in params.items() if k.startswith("arch_")},
    }


def _restore_study(market, history):
    import optuna
    from optuna.trial import TrialState, create_trial
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=SEED + (0 if market == "ask" else 100000),
        n_startup_trials=5, constant_liar=True,
    )
    study = optuna.create_study(
        study_name=f"pricing-hyperband-20261107-{market}",
        direction="minimize", sampler=sampler,
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=4, max_resource=42, reduction_factor=4),
    )
    for row in history:
        if row["market"] != market:
            continue
        state = TrialState.COMPLETE if row["state"] == "COMPLETE" else TrialState.PRUNED
        kwargs = {
            "params": row["params"],
            "distributions": _distributions(market),
            "state": state,
            "intermediate_values": {int(k): float(v)
                                    for k, v in row["intermediates"].items()},
            "system_attrs": row.get("rungs", {}),
        }
        if state == TrialState.COMPLETE:
            value = float(row["validation"]["CVaR95_absolute_USD"])
            if not math.isfinite(value):
                raise ValueError("Invalid CVaR95 from previous batch")
            kwargs["value"] = value
        study.add_trial(create_trial(**kwargs))
    return study


def _read_history(path):
    if not path:
        return []
    data = json.loads(Path(path).read_text())
    records = data["records"]
    if len(records) != len({(r["market"], r["stage"], r["slot"]) for r in records}):
        raise ValueError("Repeated study trial identifiers")
    if any(r["market"] not in MARKETS or r["state"] not in ("COMPLETE", "PRUNED")
           for r in records):
        raise ValueError("Untrusted previous study status")
    hashes = {market: {r["provenance"]["holdout_row_sha256"]["validation"]
                       for r in records if r["market"] == market and r["state"] == "COMPLETE"}
              for market in MARKETS}
    if any(len(values) > 1 for values in hashes.values()):
        raise ValueError("Mismatched holdout cohort fingerprints")
    return records


def propose(stage, history_path, output):
    if not 0 <= stage < STAGES:
        raise ValueError("Stage outside the configured 6-batch search")
    history = _read_history(history_path)
    if len(history) != stage * SLOTS * len(MARKETS):
        raise ValueError("A new batch requires every previous candidate receipt")
    proposals = {}
    for market in MARKETS:
        study = _restore_study(market, history)
        proposals[market] = []
        existing = {json.dumps(row["params"], sort_keys=True) for row in history
                    if row["market"] == market}
        for slot in range(SLOTS):
            for attempt in range(30):
                trial = study.ask()
                params = _suggest(trial, market)
                ident = json.dumps(params, sort_keys=True)
                if ident not in existing:
                    break
                study.tell(trial, state=__import__("optuna").trial.TrialState.PRUNED)
            else:
                raise ValueError("Sampler could not produce distinct configurations")
            existing.add(ident)
            proposals[market].append({
                "market": market, "stage": stage, "slot": slot,
                "trial": stage * SLOTS + slot,
                "params": params, "config": _config(params),
            })
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "status": "ready", "stage": stage, "candidates": proposals,
        "concurrent_workers": 8, "ask_workers": 4, "bid_workers": 4,
        "sampler": "Optuna TPE, batched, prior artifacts only",
        "pruner": "Optuna Hyperband, 2 brackets (min 4, max 42 epochs, reduction 4) on internal TRAIN CVaR95",
        "objective": "original validation absolute USD CVaR95",
        "auto_deploy": False, "test_evaluated": False,
    }, indent=2, allow_nan=False))
    print("ADAPTIVE_PLAN", stage, "Ask=4 Bid=4", flush=True)



def audit_and_save_bid_blend(destination, baseline_path):
    """Persist the SAME blended dollars scored for wholesale CVaR95.

    The historical native Bid trainer used to save its standalone neural
    predictions while reporting metrics for the blended Lemon+DeepFM target.
    Keep native weights and reported objective unchanged, but correct this
    adaptive study's saved validation predictions before artifact upload.
    """
    import pandas as pd
    from scripts.cvar95_historical import risk
    from scripts.neural_200_sweep import _matched_auction_baseline
    result_path = Path(destination) / "result.json"
    csv_path = Path(destination) / "validation.csv.gz"
    result = json.loads(result_path.read_text())
    frame = pd.read_csv(csv_path, dtype={"row_id": str})
    required = {"row_id", "split", "observed_price", "predicted_price"}
    if not required.issubset(frame.columns):
        raise ValueError("Wholesale prediction receipt is incomplete")
    alpha = float(result["validation"]["neural_weight"])
    if not math.isfinite(alpha) or not .5 <= alpha <= 1.:
        raise ValueError("Invalid trained blend weight")
    ids = frame["row_id"].astype(str).to_numpy()
    actual = frame["observed_price"].to_numpy(float)
    neural = frame["predicted_price"].to_numpy(float)
    baseline = _matched_auction_baseline(baseline_path, ids, actual)
    combined = alpha * neural + (1.-alpha) * baseline
    checked = risk(actual, combined)
    for metric in ("CVaR95_absolute_USD", "RMSE_USD", "MAE_USD"):
        if abs(checked[metric] - float(result["validation"][metric])) > .05:
            raise ValueError("Wholesale blended validation does not match " + metric)
    frame["deepfm_price_before_blend"] = neural
    frame["lemon_xgboost_price_before_blend"] = baseline
    frame["predicted_price"] = combined
    frame.to_csv(csv_path, index=False)
    reread = pd.read_csv(csv_path, dtype={"row_id": str})
    after = risk(reread["observed_price"], reread["predicted_price"])
    if abs(after["CVaR95_absolute_USD"] -
           result["validation"]["CVaR95_absolute_USD"]) > .05:
        raise ValueError("Wholesale CSV serialization changed the CVaR95 score")
    print("BID_BLEND_PARITY", "CVaR95=", round(after["CVaR95_absolute_USD"], 2),
          "DeepFM weight=", alpha, "rows=", len(frame), flush=True)


def train(plan_path, market, slot, output, args):
    from optuna.trial import TrialState
    from scripts import neural_200_sweep as native
    import optuna
    plan = json.loads(Path(plan_path).read_text())
    if market not in MARKETS or not 0 <= slot < SLOTS or plan.get("status") != "ready":
        raise ValueError("Invalid registered candidate")
    chosen = plan["candidates"][market][slot]
    if (chosen["market"], chosen["slot"], chosen["stage"]) != (
        market, slot, plan["stage"]):
        raise ValueError("Wrong stage/market/slot proposal")
    history = _read_history(args.history)
    study = _restore_study(market, history)
    study.enqueue_trial(chosen["params"])
    trial = study.ask()
    confirmed = _suggest(trial, market)
    if confirmed != chosen["params"]:
        raise ValueError("Runner parameters disagree with the coordinator")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    old_risk = native.risk
    intermediate = {}
    epoch = 0

    def monitored_risk(y, prediction):
        nonlocal epoch
        stats = old_risk(y, prediction)
        # The existing trainers use only an internal TRAIN slice for stopping.
        # Full original VALIDATION is scored once only after training completes.
        if len(y) != ROW_COUNTS[market]:
            epoch += 1
            score = float(stats["CVaR95_absolute_USD"])
            if not math.isfinite(score):
                raise ValueError("Nonfinite TRAIN-only pruning score")
            intermediate[epoch] = score
            trial.report(score, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned("Successive-halving stopped this trial")
        return stats

    state = "COMPLETE"
    try:
        native.risk = monitored_risk
        settings = chosen["config"]
        if market == "ask":
            native.ask("ask_structure", chosen["trial"], args.ask_features,
                       args.ask_embeddings, destination, args.max_seconds,
                       refinement=settings)
        else:
            native.bid("bid_structure", chosen["trial"], args.bid_features,
                       args.bid_baseline, destination, args.max_seconds,
                       refinement=settings)
    except optuna.TrialPruned:
        state = "PRUNED"
    finally:
        native.risk = old_risk

    if state == "COMPLETE":
        if market == "bid":
            # Native Bid reports blend CVaR95 but historically persisted its
            # DeepFM-only vector. Never pass that mismatched CSV to the audit.
            audit_and_save_bid_blend(destination, args.bid_baseline)
        record = json.loads((destination / "result.json").read_text())
        if record["trial_configuration"] != chosen["config"]:
            raise ValueError("Native checkpoint differs from registered parameters")
        if record.get("test_evaluated") is not False:
            raise ValueError("Test contamination")
        if record["validation"]["rows"] != ROW_COUNTS[market]:
            raise ValueError("Unexpected original validation count")
        study.tell(trial, float(record["validation"]["CVaR95_absolute_USD"]))
        validation = record["validation"]
        provenance = record["provenance"]
    else:
        study.tell(trial, state=TrialState.PRUNED)
        validation = None
        provenance = {}
    frozen = study.trials[-1]
    summary = {
        **{k: chosen[k] for k in ("stage", "slot", "market", "trial", "params", "config")},
        "state": state, "validation": validation, "provenance": provenance,
        "intermediates": {str(k): v for k, v in intermediate.items()},
        "rungs": dict(frozen.system_attrs),
        "test_evaluated": False, "auto_deploy": False,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2,
                                                        allow_nan=False))
    print("ADAPTIVE_TRIAL", market, chosen["trial"], state,
          validation["CVaR95_absolute_USD"] if validation else "pruned", flush=True)


def collect(plan_path, previous_history, artifacts, output):
    import pandas as pd
    from scripts.cvar95_historical import risk
    plan = json.loads(Path(plan_path).read_text())
    stage = plan["stage"]
    history = _read_history(previous_history)
    if len(history) != stage * SLOTS * len(MARKETS):
        raise ValueError("Prior batch history is incomplete")
    received = {}
    for file in sorted(Path(artifacts).rglob("summary.json")):
        row = json.loads(file.read_text())
        key = (row["market"], row["slot"])
        if key in received or row["stage"] != stage:
            raise ValueError("Duplicated or wrong-stage result")
        expected = plan["candidates"][key[0]][key[1]]
        for field in ("trial", "params", "config", "market", "slot", "stage"):
            if row[field] != expected[field]:
                raise ValueError("Result differs from pre-registered plan")
        if row["state"] == "COMPLETE":
            result = json.loads(file.with_name("result.json").read_text())
            if result["trial_configuration"] != row["config"] or result["test_evaluated"] is not False:
                raise ValueError("Validation result or test policy changed")
            if not file.with_name("weights.pt").is_file():
                raise ValueError("Completed native checkpoint missing")
            frame = pd.read_csv(file.with_name("validation.csv.gz"))
            if (len(frame) != ROW_COUNTS[key[0]]
                or frame["row_id"].astype(str).duplicated().any()
                or not frame["split"].eq("validation").all()):
                raise ValueError("Original validation rows invalid")
            metrics = risk(frame.observed_price.to_numpy(float),
                           frame.predicted_price.to_numpy(float))
            for metric in ("CVaR95_absolute_USD", "RMSE_USD", "MAE_USD"):
                if abs(float(metrics[metric]) - float(row["validation"][metric])) > .05:
                    raise ValueError("Independent CVaR95 parity failed: " + metric)
        elif row["state"] == "PRUNED":
            if not row["intermediates"] or row["validation"] is not None:
                raise ValueError("Pruning must have original TRAIN-only intermediate metrics")
        else:
            raise ValueError("Unexpected trial status")
        received[key] = row
    expected_keys = {(m, s) for m in MARKETS for s in range(SLOTS)}
    if set(received) != expected_keys:
        raise ValueError("Missing trained/pruned candidates: " + str(sorted(expected_keys-set(received))))
    combined = history + [received[(m,s)] for m in MARKETS for s in range(SLOTS)]
    for market in MARKETS:
        hashes = {r["provenance"]["holdout_row_sha256"]["validation"]
                  for r in combined if r["market"] == market and r["state"] == "COMPLETE"}
        if len(hashes) > 1:
            raise ValueError("Validation cohort drift across rounds")
    ranking = {
        market: sorted(
            [{"trial": row["trial"], "stage": row["stage"],
              "CVaR95_absolute_USD": row["validation"]["CVaR95_absolute_USD"],
              "RMSE_USD": row["validation"]["RMSE_USD"],
              "MAE_USD": row["validation"]["MAE_USD"]}
             for row in combined if row["market"] == market and row["state"] == "COMPLETE"],
            key=lambda r:(r["CVaR95_absolute_USD"],r["RMSE_USD"],r["trial"]))
        for market in MARKETS
    }
    payload = {
        "status": "complete", "completed_stages": stage+1,
        "records": combined, "leaderboard": ranking,
        "not_for_deployment": True, "test_evaluated": False,
        "pruned_counts": {m: sum(row["state"] == "PRUNED" for row in combined
                                 if row["market"] == m) for m in MARKETS},
        "compared_to_deployed_verified": incumbent_comparison(combined, ranking),
        "optimization": "Optuna TPE + Hyperband (two independent brackets)",
    }
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print("ADAPTIVE_COLLECT stage", stage, "records", len(combined),
          "complete", sum(row["state"] == "COMPLETE" for row in combined),
          "pruned", sum(row["state"] == "PRUNED" for row in combined), flush=True)



def incumbent_comparison(records, leaderboard):
    """Use release-verified validation metrics and identical holdout fingerprints."""
    root=Path(__file__).resolve().parents[1]
    output={}
    for market in MARKETS:
        path=root/"models"/f"neural-experimental-{market}-cvar95"/"manifest.json"
        incumbent=json.loads(path.read_text())
        if incumbent.get("market")!=market or incumbent.get("status")!="release_candidate_verified":
            raise ValueError("Original "+market+" benchmark is not release-verified")
        baseline=incumbent["historical_validation"]
        expected_hash=incumbent["original_holdout_row_sha256"]["validation"]
        for row in records:
            if row["market"]==market and row["state"]=="COMPLETE":
                actual=row["provenance"]["holdout_row_sha256"]["validation"]
                if actual != expected_hash:
                    raise ValueError("Cannot compare "+market+" to a different validation cohort")
        current=leaderboard[market][0] if leaderboard[market] else None
        metrics={}
        for label in ("CVaR95_absolute_USD","MAE_USD","RMSE_USD"):
            old=float(baseline[label])
            new=float(current[label]) if current else None
            metrics[label]={"incumbent":old,"challenger":new,
                            "saved_usd":old-new if new is not None else None}
        output[market]={"existing_model_trial":incumbent["trial"],
                        "historical_validation_cohort_sha256":expected_hash,
                        "challenger":current,"metrics":metrics,
                        "challenger_beats_incumbent_on_CVaR95":bool(
                            current and float(current["CVaR95_absolute_USD"])<
                            float(baseline["CVaR95_absolute_USD"]))}
    return output


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--history", default=None)
    p.add_argument("--output", required=True)
    t = sub.add_parser("train")
    t.add_argument("--plan", required=True)
    t.add_argument("--history", default=None)
    t.add_argument("--market", choices=MARKETS, required=True)
    t.add_argument("--slot", type=int, required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--ask-features", default="training-cache/neural-ask/features.npz")
    t.add_argument("--ask-embeddings", default="training-cache/bge-256")
    t.add_argument("--bid-features", default="training-cache/neural-bid/features.npz")
    t.add_argument("--bid-baseline", default="benchmark-reference/bid/predictions.csv.gz")
    t.add_argument("--max-seconds", type=float, default=3000)
    c = sub.add_parser("collect")
    c.add_argument("--plan", required=True)
    c.add_argument("--history", default=None)
    c.add_argument("--artifacts", required=True)
    c.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "propose":
        propose(args.stage, args.history, args.output)
    elif args.command == "train":
        train(args.plan, args.market, args.slot, args.output, args)
    else:
        collect(args.plan, args.history, args.artifacts, args.output)


if __name__ == "__main__":
    main()
