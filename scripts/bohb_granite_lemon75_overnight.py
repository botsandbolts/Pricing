"""Overnight 10-shard HpBandSter BOHB with IBM Granite 97M R2 INT8 ONNX Ask.

Each shard is a real BOHB KDE + successive-halving optimizer. Stages retain
trial histories and the best internally selected native FP32 checkpoint. The
10 independent shards operate in parallel; no information leaks from external
validation back into any trial or budget promotion. Final validation is a
retrospective, 75%-retained Lemon conditional analysis, not prospective proof.
The untouched original test is never used to select, tune or audit trials.
No production deployment is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

import scripts.bohb_aft_ask_30m as bohb
from scripts.granite_lemon75_features import frozen_lemon_data
from scripts.ask25_joint_search import ResearchAskingNetwork
from scripts.neural_ask_train import predict
from scripts.granite_lemon75_features import frozen_source
from scripts.cvar95_historical import risk
from scripts.granite_lemon75_encode import VARIANT, MODEL_ID, MODEL_REV, INT8_SHA256

BUDGETS = (3, 9, 27, 81, 250)
DEADLINE = datetime(2026, 11, 9, 5, 0, tzinfo=timezone.utc)  # 06:00 CET
VALIDATION_ROWS = 31668
RETAINED_VALIDATION = 23751
RETAINED_IDS_SHA256 = "aeccb9ddbcf103d8bc603659c1aa3f62efe39d6b3d5f9f5451e3b5a313ca27ac"


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


def _id_hash(values):
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def _promote_or_sample(history, generator):
    """Successive-halving 3x promotions up to a hard ceiling of 250 epochs."""
    completed = [r for r in history if r.get("status") == "complete"]
    for lower, upper in reversed(list(zip(BUDGETS[:-1], BUDGETS[1:]))):
        lower_rows = [r for r in completed if r["epochs"] == lower]
        higher_rows = [r for r in completed if r["epochs"] == upper]
        if len(lower_rows) < 3 or len(higher_rows) >= len(lower_rows) // 3:
            continue
        already = {json.dumps(r["params"], sort_keys=True) for r in history
                   if r["epochs"] == upper}
        remaining = [r for r in lower_rows
                     if json.dumps(r["params"], sort_keys=True) not in already]
        if remaining:
            winner = min(remaining,
                         key=lambda r: r["inner_metrics"]["CVaR95_absolute_USD"])
            return dict(winner["params"]), upper, winner
    return bohb._unique_config(generator, history), BUDGETS[0], None


def _validate_contract(feature_path, embeddings_path):
    values, meta = frozen_lemon_data({
        "features": str(feature_path), "embeddings": str(embeddings_path)
    })
    assert meta["fit_rows"] == 99749 and meta["train_only_inner_rows"] == 11083
    assert meta["retained_train_rows"] == 110832
    assert meta["trimmed_train_rows"] == 36945
    assert meta["original_validation_rows"] == VALIDATION_ROWS
    assert meta["original_test_rows"] == 31682
    return values, meta


def _retained_validation_mask(z, reference_file):
    frame = pd.read_csv(reference_file, dtype={"row_id": str},
                        usecols=["row_id", "split", "observed_price", "residual_log"])
    if frame.row_id.duplicated().any() or frame.row_id.isna().any():
        raise ValueError("Lemon reference row IDs are missing or duplicated")
    va = frame.loc[frame.split.eq("validation")].copy()
    if len(va) != VALIDATION_ROWS or not np.isfinite(va.residual_log.to_numpy(float)).all():
        raise ValueError("Lemon reference validation changed")
    va["abs_ref"] = va.residual_log.abs()
    va = va.sort_values(["abs_ref", "row_id"], ascending=[False, True])
    retained = va.iloc[math.ceil(.25 * VALIDATION_ROWS):].row_id.astype(str)
    if len(retained) != RETAINED_VALIDATION or _id_hash(retained) != RETAINED_IDS_SHA256:
        raise ValueError("Cleaned Lemon validation membership differs from pinned audit")
    phase = z["phase"].astype(str)
    selected = np.flatnonzero(phase == "validation")
    ids = z["ids"][selected].astype(str)
    if len(selected) != VALIDATION_ROWS or set(ids) != set(va.row_id):
        raise ValueError("Original full validation row membership changed")
    reference_y = va.set_index("row_id").loc[ids].observed_price.to_numpy(float)
    if not np.allclose(reference_y, z["y"][selected], rtol=1e-6, atol=.005):
        raise ValueError("Reference and original validation labels differ")
    mask = np.isin(ids, retained.to_numpy())
    if int(mask.sum()) != RETAINED_VALIDATION:
        raise ValueError("The independent 25% Lemon validation trim failed")
    return selected, mask


def _validation_report(args, out, best, meta):
    """One final, read-only evaluation after TRAIN-internal model selection."""
    z, vectors, provenance, encoder = frozen_source(
        args.features, args.embeddings, VARIANT)
    if (provenance["cohort_sha256"] != meta["cohort_sha256"]
            or encoder["embedding_file_sha256"] != meta["embedding_sha256"]):
        raise ValueError("Frozen cohort or Granite INT8 embedding fingerprint changed")
    selected, kept = _retained_validation_mask(z, args.reference)
    pack = torch.load(out / "weights.pt", map_location="cpu", weights_only=True)
    if pack["cohort_sha256"] != meta["cohort_sha256"]:
        raise ValueError("Resume checkpoint cohort mismatch")
    arch = bohb.architecture_params("deepcross")
    model = ResearchAskingNetwork(provenance["cardinalities"], 384, arch)
    model.load_state_dict(pack["state_dict"], strict=True)
    model.eval()
    sample = TensorDataset(
        torch.from_numpy(z["num"][selected].astype(np.float32)),
        torch.from_numpy(z["cat"][selected].astype(np.int64)),
        torch.from_numpy(vectors[selected].astype(np.float32)),
        torch.from_numpy(np.log(z["y"][selected]).astype(np.float32)),
    )
    torch.set_num_threads(2)
    predictions = predict(model, sample, np.arange(len(selected)), 1024)
    actual = z["y"][selected].astype(float)
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite original-validation prices")
    full = risk(actual, predictions)
    cleaned = risk(actual[kept], predictions[kept])
    pd.DataFrame({
        "row_id": z["ids"][selected].astype(str),
        "split": "validation",
        "observed_price": actual,
        "predicted_price": predictions,
        "retained_after_reference_lemon_trim": kept,
    }).to_csv(out / "validation.csv.gz", index=False)
    return {
        "full_original_validation": {
            k: float(full[k]) for k in
            ("CVaR95_absolute_USD", "MAE_USD", "RMSE_USD")
        },
        "cleaned_conditional_validation": {
            k: float(cleaned[k]) for k in
            ("CVaR95_absolute_USD", "MAE_USD", "RMSE_USD")
        },
        "full_original_validation_rows": VALIDATION_ROWS,
        "cleaned_conditional_validation_rows": RETAINED_VALIDATION,
        "cleaned_reference_ids_sha256": RETAINED_IDS_SHA256,
        "cleaned_cohort_is_retrospective_outcome_dependent": True,
        "cleaned_cohort_not_valid_prospective_accuracy_claim": True,
        "test_evaluated": False,
        "automatic_deployment": False,
    }


def run(args):
    if args.slot not in range(10) or args.stage not in (1, 2, 3):
        raise ValueError("Exactly ten shards and three stages are supported")
    if not 60 <= args.phase_seconds <= 16000:
        raise ValueError("Phase length out of safe range")
    bohb.BUDGETS = BUDGETS
    bohb.SEED = 20261108 + args.slot * 193
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    remaining_until_deadline = max(0., (DEADLINE - now).total_seconds())
    search_seconds = min(float(args.phase_seconds), remaining_until_deadline)
    stop = time.monotonic() + search_seconds
    trial_stop = stop - min(240., max(0., search_seconds * .04))
    values, meta = _validate_contract(args.features, args.embeddings)
    previous = Path(args.previous_dir) if args.previous_dir else None
    history = []
    best = None
    if args.stage != 1:
        if previous is None or not (previous / "history.json").is_file():
            raise FileNotFoundError("Previous BOHB stage history is required")
        previous_history = json.loads((previous / "history.json").read_text())
        if (previous_history.get("slot") != args.slot
                or previous_history.get("stage") != args.stage - 1
                or previous_history.get("cohort_sha256") != meta["cohort_sha256"]):
            raise ValueError("Stage continuation does not match its source")
        history = previous_history["trials"]
        if (previous / "weights.pt").is_file():
            pack = torch.load(previous / "weights.pt", map_location="cpu",
                              weights_only=True)
            if pack["cohort_sha256"] != meta["cohort_sha256"]:
                raise ValueError("Previous winner data fingerprint mismatch")
            best = pack["best_record"]
            torch.save(pack, out / "weights.pt")
    overview = {
        "status": "running", "stage": args.stage, "slot": args.slot,
        "architecture": "deepcross", "training_cohort": "frozen-lemon-oof-retained-75",
        "text_encoder_model": MODEL_ID,
        "text_encoder_revision": MODEL_REV,
        "text_encoder_runtime": "verified INT8 ONNX CPU",
        "text_encoder_quantized_sha256": INT8_SHA256,
        "compared_against": "BGE-small-en-v1.5 frozen 256-token / 384D",
        "optimizer": "HpBandSter true BOHB KDE with successive-halving promotions",
        "epoch_budgets": list(BUDGETS), "max_epochs_per_candidate": 250,
        "hard_cutoff_utc": DEADLINE.isoformat(), "hard_cutoff_cet": "2026-11-09 06:00 CET",
        "max_parallel_shards": 10,
        "original_train_rows": 147777, "retained_train_rows": 110832,
        "internal_train_fit_rows": 99749, "internal_train_stopper_rows": 11083,
        "full_validation_rows": 31668, "cleaned_validation_rows": 23751,
        "cohort_sha256": meta["cohort_sha256"],
        "embedding_sha256": meta["embedding_sha256"],
        "encoder_weights_int8_onnx_sha256": INT8_SHA256,
        "selection_uses_validation": False, "test_scored": False,
        "autodeploy": False, "study_start_utc": now.isoformat(),
    }
    print("LEMON75_BOHB_OVERNIGHT_START", json.dumps(overview), flush=True)
    while time.monotonic() < trial_stop - 90.:
        generator = bohb.bohb_generator(history, seed=bohb.SEED)
        hp, budget, parent = _promote_or_sample(history, generator)
        trial_id = len(history)
        seed = (parent["seed"] if parent is not None
                else bohb.SEED + 1000 + trial_id * 37)
        record = {
            "trial": trial_id, "params": hp, "epochs": int(budget),
            "promoted_from": parent["trial"] if parent is not None else None,
            "seed": int(seed), "stage": args.stage
        }
        time_left = trial_stop - time.monotonic()
        if time_left < 90.:
            break
        try:
            trained = bohb._trial(
                values, meta["cardinalities"],
                bohb.architecture_params("deepcross"),
                hp, budget, seed, trial_stop,
                max_single=max(35., time_left - 35.))
            result_epochs = int(trained["epochs_trained"])
            record.update({
                k: v for k, v in trained.items() if k != "weights"
            })
            if result_epochs < budget:
                record["status"] = "partial"
            if best is None or (
                record["inner_metrics"]["CVaR95_absolute_USD"] <
                best["inner_metrics"]["CVaR95_absolute_USD"]
            ):
                best = dict(record)
                torch.save({
                    "state_dict": trained["weights"], "best_record": best,
                    "architecture": bohb.architecture_params("deepcross"),
                    "cohort_sha256": meta["cohort_sha256"],
                    "embedding_sha256": meta["embedding_sha256"],
                    "training_cohort": "frozen-lemon-oof-retained-75",
                    "test_scored": False, "autodeploy": False,
                }, out / "weights.pt")
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            print("LEMON75_BOHB_TRIAL_FAILED", json.dumps(record), flush=True)
        history.append(record)
        save_json(out / "history.json", {
            **overview, "trials": history,
            "best_train_internal_CVaR95_USD":
                best["inner_metrics"]["CVaR95_absolute_USD"] if best else None,
        })
        print("LEMON75_BOHB_TRIAL", json.dumps(record), flush=True)
    state = {
        **overview, "status": "stage_complete",
        "trials": history,
        "stage_finished_utc": datetime.now(timezone.utc).isoformat(),
        "total_trials": len(history),
        "completed_trials": sum(x.get("status") == "complete" for x in history),
        "partially_trained_trials": sum(x.get("status") == "partial" for x in history),
        "failed_trials": sum(x.get("status") == "failed" for x in history),
        "highest_epoch_budget_fully_completed": max(
            (r["epochs"] for r in history if r.get("status") == "complete"),
            default=0
        ),
        "winning_train_selected_trial": best,
    }
    save_json(out / "history.json", state)
    if best is None:
        state["status"] = "no_valid_checkpoint"
        save_json(out / "result.json", state)
        raise RuntimeError("BOHB stage ended without any valid native checkpoint")
    if args.stage == 3:
        if not args.reference:
            raise ValueError("Stage 3 requires pinned original Lemon validation reference")
        state.update(_validation_report(args, out, best, meta))
        state["status"] = "completed"
        save_json(out / "result.json", state)
        print("LEMON75_BOHB_FINAL_VALIDATION", json.dumps({
            "slot": args.slot, "best_training_trial": best,
            "full_validation": state["full_original_validation"],
            "cleaned_validation": state["cleaned_conditional_validation"],
            "max_epoch_budget_reached": state["highest_epoch_budget_fully_completed"],
        }), flush=True)
    else:
        save_json(out / "result.json", state)
    print("LEMON75_BOHB_STAGE_COMPLETE", json.dumps({
        "slot": args.slot, "stage": args.stage,
        "trials": len(history), "highest_budget": state["highest_epoch_budget_fully_completed"],
        "deadline_utc": DEADLINE.isoformat(),
    }), flush=True)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slot", type=int, required=True)
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--features", default="training-cache/neural-ask/features.npz")
    p.add_argument("--embeddings", default="training-cache/granite")
    p.add_argument("--reference", default="")
    p.add_argument("--previous-dir", default="")
    p.add_argument("--phase-seconds", type=int, default=14400)
    p.add_argument("--output", required=True)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.check:
        assert BUDGETS[-1] == 250 and len(BUDGETS) == 5
        assert DEADLINE == datetime(2026, 11, 9, 5, tzinfo=timezone.utc)
        assert bohb.architecture_params("deepcross")["family"] == "dcnv2"
        print("BOHB_OVERNIGHT_STATIC_PREFLIGHT_OK", BUDGETS, DEADLINE)
    else:
        run(args)


if __name__ == "__main__":
    main()
