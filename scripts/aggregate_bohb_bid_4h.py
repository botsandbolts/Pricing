"""Aggregate saved four-hour Bid BOHB research shards without ML dependencies.

The training jobs have already selected their own winners on inner TRAIN CVaR95.
This script only ranks those saved winners; original validation is a diagnostic
receipt, never a selection criterion. No TEST access and no deployment.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SHARDS = 5
ARTIFACT_PREFIX = "bohb-bid-4h-shard-"
METRIC = "internal_CVaR95_USD"


def _finite_positive(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite, got {value!r}")
    return value


def aggregate(root: Path, output: Path, source_run_id: str = "") -> dict:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Collected artifact directory missing: {root}")

    winners = []
    source_hashes = set()
    for shard in range(SHARDS):
        artifact = root / f"{ARTIFACT_PREFIX}{shard}"
        if not artifact.is_dir():
            raise FileNotFoundError(f"Missing BOHB research shard artifact: {artifact}")
        winner_paths = list(artifact.rglob("selected/winner.json"))
        if len(winner_paths) != 1:
            raise ValueError(f"Expected one winner.json in {artifact}; found {len(winner_paths)}")
        winner_path = winner_paths[0]
        shard_dir = winner_path.parent.parent
        status_path = shard_dir / "status.json"
        weights_path = winner_path.parent / "weights.pt"
        if not status_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"Missing status or checkpoint for shard {shard}: {shard_dir}")

        status = json.loads(status_path.read_text(encoding="utf-8"))
        result = json.loads(winner_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            raise ValueError(f"Shard {shard} has incomplete status: {status.get('status')!r}")
        if int(status.get("shard", -1)) != shard or int(result.get("shard", -1)) != shard:
            raise ValueError(f"Shard identity mismatch in {artifact}")
        for record in (status, result):
            if record.get("original_test_scored") is not False:
                raise ValueError(f"Shard {shard} must explicitly attest TEST was not scored")
            if record.get("production_modified") is not False:
                raise ValueError(f"Shard {shard} must explicitly attest production was not modified")

        score = _finite_positive(result[METRIC], f"Shard {shard} inner CVaR95")
        status_score = _finite_positive(
            status["best_inner_CVaR95_USD"], f"Shard {shard} status inner CVaR95"
        )
        if not math.isclose(score, status_score, rel_tol=1e-7, abs_tol=0.01):
            raise ValueError(f"Shard {shard}: status and winner inner CVaR95 disagree")
        _finite_positive(
            result["validation"]["CVaR95_absolute_USD"], f"Shard {shard} validation CVaR95"
        )
        if int(result["best_epoch"]) < 1 or not isinstance(result["configuration"], dict):
            raise ValueError(f"Shard {shard} has invalid winner epoch or configuration")
        if result["configuration"].get("loss") != "cvar95_weighted_huber":
            raise ValueError(f"Shard {shard} changed the specified Bid objective")
        if result["configuration"].get("family") != "deepfm":
            raise ValueError(f"Shard {shard} changed the specified DeepFM architecture")
        source_hashes.add(status.get("dataset_sha256"))
        winners.append(result)

    if len(source_hashes) != 1 or None in source_hashes:
        raise ValueError(f"BOHB shard feature-cohort identity drifted: {source_hashes}")
    winners.sort(key=lambda r: (float(r[METRIC]), int(r["shard"])))
    best = winners[0]
    report = {
        "status": "complete",
        "study": "2026-11-09 deployed full non-Lemon CVaR95 Huber DeepFM",
        "algorithm": "TuneBOHB + HyperBandForBOHB",
        "source_run_id": source_run_id,
        "source_artifact_prefix": ARTIFACT_PREFIX,
        "source_dataset_sha256": next(iter(source_hashes)),
        "study_budget_seconds": 14400,
        "max_concurrent_trainable_slots": 10,
        "max_epochs": 250,
        "incumbent_reference": "models/ray-bid-9am-winner/manifest.json",
        "selected_by": "TRAIN-only internal CVaR95",
        "original_test_scored": False,
        "production_modified": False,
        "winner_by_train_internal": best,
        "all_shard_winners": winners,
        "caution": "Original validation is reported, not used to choose a winner; no deployment",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = [
        "# Four-hour Bid BOHB — recovered aggregate",
        "",
        f"Source run: {source_run_id or 'current workflow run'}",
        "Five completed training shards; ranking uses only inner TRAIN CVaR95.",
        "Original validation is shown as a receipt, **not** used for selection.",
        "**No TEST scoring, model deployment, or production checkpoint replacement.**",
        "",
        "| Rank | Shard | Inner TRAIN CVaR95 (USD) | Original validation CVaR95 (USD) | Best epoch |",
        "|---:|---:|---:|---:|---:|",
    ]
    for rank, winner in enumerate(winners, 1):
        lines.append(
            f"| {rank} | {winner['shard']} | "
            f"{winner[METRIC]:,.2f} | "
            f"{winner['validation']['CVaR95_absolute_USD']:,.2f} | "
            f"{winner['best_epoch']} |"
        )
    (output.parent / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("BID_BOHB_4H_ALL_RESULTS", json.dumps({
        "source_run_id": source_run_id,
        "shards": len(winners),
        "winner_shard": best["shard"],
        "winner_internal_CVaR95": round(best[METRIC], 2),
        "winner_original_validation_CVaR95": round(
            best["validation"]["CVaR95_absolute_USD"], 2
        ),
        "best_config": best["configuration"],
        "production_modified": False,
    }), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--source-run-id", default="")
    args = parser.parse_args()
    aggregate(args.aggregate, args.aggregate_output, args.source_run_id)


if __name__ == "__main__":
    main()
