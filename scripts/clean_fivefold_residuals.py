"""Audit five sequential 5%-of-survivors cuts from genuine CatBoost OOF residuals.

No model is refitted between cuts: this is fixed-score sequential trimming.
Validation/test label selection is diagnostic and cannot establish generalization.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def clean(predictions, output, *, market, source_run):
    d = pd.read_csv(predictions, dtype={'row_id': str, 'vehicle_group': str})
    required = {'row_id', 'vehicle_group', 'split', 'fold', 'observed_price',
                'predicted_price', 'residual_log'}
    if not required.issubset(d.columns) or d.row_id.isna().any() or d.row_id.duplicated().any():
        raise ValueError('Invalid or duplicated five-fold prediction identities')
    if not set(d.split) == {'calibration', 'validation', 'test'}:
        raise ValueError('Expected all three original splits')
    cal = d.split.eq('calibration')
    if not d.loc[cal, 'fold'].isin(range(5)).all() or not d.loc[~cal, 'fold'].eq(-1).all():
        raise ValueError('Calibration must have five held-out folds; validation/test use refit')
    observed = d.observed_price.to_numpy(float)
    predicted = d.predicted_price.to_numpy(float)
    residual = d.residual_log.to_numpy(float)
    if not (np.isfinite(observed).all() and np.isfinite(predicted).all()
            and np.isfinite(residual).all() and (observed > 0).all() and (predicted > 0).all()):
        raise ValueError('Nonfinite or nonpositive price/prediction')
    if not np.allclose(residual, np.log(observed / predicted), atol=1e-6):
        raise ValueError('Residuals do not match log(observed/predicted)')
    active = np.ones(len(d), dtype=bool)
    removed_stage = np.zeros(len(d), dtype=np.int8)
    stages = []
    for stage in range(1, 6):
        counts = {}
        for split in ('calibration', 'validation', 'test'):
            indices = np.flatnonzero(active & d.split.eq(split).to_numpy())
            n = math.ceil(.05 * len(indices))
            if n >= len(indices):
                raise ValueError(f'Insufficient survivors in {split}')
            # Stable identity tie break, independent of file row order.
            ranking = np.lexsort((d.row_id.iloc[indices].to_numpy(str), -np.abs(residual[indices])))
            removed = indices[ranking[:n]]
            active[removed] = False
            removed_stage[removed] = stage
            counts[split] = {'before': len(indices), 'removed': n,
                             'retained': len(indices)-n,
                             'cutoff_absolute_log_residual': float(np.abs(residual[removed]).min())}
        stages.append({'stage': stage, 'counts': counts})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    d['absolute_log_residual'] = np.abs(residual)
    d['removed_stage'] = removed_stage
    d['retained'] = active
    d.to_csv(out / 'row-audit.csv.gz', index=False)
    report = {'market': market, 'source_run': source_run,
              'method': 'Five sequential cuts using fixed CatBoost five-fold OOF residuals',
              'residual': 'log(observed_price/predicted_price)',
              'train_predictions': 'five group-held-out CatBoost folds',
              'validation_test_predictions': 'calibration-only CatBoost refit',
              'refit_between_stages': False, 'fraction_per_stage': .05,
              'stages': stages, 'original_rows': len(d), 'retained_rows': int(active.sum()),
              'removed_fraction': float((~active).mean()), 'splits': {}}
    for split in ('calibration', 'validation', 'test'):
        mask = d.split.eq(split).to_numpy()
        report['splits'][split] = {
            'original_n': int(mask.sum()), 'retained_n': int((mask & active).sum()),
            'original_mae_usd': float(np.mean(np.abs(observed[mask]-predicted[mask]))),
            'retained_mae_usd_selected': float(np.mean(np.abs(observed[mask & active]-predicted[mask & active])))}
    report['warning'] = ('Retained validation/test MAE uses their true prices to select rows and is optimistic. '
                         'Original MAE is for the existing CatBoost, not a retrained model. '
                         'A fresh external holdout and training on retained rows are needed for an improvement claim. '
                         'Large residuals may be genuine unusual vehicles, not data errors.')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--market', required=True, choices=['ask', 'bid'])
    p.add_argument('--source-run', type=int, required=True)
    args = p.parse_args()
    print(json.dumps(clean(args.predictions, args.output, market=args.market,
                           source_run=args.source_run), indent=2), flush=True)
