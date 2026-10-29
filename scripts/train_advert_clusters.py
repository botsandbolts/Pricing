"""Fit train-only advert groups and diagnose frozen chronological holdouts."""
import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pricing.advert_clusters import clean_description, encode, fit


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('cohort', help='JSONL produced by scripts.prepare_all')
    parser.add_argument('--output', default='models/advert-clusters')
    parser.add_argument('--max-train', type=int, default=20000)
    parser.add_argument('--holdout', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--diagnostics-only', action='store_true', help='Recheck the installed frozen fit without refitting')
    args = parser.parse_args()
    started = time.monotonic()
    data = pd.read_json(args.cohort, lines=True)
    # The upstream cohort's original chronological phases are preserved.
    data['text'] = data.text.fillna('').map(clean_description)
    data = data[data.text.str.split().str.len() >= 5].copy()
    data['priority'] = data.row_id.map(lambda x: hashlib.sha256(f'{args.seed}:{x}'.encode()).hexdigest())
    training = data[data.phase == 'train'].sort_values('priority').drop_duplicates('text')
    available = len(training)
    if args.max_train:
        training = training.head(args.max_train)
    print(('Rechecking frozen fit' if args.diagnostics_only else f'Fitting {len(training):,} train-only descriptions with mixed features'), flush=True)
    if args.diagnostics_only:
        from pricing.advert_clusters import load_bundle
        bundle = load_bundle(args.output)
        report = json.loads((Path(args.output)/'report.json').read_text())
        if report['seed'] != args.seed:
            raise ValueError('Diagnostics seed differs from the installed fit')
    else:
        bundle, report = fit(training, seed=args.seed)
    print(f"Found {report['cluster_count']} clusters; measuring frozen holdout assignments", flush=True)
    import hdbscan
    holdouts = {}
    seen = set(data.loc[data.phase == 'train', 'text'])
    for phase in ('validation', 'test'):
        cohort = data[(data.phase == phase) & ~data.text.isin(seen)].sort_values('priority').drop_duplicates('text').head(args.holdout)
        features, _, usable = encode(bundle, cohort)
        labels = np.full(len(cohort), -1, dtype=int)
        strengths = np.zeros(len(cohort))
        if usable.any():
            embedding = bundle['reducer'].transform(features[usable])
            labels[usable], strengths[usable] = hdbscan.approximate_predict(bundle['clusterer'], embedding)
        assigned = (labels >= 0) & (strengths >= bundle['min_strength'])
        holdouts[phase] = dict(rows=len(cohort), assigned=int(assigned.sum()),
                               assignment_fraction=float(assigned.mean()) if len(cohort) else None,
                               vocabulary_usable_fraction=float(usable.mean()) if len(cohort) else None,
                               median_assigned_strength=float(np.median(strengths[assigned])) if assigned.any() else None,
                               clusters_reached=len(set(labels[assigned])))
    report.update(source='Tom, Used Cars Dataset, Zenodo 10457828, CC BY 4.0',
                  source_url='https://zenodo.org/records/10457828', seed=args.seed,
                  available_unique_training_descriptions=available,
                  training_selection='Deterministic SHA256(seed:row_id) ranking within original train phase',
                  original_split='scripts.prepare_all chronological 70/15/15; original eligibility and VIN/text deduplication',
                  holdout_selection='Deterministic samples; normalized descriptions occurring anywhere in train excluded',
                  holdouts=holdouts, price_features_used=False,
                  limitations=['Unsupervised groups, no validated dealer/private labels or classification accuracy.',
                               'Coverage is not accuracy; membership strength is not calibrated probability.',
                               'UMAP distorts density; clusters may follow boilerplate, geography or vehicle makes.',
                               'Weights and rejection threshold are explicit heuristics, not supervised optima.',
                               'Historical Craigslist asking adverts, not auction bid descriptions.'],
                  diagnostic_run_seconds=round(time.monotonic()-started, 2))
    report['projection'] = 'Frozen UMAP embedding; exact weighted Hellinger reference distances; UMAP precomputed-distance transform'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    artifact = output/'model.joblib'
    joblib.dump(bundle, artifact, compress=3)
    versions = {key:version(key) for key in ('numpy','scipy','scikit-learn','umap-learn','hdbscan','numba','llvmlite','pynndescent','joblib')}
    report['versions'] = versions
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    manifest = dict(schema_version=1, status='ready', sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    bytes=artifact.stat().st_size, versions=versions, training_rows=report['training_rows'],
                    cluster_count=report['cluster_count'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    from pricing.suite_storage import split_model
    split_model(artifact, part_bytes=512*1024)
    print(json.dumps(dict(manifest=manifest, holdouts=holdouts), indent=2), flush=True)


if __name__ == '__main__':
    main()
