import json
from pathlib import Path
rows=[]
for arm in ['baseline','bge']:
 p=Path('experiment-results')/arm/'ask/bayesian/bayesian-evaluation.json'
 r=json.loads(p.read_text())
 rows.append({'Model':arm,'MAE ($)':r['point_scores']['mae_usd'],'90% coverage':r['test_coverage'],'Test cars':r['test_rows']})
current=json.loads(Path('models/xgboost-hierarchy/ask/bayesian/bayesian-evaluation.json').read_text())
rows.append({'Model':'Currently deployed (historical reference)','MAE ($)':current['point_scores']['mae_usd'],'90% coverage':current['test_coverage'],'Test cars':current['test_rows']})
out=Path('reports/bge-umap-experiment');out.mkdir(parents=True,exist_ok=True)
(out/'comparison.json').write_text(json.dumps({'comparison':rows,'limits':'Research only. Paired controls retrain MiniLM heads on money-redacted descriptions; current deployment is a historical reference and includes an additional validation calibration. Each arm uses identical retained Lemon rows, five group OOF folds, training-only 32-dimensional cosine UMAP for BGE (30 neighbours, min_dist 0.1, seed 42), AFT and robust Bayesian likelihood selection. Retained-cohort CV is conditional on earlier cleaning; historical test has been examined before. No claim of current-market validation or serving readiness.'},indent=2))
