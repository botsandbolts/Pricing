"""Paired raw-ELECTRA XGBoost assessment after matching Bayesian refits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    rows=[];predictions={};details={};reference=None
    for arm in ['baseline','electra_raw_xgboost']:
        root=Path('experiment-results')/arm
        base=json.loads((root/'ask/evaluation.json').read_text())
        bayes=json.loads((root/'ask/bayesian/bayesian-evaluation.json').read_text())
        source=pd.read_csv(root/'ask/predictions.csv.gz',dtype={'row_id':str,'vehicle_group':str})
        source=source.loc[source.split.eq('test'),['row_id','vehicle_group','observed_price']]
        predicted=pd.read_csv(root/'robust/ask/test-predictions.csv.gz',dtype={'row_id':str})
        joined=source.merge(predicted,on='row_id',validate='one_to_one').sort_values('row_id').reset_index(drop=True)
        if len(joined)!=len(source) or not np.allclose(joined.observed_price,joined.observed):
            raise ValueError('Mismatched hierarchy test predictions')
        identity=joined[['row_id','vehicle_group','observed_price']]
        if reference is None:reference=identity
        elif not reference.equals(identity):raise ValueError('Arms do not use identical test vehicles and targets')
        predictions[arm]=joined.new_bayesian.to_numpy()
        details[arm]={'xgboost':base,'bayesian':bayes}
        rows.append({'Model':arm,'MAE ($)':bayes['point_scores']['mae_usd'],
                     'XGBoost-only MAE ($)':base['scores']['test']['mae_usd'],
                     '90% coverage':bayes['test_coverage'],'Test cars':bayes['test_rows']})
    truth=reference.observed_price.to_numpy()
    improvement=np.abs(truth-predictions['baseline'])-np.abs(truth-predictions['electra_raw_xgboost'])
    grouped=pd.DataFrame({'group':reference.vehicle_group,'saving':improvement}).groupby('group').saving.agg(['sum','count'])
    sums=grouped['sum'].to_numpy();counts=grouped['count'].to_numpy()
    rng=np.random.default_rng(90210);means=[]
    for _ in range(600):
        idx=rng.integers(0,len(grouped),len(grouped));means.append(float(sums[idx].sum()/counts[idx].sum()))
    report={'comparison':rows,'details':details,
            'paired_improvement':{'mean_mae_saved_usd':float(improvement.mean()),
                                 'group_bootstrap_95_usd':np.quantile(means,[.025,.975]).tolist(),
                                 'vehicle_groups':len(grouped)},
            'architecture':'Vehicle facts + MiniLM dollar-price feature in both XGBoost AFT arms. Experimental arm adds all 256 frozen, mean-pooled, L2-normalized ELECTRA coordinates with no dimensionality reduction. Same Bayesian hierarchy structure refitted separately on each arm OOF residuals; no ELECTRA predictors in hierarchy.',
            'limits':'Research only. Identical retained Lemon cohort and five group OOF folds. MiniLM price heads refitted within each outer training fold. Historical test inspected in earlier experiments; retained-cohort CV conditional on earlier cleaning. No current-market or serving-readiness claim.'}
    dest=Path('reports/electra-raw-xgboost-experiment');dest.mkdir(parents=True,exist_ok=True)
    (dest/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
