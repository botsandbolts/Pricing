"""Refit current robust hierarchy against the new, matching Lemon OOF base."""
import argparse,hashlib,json
from pathlib import Path
from scripts.compare_robust_bayes import prepare,run
from pricing.bayesian_hierarchy import fit,assess,save
p=argparse.ArgumentParser();p.add_argument('--market',required=True);a=p.parse_args()
side=a.market;root=Path('training-cache/lemon-hierarchy');folder=root/side
data=prepare(folder/'hierarchy-input.csv.gz',side)
meta,draws=fit(data.loc[data.split.eq('calibration')])
report,_,_=assess(meta,draws,data.loc[data.split.eq('validation')],data.loc[data.split.eq('test')])
report['base_test_mae_usd']=float(abs(data.loc[data.split.eq('test'),'observed_price']-data.loc[data.split.eq('test'),'base_price']).mean())
meta['base_evaluation_sha256']=hashlib.sha256((folder/'evaluation.json').read_bytes()).hexdigest()
save(folder/'bayesian',meta,draws,report)
inputs=root/'inputs';inputs.mkdir(exist_ok=True)
data.to_csv(inputs/(side+'-minilm-cleaned.csv.gz'),index=False)
run(side,inputs/(side+'-minilm-cleaned.csv.gz'),root/'robust',root,240,240)
comparison=json.loads((root/'robust'/side/'comparison.json').read_text())
if 'selected_likelihood' not in comparison:raise ValueError('No robust candidate passed convergence')
import shutil
shutil.rmtree(folder/'bayesian')
shutil.copytree(root/'robust'/side/'bayesian',folder/'bayesian')
print('Matching robust hierarchy fitted',side,comparison['selected_likelihood'],flush=True)
