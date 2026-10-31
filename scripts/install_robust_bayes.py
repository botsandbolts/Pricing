"""Recheck promotion and matching base identity; replace only residual posteriors."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np

def install(source,root):
    source=Path(source);root=Path(root);installed=[]
    for side in ['ask','bid']:
        report=json.loads((source/side/'comparison.json').read_text());evidence=Path('reports/robust-bayesian')/side;evidence.mkdir(parents=True,exist_ok=True)
        (evidence/'comparison.json').write_text(json.dumps(report,indent=2))
        if not report.get('promoted'):print('RETAIN previous',side,report.get('reason','checks failed'),flush=True);continue
        current=root/side;digest=hashlib.sha256((current/'evaluation.json').read_bytes()).hexdigest()
        selected=next(c for c in report['candidates'] if c['likelihood']==report['selected_likelihood'])
        if not (digest==report['base_model_identity'] and selected['max_split_rhat']<1.1 and .88<=report['test_coverage']<=.95 and
                report['improvement_over_previous']['group_bootstrap_95_usd'][0]>0 and report['improvement_over_regular']['group_bootstrap_95_usd'][0]>0 and
                selected['validation_mae_usd']<report['validation_previous_bayes_mae_usd']):raise ValueError('Promotion gate failed: '+side)
        from pricing.bayesian_hierarchy import load
        meta,arrays=load(source/side/'bayesian')
        if meta.get('base_evaluation_sha256')!=digest or not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Invalid/mismatched posterior')
        shutil.rmtree(current/'bayesian');shutil.copytree(source/side/'bayesian',current/'bayesian');installed.append(side)
    release=json.loads((root/'release.json').read_text());release['bayesian_redesign_markets']=installed
    release['bayesian_method']='validation-selected cross-classified nonlinear heteroscedastic log residuals';(root/'release.json').write_text(json.dumps(release,indent=2))
    print('PROMOTED',installed,flush=True);return installed

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root',default='models/xgboost-hierarchy');a=p.parse_args();install(a.source,a.root)
