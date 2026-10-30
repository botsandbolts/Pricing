"""Stage light public forest results; retain large fitted forests offline."""
import argparse,hashlib,json,shutil
from pathlib import Path

FILES=['model.json.gz','evaluation.json','examples.csv.gz','base-assessment.json','provenance.json']

def stage(source,output):
    source=Path(source);output=Path(output)
    for side in ['ask','bid']:
        src=source/side;dest=output/side;dest.mkdir(parents=True,exist_ok=True)
        for name in FILES:shutil.copy2(src/name,dest/name)
        spec={'scope':'Full-cohort forest evaluation, hierarchy weights and display examples. Large base forests and complete row-level RF residuals are generated offline by the reproduction command; they are not loaded or installed for live forest inference.',
            'reproduce':f'python -m scripts.train_original_residual_comparison --engine forest --output training-cache/forest-assessment-fit --side {side}',
            'source_artifacts':{}}
        for name in ['base-assessment.ubj','residuals.csv.gz','test_predictions.csv.gz']:
            p=src/name;h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            spec['source_artifacts'][name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
        (dest/'release-scope.json').write_text(json.dumps(spec,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='training-cache/forest-assessment-fit');parser.add_argument('--output',default='models/forest-original');a=parser.parse_args();stage(a.source,a.output)
