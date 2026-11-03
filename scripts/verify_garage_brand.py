"""Full installed-catalogue UI verification for Drive Chemistry; no retraining."""
from pathlib import Path
import json
from scripts.taste_ui_check import exercise_taste_ui
from pricing.taste_ranking import load
def main():
    reports={}
    for k in [10,25]:
        meta,arrays=load(Path('.'),k)
        assert meta['density_method']=='direct-weighted-manhattan'
        reports[str(k)]={'rows':meta['rows'],'ui':exercise_taste_ui(Path('.'),k,[1+i%4 for i in range(k)])}
    Path('drive-chemistry-ui-check.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
