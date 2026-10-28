import json,argparse,hashlib
from pathlib import Path
from marketbridge.market_clock import fetch_snapshot,backtest
p=argparse.ArgumentParser();p.add_argument('--output',default='models/time-transport');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
d,meta=fetch_snapshot();d.to_csv(out/'market-index.csv',index=False);meta['sha256']=hashlib.sha256((out/'market-index.csv').read_bytes()).hexdigest();(out/'market-index-meta.json').write_text(json.dumps(meta,indent=2))
r,rows=backtest(d);(out/'market-backtest.json').write_text(json.dumps(r,indent=2));rows.to_csv(out/'market-backtest-rows.csv',index=False);print(json.dumps(r,indent=2))
