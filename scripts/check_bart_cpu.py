"""Real installed BART smoke, outside the export process, with both pricing models loaded."""
import json,resource,time,sys
from pathlib import Path
from datetime import date
from pricing.hierarchy_sparse_live import estimate
from pricing.current_tools import build_report
from pricing.bart_summary import SummaryEngine,summarize_story,qualitative_report_source,valuation_receipt
fields=dict(year=2015,manufacturer='Toyota',model='Corolla',odometer=95000,state='ca')
results,_,_=estimate(Path('.'),{**fields,'text':'Well maintained. Needs new tires.'},date.today(),True)
for side,r in results.items():assert 'error' not in r,(side,r)
report=build_report(Path('.'),fields,date.today(),results)
# Match production: finish pricing, then release cached pricing weights before BART.
from pricing.hierarchy_sparse_live import _load
import gc
_load.clear();gc.collect()
engine=SummaryEngine()
assert engine.onnx is not None
assert 'torch' not in sys.modules
entries=[]
for description in ['Well maintained. Needs new tires.','New brakes. Air conditioning does not work.']:
    story=qualitative_report_source(fields,description,report)
    started=time.perf_counter();summary=summarize_story(story,engine)
    if summary['status']=='review_required':
        summary=summarize_story(story,engine,qualitative=True)
    assert summary['status']=='ready',summary
    entries.append(dict(summary=summary,seconds=time.perf_counter()-started))
    print(json.dumps(entries[-1]),flush=True)
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
assert peak<900,peak
receipt=valuation_receipt(report)
assert all(f"{r['price']:,.0f}" in receipt and f"{r['interval_low']:,.0f}" in receipt and f"{r['interval_high']:,.0f}" in receipt for r in results.values())
path=Path('reports/bart-int8');path.mkdir(parents=True,exist_ok=True)
(path/'smoke.json').write_text(json.dumps({'peak_rss_mib':peak,'results':entries,'receipt':receipt,'limits':'Two real-case CPU smoke checks, not semantic quality or concurrent-user coverage.'},indent=2))
print('REAL BART + ASK + BID VERIFIED; peak MiB',peak,flush=True)
