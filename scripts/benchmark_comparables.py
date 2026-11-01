"""Engineering and self-retrieval audit; not a human relevance benchmark."""
import json,time,resource
from pathlib import Path
from pricing.comparable_retrieval import ComparableIndex,reference_rows

def run(root):
    report={'evaluation':'Self-retrieval and structural guardrails only; no judged NDCG or relevance superiority claim. Reference prices are excluded.','results':[]}
    for market in ('ask','bid'):
        rows=reference_rows(root,market)
        for dims in (32,64,128):
            start=time.perf_counter();index=ComparableIndex(rows,dims);fit=time.perf_counter()-start
            start=time.perf_counter();found=0;violations=0
            for q in index.rows[::max(1,len(index.rows)//40)][:40]:
                matches=index.search(q,exclude_id=q.get('id'))
                found+=bool(matches)
                violations+=sum(x['vehicle']['manufacturer'].lower()!=q['manufacturer'].lower() or abs(x['vehicle']['year']-q['year'])>5 for x in matches)
            report['results'].append({'market':market,'requested_dimensions':dims,'actual_dimensions':index.dimensions,'rows':len(index.rows),'fit_seconds':fit,'query_ms_mean':1000*(time.perf_counter()-start)/40,'queries_with_comparisons':found,'guardrail_violations':violations,'vector_bytes':index.vectors.nbytes})
    report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return report
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];out=root/'reports/comparable-retrieval.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(run(root),indent=2));print(out.read_text())
