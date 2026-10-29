"""CPU-only BART-base-CNN smoke test on the same stories as SmolLM2."""
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
import re
from test_smollm_summary import cases,story,ROOT

MODEL='ainize/bart-base-cnn'
REVISION='b90bc9a7c93de6449a8c531ed5f957d84649b99a'


def main():
    import torch,transformers
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
    torch.set_num_threads(2);torch.manual_seed(42)
    out=ROOT/'reports/bart-base-cnn-cpu';out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    # Publisher supplies a PyTorch state_dict, not safetensors. Restrict unpickling.
    model=AutoModelForSeq2SeqLM.from_pretrained(MODEL,revision=REVISION,
                     use_safetensors=False,weights_only=True,trust_remote_code=False,
                     dtype=torch.float32).to('cpu').eval()
    load_seconds=time.perf_counter()-start
    entries,dataset=cases();results=[]

    def one(text):
        inputs=tokenizer(text,return_tensors='pt',truncation=False)
        if inputs['input_ids'].shape[1]>1024:raise ValueError('Input over encoder budget')
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=128,min_length=0,min_new_tokens=0,
                      num_beams=1,do_sample=False,use_cache=True,no_repeat_ngram_size=3,
                      forced_eos_token_id=None)
        new=output[0,1:]
        return tokenizer.decode(output[0],skip_special_tokens=True).strip(),len(new),new[-1].item()==tokenizer.eos_token_id

    for row in entries:
        source=story(row['fields'],row['text'])
        ids=tokenizer.encode(source,add_special_tokens=False)
        chunks=[tokenizer.decode(ids[i:i+960],skip_special_tokens=True) for i in range(0,len(ids),960)]
        timings=[];summaries=[];finished=[];generated=[]
        for _ in range(2):
            t=time.perf_counter();steps=[one(c) for c in chunks]
            if len(steps)>1:
                merged=' '.join(s[0] for s in steps)
                # At these input sizes the per-chunk outputs fit a single merge call.
                final=one(merged);steps.append(final)
            else:final=steps[0]
            timings.append(time.perf_counter()-t);summaries.append(final[0]);generated.append(sum(s[1] for s in steps));finished.append(all(s[2] for s in steps))
        summary=summaries[-1];lower=summary.lower()
        numbers=lambda t:set(re.findall(r'\d+(?:\.\d+)?',t.replace(',','')))
        results.append({**row,'story':source,'summary':summary,'input_tokens':len(ids),'source_chunks':len(chunks),
            'timings_seconds':timings,'new_tokens':generated,'eos_finished':finished,
            'repeated_output_identical':summaries[0]==summaries[1],
            'missing_keyword_groups':[g for g in row['required_groups'] if not any(v.lower() in lower for v in g)],
            'forbidden_phrase_hits':[v for v in row['forbidden'] if v.lower() in lower],
            'unsupported_digit_numbers':sorted(numbers(summary)-numbers(source))})
        print(row['id'],round(timings[-1],3),summary,flush=True)
        (out/'partial.json').write_text(json.dumps(results,indent=2))
    times=[t for r in results for t in r['timings_seconds']]
    report=dict(model=MODEL,revision=REVISION,device='cpu',dtype='float32',threads=2,
      parameters=sum(p.numel() for p in model.parameters()),load_seconds=load_seconds,
      peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
      latency_median_seconds=statistics.median(times),latency_min_seconds=min(times),latency_max_seconds=max(times),
      deterministic_cases=sum(r['repeated_output_identical'] for r in results),
      completed_outputs=sum(sum(r['eos_finished']) for r in results),total_outputs=len(times),results=results,
      dataset_attribution={k:dataset[k] for k in ['source','creator','license','license_url','modifications']},
      limitations='Small nonrandom smoke test; keyword checks are not semantic accuracy. Greedy FP32 CPU, two threads, 128-token output cap. No GPU or live deployment. Loading includes first download. RAM is process RSS including runtime.')
    (out/'results.json').write_text(json.dumps(report,indent=2))
    print('REPORT',json.dumps({k:v for k,v in report.items() if k not in ['results','dataset_attribution']}),flush=True)

if __name__=='__main__':main()
