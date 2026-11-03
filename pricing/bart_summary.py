"""Local BART-base-CNN abstractive summaries; no LexRank or external inference."""
import hashlib
import re
import threading
from pathlib import Path
from pricing.vehicle_narrative import narrative

MODEL_ID='ainize/bart-base-cnn'
REVISION='b90bc9a7c93de6449a8c531ed5f957d84649b99a'
CHUNK_TOKENS=960
OUTPUT_TOKENS=128
PROMPT=''
PROMPT_SUFFIX=''
# Serialize model loading and generation across sessions under the CPU/RAM budget.
_SUMMARY_WORKER=threading.BoundedSemaphore(1)


class SummaryEngine:
    def __init__(self):
        from pathlib import Path
        folder=Path(__file__).resolve().parents[1]/'models/bart-cpu'
        if (folder/'manifest.json').exists():
            from pricing.bart_onnx import OnnxBartEngine
            self.onnx=OnnxBartEngine(folder)
            return
        self.onnx=None
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        torch.set_num_threads(min(2,torch.get_num_threads()))
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,use_fast=True)
        self.model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID,revision=REVISION,
                                                       use_safetensors=False,weights_only=True,trust_remote_code=False,dtype=torch.float32).to('cpu').eval()
        self.lock=threading.RLock()
        self.numeric_tokens=[[i] for token,i in self.tokenizer.get_vocab().items() if re.search(r'\d',token)]

    def encode(self,text):
        if self.onnx is not None:return self.onnx.encode(text)
        return self.tokenizer.encode(text,add_special_tokens=False)

    def decode(self,tokens):
        if self.onnx is not None:return self.onnx.decode(tokens)
        return self.tokenizer.decode(tokens,skip_special_tokens=True)

    def generate(self,text,forbid_numbers=False):
        if self.onnx is not None:return self.onnx.generate(text,forbid_numbers)
        inputs=self.tokenizer(PROMPT+text+PROMPT_SUFFIX,return_tensors='pt',truncation=False)
        if inputs.input_ids.shape[1]>1024:raise ValueError('Summary input exceeds token budget')
        with self.lock,self.torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=OUTPUT_TOKENS,num_beams=1,min_length=0,min_new_tokens=0,forced_eos_token_id=None,
                                       do_sample=False,use_cache=True,no_repeat_ngram_size=3,
                                       bad_words_ids=self.numeric_tokens if forbid_numbers else None)
        text=self.tokenizer.decode(output[0],skip_special_tokens=True).strip()
        if output[0,-1].item()!=self.tokenizer.eos_token_id:
            ends=list(re.finditer(r'[.!?](?:\s|$)',text))
            if not ends:raise ValueError('BART produced no complete sentence')
            text=text[:ends[-1].end()].strip()
        return text


def build_story(fields,description):
    facts,text=narrative(fields,description)
    return facts+('\nSeller description (unverified):\n'+text if text else '')


def summarize_story(story,engine,qualitative=False):
    """Map/reduce every token; long descriptions are never silently truncated."""
    ids=engine.encode(story)
    if not ids:return {'summary':'','chunks':0,'status':'empty'}
    chunks=[engine.decode(ids[i:i+CHUNK_TOKENS]) for i in range(0,len(ids),CHUNK_TOKENS)]
    generate=(lambda c:engine.generate(c,forbid_numbers=True)) if qualitative else engine.generate
    summaries=[generate(c) for c in chunks]
    if any(not s for s in summaries):raise ValueError('Model returned an empty summary')
    for _ in range(8):
        if len(summaries)==1:break
        merged=engine.encode(' '.join(summaries))
        groups=[engine.decode(merged[i:i+CHUNK_TOKENS]) for i in range(0,len(merged),CHUNK_TOKENS)]
        summaries=[generate(c) for c in groups]
    else:raise ValueError('Summary reduction did not converge')
    summary=summaries[0]
    numbers=lambda text:set(re.findall(r'\d+(?:\.\d+)?',text.replace(',','')))
    unexpected=numbers(summary)-numbers(story)
    unsupported_claim=any(term in summary.lower() and term not in story.lower()
                          for term in ['dealer profit','guaranteed discount','live bid','dealer offer'])
    review=bool(unexpected or unsupported_claim)
    return {'summary':summary if not review else '', 'chunks':len(chunks),
            'status':'review_required' if review else 'ready'}


def valuation_receipt(valuation):
    """Exact prose from the same structured values as the price cards."""
    if not valuation:return ''
    sentences=['Valuation date: '+str(valuation.get('valuation_date','unspecified'))+'.']
    for side in ['ask','bid']:
        r=valuation.get('market_estimates',{}).get(side)
        if not r:continue
        label='Seller asking estimate' if side=='ask' else 'Wholesale auction estimate'
        sentences.append(f"{label}: USD {float(r['estimated_price_usd']):,.0f}.")
        if r.get('interval_low_usd') is not None and r.get('interval_high_usd') is not None:
            level=float(r.get('interval_level',.9))*100
            sentences.append(f"Its {level:g}% historical prediction range, transported to the valuation date, is USD {float(r['interval_low_usd']):,.0f} to USD {float(r['interval_high_usd']):,.0f}.")
    if any(r.get('interval_low_usd') is not None for r in valuation.get('market_estimates',{}).values()):
        sentences.append('These ranges were calibrated on historical outcomes; their coverage at the selected valuation date is not validated.')
    return ' '.join(sentences)


def build_valuation_story(fields,description,valuation=None):
    """Give the news-trained summarizer readable prose, not raw JSON keys."""
    facts,seller=narrative(fields,description)
    if not valuation:return build_story(fields,description)
    sentences=[facts, valuation_receipt(valuation)]
    for side,r in valuation.get('market_estimates',{}).items():
        target='seller advertised asking price' if side=='ask' else 'wholesale auction sale price'
        sentences.append(f"The estimated {target} is {float(r['estimated_price_usd']):,.2f} US dollars.")
        sentences.append(f"Its historical model anchor is {float(r['historical_anchor_usd']):,.2f} dollars, multiplied by {r['age_multiplier']} for vehicle ageing and {r['market_multiplier']} for market movement. Vehicle age at valuation is {r['vehicle_age_years']} years.")
        market=r.get('market_index_status',{})
        if market.get('kind')=='forecast_monthly_index':
            sentences.append(f"The market index for {market.get('month')} is forecast {market.get('horizon')} months beyond observed data, using {market.get('method')}.")
        if r.get('pooled_depreciation'):sentences.append('Sparse or unseen trim uses pooled make and body depreciation.')
        if r.get('newer_generation_analogue'):sentences.append('The auction estimate assumes an unobserved newer-generation analogue.')
    for side in valuation.get('unavailable_markets',[]):sentences.append(f'The {side} estimate is unavailable; no price is invented.')
    sentences.append('The market source is '+str(valuation.get('market_source'))+', observed through '+str(valuation.get('latest_observed_month'))+'.')
    sentences.append('Asking and auction prices have different economic targets; their difference is not dealer profit. Date-adjusted estimates extrapolate historical data and are not verified current sales. Current-date individual prediction interval coverage has not been validated.')
    for section in valuation.get('sections',[]):
        sentences.append(section['title']+': '+section['text'])
    if seller:sentences.append('The seller says, without independent verification: '+seller)
    return '\n'.join(sentences)


def qualitative_report_source(fields,description,valuation):
    """Readable advert/valuation story, with card-rounded figures for numeric checks."""
    facts,seller=narrative(fields,description)
    parts=[facts,valuation_receipt(valuation)]
    if seller:parts.append('The seller states, without independent verification: '+seller)
    parts.append('Car Price Studio estimates this vehicle for '+str(valuation.get('valuation_date'))+'.')
    for side,r in valuation.get('market_estimates',{}).items():
        meaning='seller advertised asking price before negotiation' if side=='ask' else 'wholesale auction sale price estimate'
        parts.append(f"The {meaning} is approximately {float(r['estimated_price_usd']):,.0f} US dollars.")
    parts.append('The asking and auction estimates describe distinct markets and vehicle populations.')
    for section in valuation.get('sections',[]):
        if section['title'] in {'Valuation method','Evaluation technique'}:
            parts.append(section['title']+': '+section['text'])
    parts.append('Historical price models are adjusted for vehicle age, mileage and a monthly market index. Held-out historical test error measures past average dollar mistakes, not a guarantee for this car or a validated current sale price. The original advert and deterministic price cards remain the source of exact facts.')
    return '\n'.join(parts)


def _cached_overview(state,fingerprint,produce,limit=4):
    """Bounded per-session LRU: unchanged estimates never rerun the model."""
    cache=state.setdefault('_bart_overview_cache',{})
    if fingerprint in cache:
        result=cache.pop(fingerprint)
    else:
        result=produce()
    cache[fingerprint]=result
    while len(cache)>limit:
        del cache[next(iter(cache))]
    return result


def render_story(fields,description,valuation=None):
    import streamlit as st
    @st.cache_resource(show_spinner=False,max_entries=1)
    def get_engine(model_id,revision):
        return SummaryEngine()
    # Both narratives stay internal; only the model's checked output is rendered.
    story=build_valuation_story(fields,description,valuation)
    summary_source=qualitative_report_source(fields,description,valuation) if valuation else story
    backend_manifest=Path(__file__).resolve().parents[1]/'models/bart-cpu/manifest.json'
    backend_release=hashlib.sha256(backend_manifest.read_bytes()).hexdigest() if backend_manifest.exists() else 'pytorch-fallback'
    fingerprint=hashlib.sha256(str(('automatic-private-overview-v5-prices-intervals',backend_release,MODEL_ID,REVISION,PROMPT,PROMPT_SUFFIX,CHUNK_TOKENS,OUTPUT_TOKENS,story,summary_source)).encode()).hexdigest()
    st.markdown('### 📖 Your valuation overview · BART' if valuation else '### 📖 Your car overview · BART')
    st.caption('An automatic summary of the advert, valuation and method, with the exact prices and historical ranges below.' if valuation else 'Automatically summarized from the supplied vehicle details and seller description.')
    if valuation:
        st.write(valuation_receipt(valuation))
    def produce():
        with st.spinner('Preparing your car overview…'):
            with _SUMMARY_WORKER:
                try:
                    from pricing.hierarchy_sparse_live import _load
                    _load.clear()
                    import gc
                    gc.collect()
                    return summarize_story(summary_source,get_engine(MODEL_ID,REVISION))
                except Exception as error:
                    import logging
                    logging.getLogger(__name__).exception('BART summary failed during loading or generation')
                    return {'status':'unavailable','summary':''}
                finally:
                    get_engine.clear()
                    import gc
                    gc.collect()
    result=_cached_overview(st.session_state,fingerprint,produce)
    if result['status']=='ready':
        st.write(result['summary'])
    elif result['status']=='review_required':
        st.caption('The AI draft contained an unsupported figure or market claim and was withheld. Your price cards remain available.')
    else:
        st.caption('The BART narrative is temporarily unavailable. Your exact valuation overview is shown above.')
        if st.button('Retry summary',key='bart_retry_'+fingerprint[:16]):
            st.session_state.get('_bart_overview_cache',{}).pop(fingerprint,None)
            st.rerun()
    st.caption('AI-generated from unverified listing information. It can omit defects or misstate facts; check the original advert. The summary never changes the valuation.')
