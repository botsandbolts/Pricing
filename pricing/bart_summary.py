"""Local BART-base-CNN abstractive summaries; no LexRank or external inference."""
import hashlib
import re
import threading
from pricing.vehicle_narrative import narrative

MODEL_ID='ainize/bart-base-cnn'
REVISION='b90bc9a7c93de6449a8c531ed5f957d84649b99a'
CHUNK_TOKENS=960
OUTPUT_TOKENS=128
PROMPT=''
PROMPT_SUFFIX=''


class SummaryEngine:
    def __init__(self):
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
        return self.tokenizer.encode(text,add_special_tokens=False)

    def decode(self,tokens):
        return self.tokenizer.decode(tokens,skip_special_tokens=True)

    def generate(self,text,forbid_numbers=False):
        inputs=self.tokenizer(PROMPT+text+PROMPT_SUFFIX,return_tensors='pt',truncation=False)
        if inputs.input_ids.shape[1]>1024:raise ValueError('Summary input exceeds token budget')
        with self.lock,self.torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=OUTPUT_TOKENS,num_beams=1,min_new_tokens=0,forced_eos_token_id=None,
                                       do_sample=False,use_cache=True,no_repeat_ngram_size=3,
                                       bad_words_ids=self.numeric_tokens if forbid_numbers else None)
        if output[0,-1].item()!=self.tokenizer.eos_token_id:
            raise ValueError('Summary output did not finish within its token budget')
        return self.tokenizer.decode(output[0],skip_special_tokens=True).strip()


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
    return {'summary':summary if not unexpected else '', 'chunks':len(chunks),
            'status':'review_required' if unexpected else 'ready'}


def build_valuation_story(fields,description,valuation=None):
    """Give the news-trained summarizer readable prose, not raw JSON keys."""
    facts,seller=narrative(fields,description)
    if not valuation:return build_story(fields,description)
    sentences=[facts, 'This valuation is for '+str(valuation.get('valuation_date','an unspecified date'))+'.']
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
    sentences.append('Asking and auction prices have different economic targets; their difference is not dealer profit. Date-adjusted estimates extrapolate historical data and are not verified current sales. No validated individual prediction interval is available.')
    for section in valuation.get('sections',[]):
        sentences.append(section['title']+': '+section['text'])
    if seller:sentences.append('The seller says, without independent verification: '+seller)
    return '\n'.join(sentences)


def qualitative_report_source(fields,description,valuation):
    """Readable advert/valuation story, with card-rounded figures for numeric checks."""
    facts,seller=narrative(fields,description)
    parts=[facts]
    if seller:parts.append('The seller states, without independent verification: '+seller)
    parts.append('Car Price Studio estimates this vehicle for '+str(valuation.get('valuation_date'))+'.')
    for side,r in valuation.get('market_estimates',{}).items():
        meaning='seller advertised asking price before negotiation' if side=='ask' else 'wholesale auction estimate, not a live bid or dealer offer'
        parts.append(f"The {meaning} is approximately {float(r['estimated_price_usd']):,.0f} US dollars.")
    parts.append('The asking and auction estimates describe different markets; their gap is not dealer profit or a guaranteed discount.')
    for section in valuation.get('sections',[]):
        if section['title'] in {'Valuation method','Evaluation technique'}:
            parts.append(section['title']+': '+section['text'])
    parts.append('Historical price models are adjusted for vehicle age, mileage and a monthly market index. Held-out historical test error measures past average dollar mistakes, not a guarantee for this car or a validated current sale price. The original advert and deterministic price cards remain the source of exact facts.')
    return '\n'.join(parts)


def render_story(fields,description,valuation=None):
    import streamlit as st
    # Define lazily: unit tests can import orchestration without loading Streamlit/torch.
    @st.cache_resource(show_spinner=False)
    def get_engine(model_id,revision):
        return SummaryEngine()
    story=build_valuation_story(fields,description,valuation)
    summary_source=qualitative_report_source(fields,description,valuation) if valuation else story
    fingerprint=hashlib.sha256(str(('cross-panel-valuation-v3',MODEL_ID,REVISION,PROMPT,PROMPT_SUFFIX,CHUNK_TOKENS,OUTPUT_TOKENS,story,summary_source)).encode()).hexdigest()
    st.markdown('#### 📖 Your car & price story · BART summary' if valuation else '#### 📖 The car’s story · AI summary')
    if valuation:st.caption('The advert and seller’s description, both valuations, and a short explanation of the evaluation method. Exact figures stay in the price cards and full source story.')
    saved=st.session_state.get('_bart_summary',{})
    if saved.get('fingerprint')!=fingerprint:
        try:
            # Keep pricing posteriors and the BART weights out of RAM together.
            from pricing.hierarchy_sparse_live import _load
            _load.clear()
            import gc
            gc.collect()
            with st.spinner('Condensing the car’s story…'):
                result=summarize_story(summary_source,get_engine(MODEL_ID,REVISION))
            saved={'fingerprint':fingerprint,'result':result}
            st.session_state['_bart_summary']=saved
        except Exception as error:
            import logging
            logging.getLogger(__name__).warning('BART summary unavailable: %s',type(error).__name__)
            saved={'fingerprint':fingerprint,'result':{'status':'unavailable','summary':''}}
            st.session_state['_bart_summary']=saved
        finally:
            get_engine.clear()
            import gc
            gc.collect()
    result=saved['result']
    if result['status']=='ready':st.write(result['summary'])
    elif result['status']=='review_required':st.caption('The draft introduced an unsupported number and was withheld. Read the supplied story below.')
    else:st.caption('The summary is unavailable. Your price estimate is still ready; the supplied story is below.')
    if valuation:st.caption('AI overview, not a valuation calculation. BART receives a qualitative briefing covering the report’s topics; exact figures remain in the cards and source report. A number check withholds drafts that introduce figures absent from the supplied story. It cannot detect every factual mistake or a number attached to the wrong fact.')
    st.caption('ainize/bart-base-cnn · AI-generated summary of unverified listing information. It can omit defects, title issues or misstate facts; check the original. This summary does not affect the price model.')
    with st.expander('Full supplied story · model & limitations'):
        st.markdown(story.replace('\n','\n\n'))
        if valuation:
            with st.expander('Qualitative briefing supplied to BART'):
                st.text(summary_source)
            st.download_button('Download full valuation story',story,'car-valuation-story.txt','text/plain',key='download_valuation_story')
        st.caption('BART-base-CNN runs locally on CPU, using greedy decoding. Pricing model caches are released before generation; BART weights are released afterward. Long stories are summarized in token-sized chunks and merged. Results are cached in your session. No LexRank, Qwen, Kaggle job or external AI API. A numeric check catches some invented numbers; it does not establish factual accuracy.')
