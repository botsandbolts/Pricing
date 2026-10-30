"""Local Pegasus CNN/DailyMail abstractive summaries; no LexRank or external inference."""
import hashlib
import re
import threading
from pricing.vehicle_narrative import narrative

MODEL_ID='google/pegasus-cnn_dailymail'
REVISION='40d588fdab0cc077b80d950b300bf66ad3c75b92'
CHUNK_TOKENS=960
OUTPUT_TOKENS=128
PROMPT=''
PROMPT_SUFFIX=''


class SummaryEngine:
    def __init__(self):
        # Avoid killing the whole Streamlit process on memory-limited hosts.
        from pathlib import Path
        limits=[('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]
        for limit_path,usage_path in limits:
            try:
                limit=Path(limit_path).read_text().strip()
                if limit!='max' and int(limit)-int(Path(usage_path).read_text()) < 3*1024**3:
                    raise RuntimeError('Pegasus requires at least 3 GiB available RAM for safe CPU loading')
                break
            except (OSError,ValueError):continue
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        torch.set_num_threads(min(2,torch.get_num_threads()))
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,use_fast=False)
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
    """Readable cross-panel briefing; exact numeric tables stay in the source report."""
    facts,seller=narrative({**fields,'year':None,'odometer':None},description)
    facts+=' The exact model year and mileage are in the vehicle profile.'
    parts=[
        'The car valuation report separates the seller advertised asking price from the wholesale auction sale estimate. Their gap is not dealer profit or a guaranteed discount.',
        'Historical price models are adjusted to the selected date using vehicle ageing and a public monthly used-car market index. Past same-car prices are reconstructed scenarios, not observed sales. Mileage is held constant for the report; the interactive timeline can instead assume annual driving.',
        'The index can require a short-horizon forecast beyond its latest observation. Sparse trims use pooled depreciation when flagged. Newer auction generations require an explicit analogue assumption. Current individual-car prices remain experimental without validated prediction intervals.',
        'The supplied vehicle profile follows. '+facts,
    ]
    for section in valuation.get('sections',[]):
        title=section['title']
        if title=='Description branch':parts.append('The description branch uses MiniLM to supply a text-price feature to the asking model. The auction dataset has no descriptions.')
        elif title=='Feature contributions':parts.append('Native feature-contribution charts explain the historical model response. Separate waterfalls explain ageing and market adjustments in dollars. Phrase analysis is available on request. Explanations describe model associations, not causal effects.')
        elif title.endswith('model & evaluation'):parts.append(title+': historical tests report errors on the whole original test population and on the retained cohort. Filtering by actual outcomes makes the retained scores optimistic. An independent new cohort is required to establish increased accuracy.')
        elif title=='Market clock backtests':parts.append('The market forecast has rolling-origin backtests. Component tests do not establish the accuracy of the whole current-car valuation.')
        elif title.endswith('depreciation evidence'):parts.append(title+': separately fitted depreciation components have historical validation evidence, with their own limitations.')
        elif title=='Other studio tools':parts.append(section['text'])
        elif title=='Price targets & datasets':parts.append('The sources are historical Craigslist advertised prices and reported wholesale auction selling prices, covering different years and market segments.')
    if seller:parts.append('The seller states, without independent verification: '+seller)
    return '\n'.join(parts)


def render_story(fields,description,valuation=None):
    import streamlit as st
    # Define lazily: unit tests can import orchestration without loading Streamlit/torch.
    @st.cache_resource(show_spinner=False)
    def get_engine(model_id,revision):
        return SummaryEngine()
    story=build_valuation_story(fields,description,valuation)
    summary_source=qualitative_report_source(fields,description,valuation) if valuation else story
    fingerprint=hashlib.sha256(str(('cross-panel-pegasus-v1',MODEL_ID,REVISION,PROMPT,PROMPT_SUFFIX,CHUNK_TOKENS,OUTPUT_TOKENS,story,summary_source)).encode()).hexdigest()
    st.markdown('#### 📖 Your car & price story · Pegasus summary' if valuation else '#### 📖 The car’s story · AI summary')
    if valuation:st.caption('Your car, both markets, historical scenarios, model drivers and evaluation evidence — one briefing. Pegasus is the narrator; the receipts keep the numbers honest.')
    saved=st.session_state.get('_bart_summary',{})
    if saved.get('fingerprint')!=fingerprint:
        try:
            with st.spinner('Condensing the car’s story…'):
                result=summarize_story(summary_source,get_engine(MODEL_ID,REVISION),qualitative=bool(valuation))
            saved={'fingerprint':fingerprint,'result':result}
            st.session_state['_bart_summary']=saved
        except Exception as error:
            import logging
            logging.getLogger(__name__).warning('Pegasus summary unavailable: %s',type(error).__name__)
            saved={'fingerprint':fingerprint,'result':{'status':'unavailable','summary':''}}
    result=saved['result']
    if result['status']=='ready':st.write(result['summary'])
    elif result['status']=='review_required':st.caption('The draft introduced an unsupported number and was withheld. Read the supplied story below.')
    else:st.caption('Pegasus is unavailable on this host (model loading, memory or generation limit). Your price estimate is still ready; read the complete supplied story below.')
    if valuation:st.caption('AI overview, not a valuation calculation. Pegasus receives a qualitative briefing covering the report’s topics; exact figures remain in the cards and source report. The overview disables numeric output tokens; it can still omit or misstate facts.')
    st.caption('google/pegasus-cnn_dailymail · AI-generated summary of unverified listing information. It can omit defects, title issues or misstate facts; check the original. This summary does not affect the price model.')
    with st.expander('Full supplied story · model & limitations'):
        st.markdown(story.replace('\n','\n\n'))
        if valuation:
            with st.expander('Qualitative briefing supplied to Pegasus'):
                st.text(summary_source)
            st.download_button('Download full valuation story',story,'car-valuation-story.txt','text/plain',key='download_valuation_story')
        st.caption('Pegasus CNN/DailyMail runs locally on CPU, using greedy decoding and a shared cached model. Long stories are summarized in token-sized chunks and merged. Results are cached in your session. No LexRank, Qwen, Kaggle job or external AI API. A numeric check catches some invented numbers; it does not establish factual accuracy.')
