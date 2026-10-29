# Falconsai vehicle-story summaries

After Estimate, the asking receipt serializes the supported vehicle fields into a
story and appends the cleaned seller description. The complete story is input to
`Falconsai/text_summarization`, pinned at revision
`6e505f907968c4a9360773ff57885cdc6dca4bfd`. This is a T5-small summarization
checkpoint (60.5M parameters), licensed Apache 2.0 by its publisher.

LexRank, TF-IDF ranking, graph selection and domain sentence-ranking rules have
been removed. The new summary is generated prose. The full supplied story remains
in an expander. Numeric price targets and text-price features do not enter the
structured narrative; original seller wording may contain prices/claims. Common
contact/URL/VIN patterns are removed; this does not guarantee complete PII removal.

Serving uses the application's existing CPU PyTorch/Transformers dependencies,
plus SentencePiece. It loads the model directly with AutoModelForSeq2SeqLM rather
than the removed Transformers v5 summarization pipeline API. Safetensors only;
no remote code. Model loading is lazy and shared with st.cache_resource. Greedy
single-beam decoding produces at most 80 new tokens per call, with EOS checks.
PyTorch intra-op threads are capped at two for CPU serving (a process-wide limit).
A lock serializes model generation. Initial download is roughly 242MB for FP32
weights; disk size is not a RAM estimate. No external AI inference API, Kaggle
job, training or GPU is used.

Input chunks contain at most 440 tokenizer tokens, leaving space within the
512-token serving input budget for the summarize prefix and special tokens. Long
stories use map/reduce over all chunks; source tails are not silently truncated.
The merge can omit details, just like each individual summary. Session-local
results are cached by source, revision and generation-budget fingerprint.

A numeric consistency check withholds outputs with digit-based numbers absent
from the source. It does not catch all factual errors, changed negations, wrong
attribution, missing defects or number words. Claims remain unverified. Failed,
empty or unfinished generations produce an unavailable caption and the full story;
no LexRank or fabricated AI summary is substituted. The asking/bid price models,
weights and original-description inputs are unchanged.

Validation: tests cover story/target separation, long-input tail and merge,
unsupported numbers, empty inputs and common contact redaction. Real model CPU
smoke tests assess serving and sample behavior, not general automotive accuracy.
Do not treat the publisher's unverified model-card metrics as our own benchmark.

Model card: https://huggingface.co/Falconsai/text_summarization

Run checks:
`python -m pytest tests/test_falconsai_summary.py tests/test_analytics.py tests/test_ga_events.py -q`
