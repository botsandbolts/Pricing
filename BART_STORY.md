# BART-base-CNN vehicle-story summaries

The asking-price receipt summarizes the structured vehicle narrative plus seller
description with `ainize/bart-base-cnn`, pinned at revision
`b90bc9a7c93de6449a8c531ed5f957d84649b99a`. This checkpoint has 139,420,416
parameters and was fine-tuned on CNN/DailyMail news summarization. Its publisher
lists Apache 2.0. Automotive summaries require separate validation.

This replaces the Falconsai GUI backend. FLAN-T5 and SmolLM2 experiments are not
deployed. The existing MiniLM price branch and asking/bid model weights stay unchanged;
only the narrative receipt changes. No LexRank, training, Kaggle job or external
inference API is used. The full supplied story remains in an expander.

The model runs locally on CPU using the existing PyTorch/Transformers runtime.
The publisher supplies a PyTorch state dict rather than safetensors; loading uses
weights_only=True and trust_remote_code=False. No publisher Python code is run.
Two intra-op threads, FP32, single-beam greedy generation, 128-new-token output
budget, no forced EOS, and three-gram repetition blocking match the CPU experiment.
The minimum-new-token count is zero to avoid forcing news-length summaries on short ads.
The thread cap is process-wide. Model initialization is lazy and globally cached;
generation is locked. Session results are cached by model/revision/source/settings.

The roughly 558MB FP32 weights download on first use. Download size is not RAM:
our standalone test peaked at about 1,472MiB process RSS including Python, PyTorch,
weights and inference. A shared Streamlit host can have different latency/memory.
Root requirements changes request a fresh process so the previous summary model
is not intentionally retained alongside BART.

Narratives use allowlisted vehicle specifications and the original seller wording,
with common contacts/URLs/VIN patterns redacted. No price target or predicted text
price is added as a structured field. This is not complete PII protection.
All input tokens are processed in chunks of at most 960 tokens, within the 1,024-token
encoder limit after special tokens. Multiple summaries are merged through the same
model. Input tails are not silently truncated, though summarization can omit content.

Drafts with a digit-based number absent from the source are withheld. Empty,
unfinished or failed generations show an unavailable caption while leaving the price
and full story available. The check does not prove factual consistency, preserve
negations, detect all wrong model names, or guarantee that critical defects remain.

## Measured limitations

The 12-story / 24-generation CPU test averaged a median 1.45-second generation time
(range 0.92–4.62s), with 22/24 EOS-finished outputs and identical repeated pairs.
These are local test results, not production SLAs or accuracy estimates.

Manual inspection found omissions of salvage status, battery problems and windshield
damage, corrupted mileage digits, repeated car names, and irrelevant dealer text.
The GUI explicitly warns that defects/title issues can be omitted and points to the
original story. It never presents the output as an inspection or verified vehicle facts.
See `reports/bart-base-cnn-cpu/REPORT.md` and `results.json` for full examples,
parameters, attribution and comparison with SmolLM2. Keyword checks are diagnostic,
not a semantic accuracy score. No held-out gold-summary benchmark is available.

Model card: https://huggingface.co/ainize/bart-base-cnn

Tests: `python -m pytest tests/test_bart_summary.py tests/test_analytics.py tests/test_ga_events.py -q`
