# Car-story summaries in the pricing GUI

The advert-clustering releases were reverted in commit
`717c6a69ba9e5d4e100c486f5c3d350547a9c621`, restoring the exact pre-clustering tree.
This replacement is **GUI-only**. No Kaggle job, full-dataset processing, model
fine-tuning or price-model refit is performed.

After **Estimate asking price**, the receipt displays **The car, without the sales
pitch**. A shared `summarize_vehicle(fields, description, engine)` function first
serializes the supplied vehicle into a narrative:

> The make is Ford. The model and trim is Focus. The model year is 2014. The mileage
> in miles is 95000. Seller description (unverified): ...

All 13 available structured model fields can enter: make, model, year, mileage,
condition, cylinders, fuel, title, transmission, drivetrain, body, color and state.
Unknown values are omitted. An allowlist prevents observed prices, predicted
prices, residuals and analytics fields from entering the prompt. Common price,
phone, email, URL and VIN patterns are redacted from the description; this is not
a guarantee of complete personal-data removal. The pricing calculator's existing
8,000-character description limit remains in effect.

## Model and inference

Use the official **Qwen2.5-0.5B-Instruct**, quantized **Q4_K_M GGUF**, with
`llama-cpp-python` CPU inference on the application server. The model is an
instruction-following causal transformer, producing an abstractive paragraph
rather than selecting keywords or cluster labels. Its instruction-following and
long-context capabilities suit a mixture of structured specifications and seller
language. This is a practical choice for the hosting budget, **not a claim that
it is the universally best automotive summarizer**; no labeled car-summary
benchmark was supplied.

Pinned repositories/revisions:

- Base: `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`.
- GGUF: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` at `9217f5db79a29953eb74d5343926648285ec7e67`.
- File: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (approximately 491 MB).
- Runtime: `llama-cpp-python==0.3.36`, official CPU wheel index.
- Serving context: 2,048 tokens; generation: greedy, at most 192 new tokens,
  temperature 0, seed 42. The prompt requests at most 70 words / three sentences.

The quantized weights download once from the official public Hugging Face repo
and are cached on the server. Visitor descriptions are processed locally; they
are not sent to an external AI inference provider. Cold startup may take longer
and requires outbound access for the model download. Subsequent requests reuse
one locked model instance. Summary results are cached only in the visitor's
Streamlit session and invalidate when supplied facts or description change.

Long stories are split by actual model tokens, each part is summarized, and
multiple results are merged with the supplied facts. The tail is not silently
truncated. A numeric consistency check withholds drafts containing digits absent
from the supplied story. A separate check withholds explicit transmission or
drivetrain claims that conflict with the supplied category. Residual generated
sales boilerplate is removed and at most three generated sentences are shown.
Incomplete outputs are also withheld. These checks do
**not** establish factual correctness, detect every contradiction, validate
seller claims, or catch hallucinations written as number words. Users can inspect
the full supplied story in the explanation panel.

## Relationship to the price

The summary is part of the estimate **presentation**, not a new pricing feature.
The existing MiniLM + CatBoost asking-price pipeline continues to receive the
original description and vehicle fields. The auction model and pricing weights
are unchanged. The GUI says that the prose is a summary of unverified listing
information, not an inspection or causal explanation of the dollar estimate.
Feeding summaries into a price model would require a separate train/validation/test
experiment and consistent preprocessing; that is outside this GUI replacement.

## Validation and attribution

Unit tests cover narrative serialization, exclusion of price targets, shared
single/batch orchestration, long-input tail handling, numeric-hallucination
rejection and session caching after Estimate. Real CPU smoke tests use a few
existing public description examples; these establish serving behavior, not
semantic accuracy or corpus-wide performance. No fake summary is substituted if
the model cannot run: the price stays visible with a brief unavailable message.

[Official model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
(Apache 2.0). [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
retains its own license. Project software remains MIT © Amit Steinberg. Existing
example descriptions retain their original dataset attribution and license.

```bash
python -m pip install -r requirements-summary.txt
python -m pytest tests/test_vehicle_summary.py tests/test_summary_ui.py -q
python -m streamlit run app.py
```

## Fast GUI release

Replaces the 1.5B model with the official 0.5B Q4_K_M model to reduce CPU work and cold download size. This favors responsiveness over language-model capacity; factual safeguards remain enabled. Actual latency depends on hosting load and description length. Pricing models are unchanged.
