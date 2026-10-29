# Fast vehicle stories: no generative summarization model

The asking receipt serializes all available make/model/year/mileage and supported
vehicle specifications into a factual narrative, then appends the cleaned seller
description. It displays those supplied facts alongside up to three original seller
sentences selected by continuous LexRank plus automotive priority rules.

Implementation: TF-IDF cosine similarities between story/seller sentences create
an undirected weighted graph. Damped centrality (0.85) scores the graph; dangling
rows use a uniform transition. Disclosed-defect and maintenance indicators take
priority over centrality. Duplicate/near-duplicate sentences and common sales pitches
are suppressed. The selected sentences retain source wording and original order.
A full-story expander remains available, including sentences omitted from the short
view. Empty or stopword-only inputs have deterministic fallbacks. No prices or text
price predictions are added as structured narrative fields. Contact patterns are
redacted, with no guarantee of complete PII detection.

This is extractive summarization with template-rendered facts. No new claims are
generated. Selection can still omit important information, inherit misleading claims,
or select poor fragments. Priority rules are fallible, not a vehicle inspection.
The original description still feeds the existing MiniLM pricing branch unchanged;
this presentation feature does not modify pricing weights, targets or features.

There is no Qwen, model download, GPU/Kaggle job or added runtime dependency.
Uses existing NumPy/scikit-learn. Tests cover defects/negations, boilerplate,
deduplication, target exclusion, normalization, empty input, tail retention and contacts.
A local repeated-advert microbenchmark averaged about 1.65 ms over 100 calls; this
is not a production latency guarantee or a summary-quality benchmark.

Reference: Erkan & Radev, LexRank (JAIR 2004):
https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html

Validation: python -m pytest tests/test_lexrank_story.py -q
