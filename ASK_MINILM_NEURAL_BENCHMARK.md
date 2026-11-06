# MiniLM two-out-of-bag asking-price neural benchmark

Same 18 arms as the completed original experiment: 3 architectures x 3 losses
x full/Lemon training. Original train/validation/test, category features,
24 train-fitted TF-IDF/SVD coordinates and model hyperparameters are unchanged.

One extra numerical input: the pretrained all-MiniLM-L6-v2 384-dimensional
description embeddings reduced to a **cross-fitted price signal**. We prefer
the frozen original GitHub cache (originally generated using Kaggle source
data); if unavailable we re-encode the already-verified Parquet with the
pinned original ONNX encoder, without any transformer training.

To obtain exactly TWO out-of-bag predictions from THREE heads, each Ridge head
is trained on only one of three disjoint neural FIT-only groups. Every FIT row
is absent from two heads, whose predictions are averaged. The original
training-only internal early stopper and untouched chronological
validation/test never enter any head fitting; they get the mean of all 3 heads.

The original archived Kaggle three-fold models were trained on 2/3 of the
training data each, so averaging the other two old predictions would NOT be
out-of-fold. The small Ridge price heads are re-fitted safely; the encoder is
reused unchanged.

Workflow: .github/workflows/ask-minilm-neural-sweep.yml
Results: reports/ask-minilm-neural-benchmark/comparison.json
Outputs appear only after all 18 training jobs and row-level audits succeed.
Production ask and bid models are not changed.
