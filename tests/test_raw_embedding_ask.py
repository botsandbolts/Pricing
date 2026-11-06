"""No-pricing-head encoder contract + DeepCross+AFT projection checks."""
import numpy as np
import pytest
import torch

from scripts.raw_embedding_ask_encode import SPECS, id_hash, consistency
from scripts.raw_embedding_ask_train import EmbeddingOnlyDeepCross, PROJECTION_DIM
from scripts.neural_ask_train import objective


def test_prespecified_encoder_grid():
    assert len(SPECS)==8
    for name,(source,length,dim) in SPECS.items():
        assert source and length in (128,256) and dim in (384,768)
    assert "BAAI/bge-small-en-v1.5" in [i[0] for i in SPECS.values()]
    assert "thenlper/gte-small" in [i[0] for i in SPECS.values()]
    assert "sentence-transformers/all-distilroberta-v1" in [i[0] for i in SPECS.values()]
    assert "sentence-transformers/all-MiniLM-L12-v2" in [i[0] for i in SPECS.values()]


def test_full_neural_model_predicts_price_from_embeddings():
    model=EmbeddingOnlyDeepCross([5]*11,7,384)
    numerical=torch.zeros((4,7))
    categorical=torch.ones((4,11),dtype=torch.long)
    embeddings=torch.randn((4,384))
    result=model(numerical,categorical,embeddings)
    assert result.shape==(4,2)
    loss=objective(result,torch.ones(4)*9.,"aft")
    assert torch.isfinite(loss)
    loss.backward()
    assert model.projector[1].weight.grad is not None
    assert torch.isfinite(model.projector[1].weight.grad).all()
    assert model.projector[1].out_features==PROJECTION_DIM
    assert not hasattr(model,"ridge_head")
    assert not hasattr(model,"text_price_head")


def test_large_distilroberta_embedding_geometry():
    model=EmbeddingOnlyDeepCross([3]*11,7,768)
    out=model(torch.zeros((2,7)),torch.ones((2,11),dtype=torch.long),
              torch.zeros((2,768)))
    assert torch.isfinite(out).all()
    assert out.shape==(2,2)


class StableEncoder:
    def encode(self,texts,**kwargs):
        from hashlib import sha256
        numbers=np.array([[float(x)/255 for x in sha256(t.encode()).digest()[:8]]
                          for t in texts],dtype=np.float32)
        return numbers/np.linalg.norm(numbers,axis=1,keepdims=True)


class BadBatchEncoder:
    def encode(self,texts,**kwargs):
        values=StableEncoder().encode(texts)
        if len(texts)>1:
            values=np.roll(values,1,axis=1)
        return values


def test_batch_parity_is_measured_not_assumed():
    s=StableEncoder()
    check=consistency(s,["a car","second used car","third car"])
    assert check["minimum_cosine_similarity"]>.9999
    assert check["max_absolute_difference"]<.00001
    with pytest.raises(ValueError,match="parity"):
        consistency(BadBatchEncoder(),["a car","second car","third car"])


def test_holdout_id_fingerprint_is_order_invariant():
    assert id_hash(["b","a"])==id_hash(["a","b"])
    assert id_hash(["a","b"])!=id_hash(["a","c"])
