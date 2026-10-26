"""Fine-tune the actual E5 transformer, not a frozen embedding regressor."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from scripts.common import clean

MODEL_ID = 'embaas/sentence-transformers-multilingual-e5-base'
MAX_TOKENS = 512
MAX_CHARACTERS = 8000


def windows(tokenizer, text):
    # Prefix and special tokens count towards the native 512-token limit.
    ids = tokenizer.encode(clean(text)[:MAX_CHARACTERS], add_special_tokens=False)
    prefix = tokenizer.encode('query: ', add_special_tokens=False)
    budget = MAX_TOKENS - len(prefix) - tokenizer.num_special_tokens_to_add(False)
    assert tokenizer.num_special_tokens_to_add(False) == 2
    return [[tokenizer.bos_token_id] + prefix + ids[i:i+budget] + [tokenizer.eos_token_id]
            for i in range(0, max(1, len(ids)), budget)]


class TextPriceRegressor(nn.Module):
    def __init__(self, backbone, mean=0., std=1.):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, 1)
        self.register_buffer('target_mean', torch.tensor(float(mean)))
        self.register_buffer('target_std', torch.tensor(float(max(std, 1e-6))))

    def forward(self, input_ids, attention_mask, document_ids, n_documents):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        chunks = nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)
        pooled = chunks.new_zeros((n_documents, chunks.shape[-1])).index_add(0, document_ids, chunks)
        counts = torch.bincount(document_ids, minlength=n_documents).unsqueeze(-1).clamp_min(1)
        pooled = nn.functional.normalize(pooled / counts, dim=-1)
        return self.head(pooled).squeeze(-1)

    def log_price(self, *args):
        return self(*args) * self.target_std + self.target_mean


def batch(tokenizer, texts, device):
    ids, doc_ids = [], []
    for i, text in enumerate(texts):
        w = windows(tokenizer, text)
        ids.extend(w)
        doc_ids.extend([i] * len(w))
    padded = tokenizer.pad({'input_ids': ids}, padding=True, return_tensors='pt')
    return (padded['input_ids'].to(device), padded['attention_mask'].to(device),
            torch.tensor(doc_ids, device=device), len(texts))


def fit_text(texts, prices, *, tokenizer, source, revision=None, epochs=2, batch_size=1,
             accumulation=8, lr=2e-5, device='cuda', seed=42, log=print):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    y = np.log1p(np.asarray(prices, dtype=float))
    model = TextPriceRegressor(AutoModel.from_pretrained(source, revision=revision), y.mean(), y.std()).to(device)
    # All transformer weights remain trainable. Checkpointing controls activation memory.
    model.backbone.gradient_checkpointing_enable()
    model.backbone.config.use_cache = False
    model.backbone.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr},
        {'params': model.head.parameters(), 'lr': 1e-3}], weight_decay=.01)
    rng = np.random.default_rng(seed)
    texts = list(texts)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(texts))
        batches = [order[i:i+batch_size] for i in range(0, len(order), batch_size)]
        total = 0.
        for group_start in range(0, len(batches), accumulation):
            group = batches[group_start:group_start+accumulation]
            group_size = sum(len(idx) for idx in group)
            optimizer.zero_grad(set_to_none=True)
            for idx in group:
                inputs = batch(tokenizer, [texts[i] for i in idx], device)
                target = torch.tensor((y[idx]-y.mean()) / max(y.std(), 1e-6), dtype=torch.float32, device=device)
                loss = nn.functional.mse_loss(model(*inputs), target)
                (loss * len(idx)/group_size).backward()
                total += loss.item()*len(idx)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            log(f'epoch={epoch+1} seen={min((group_start+len(group))*batch_size,len(texts))}/{len(texts)}', flush=True)
        log(f'epoch={epoch+1} standardized_log_mse={total/len(texts):.6f}', flush=True)
    model.eval()
    return model


@torch.inference_mode()
def predict_text(model, tokenizer, texts, device, batch_size=1):
    model.eval()
    output = []
    for i in range(0, len(texts), batch_size):
        log_price = model.log_price(*batch(tokenizer, list(texts[i:i+batch_size]), device))
        output.extend(log_price.float().cpu().numpy())
    return np.expm1(np.clip(np.asarray(output), 0, np.log1p(1e7)))


def save_text(model, tokenizer, path, provenance):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path/'backbone', safe_serialization=True)
    tokenizer.save_pretrained(path/'tokenizer')
    torch.save({'head': model.head.state_dict(), 'mean': model.target_mean.cpu(),
                'std': model.target_std.cpu()}, path/'regression.pt')
    (path/'provenance.json').write_text(json.dumps(provenance, indent=2))


def load_text(path, device='cpu'):
    path = Path(path)
    state = torch.load(path/'regression.pt', map_location='cpu', weights_only=True)
    model = TextPriceRegressor(AutoModel.from_pretrained(path/'backbone'), state['mean'], state['std'])
    model.head.load_state_dict(state['head'])
    return model.to(device).eval(), AutoTokenizer.from_pretrained(path/'tokenizer')

