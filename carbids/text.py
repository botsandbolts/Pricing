"""Fine-tune the actual MiniLM transformer, not a frozen embedding regressor."""
from pathlib import Path
import json
import hashlib
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from carbids.common import clean

MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MAX_TOKENS = 256
MAX_CHARACTERS = 8000


def windows(tokenizer, text):
    # Use the checkpoint's standard 256-token sentence-transformer window.
    # Build special tokens through the tokenizer: MiniLM uses CLS/SEP, not BOS/EOS.
    ids = tokenizer.encode(clean(text)[:MAX_CHARACTERS], add_special_tokens=False)
    budget = MAX_TOKENS - tokenizer.num_special_tokens_to_add(False)
    start = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    end = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    if start is None or end is None or tokenizer.num_special_tokens_to_add(False) != 2:
        raise ValueError("A CLS/SEP or BOS/EOS tokenizer with two special tokens is required")
    return [[start] + ids[i:i+budget] + [end]
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
             accumulation=8, lr=2e-5, device='cuda', seed=42, log=print,
             checkpoint_dir=None, checkpoint_steps=100):
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
    texts = list(texts)
    checkpoint = None
    resume_epoch, resume_group, updates = 0, 0, 0
    if checkpoint_dir is not None:
        identity = hashlib.sha256(Path(__file__).read_bytes())
        identity.update(json.dumps(dict(source=str(source),revision=revision,epochs=epochs,
            batch_size=batch_size,accumulation=accumulation,lr=lr,seed=seed),sort_keys=True).encode())
        identity.update(y.tobytes())
        for text in texts:
            encoded = str(text).encode();identity.update(len(encoded).to_bytes(8,'little'));identity.update(encoded)
        checkpoint = Path(checkpoint_dir)/f'{identity.hexdigest()}.pt'
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        if checkpoint.exists():
            state = torch.load(checkpoint,map_location=device,weights_only=True)
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            resume_epoch,resume_group,updates = state['epoch'],state['group'],state['updates']
            torch.set_rng_state(state['rng'].cpu())
            if torch.cuda.is_available() and state['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
            log(f'Resuming optimizer checkpoint: epoch={resume_epoch+1} group={resume_group} updates={updates}',flush=True)
    def save_progress(epoch, group):
        if checkpoint is None:return
        temporary=checkpoint.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            epoch=epoch,group=group,updates=updates,rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),temporary)
        temporary.replace(checkpoint)
    for epoch in range(resume_epoch,epochs):
        model.train()
        order = np.random.default_rng(seed+epoch).permutation(len(texts))
        batches = [order[i:i+batch_size] for i in range(0, len(order), batch_size)]
        total = 0.
        first_group = resume_group if epoch == resume_epoch else 0
        seen_loss = 0
        for group_start in range(first_group, len(batches), accumulation):
            group = batches[group_start:group_start+accumulation]
            group_size = sum(len(idx) for idx in group)
            optimizer.zero_grad(set_to_none=True)
            for idx in group:
                inputs = batch(tokenizer, [texts[i] for i in idx], device)
                target = torch.tensor((y[idx]-y.mean()) / max(y.std(), 1e-6), dtype=torch.float32, device=device)
                loss = nn.functional.mse_loss(model(*inputs), target)
                (loss * len(idx)/group_size).backward()
                total += loss.item()*len(idx)
                seen_loss += len(idx)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            updates += 1
            if updates % checkpoint_steps == 0:
                save_progress(epoch,group_start+len(group))
            log(f'epoch={epoch+1} seen={min((group_start+len(group))*batch_size,len(texts))}/{len(texts)}', flush=True)
        save_progress(epoch+1,0)
        log(f'epoch={epoch+1} standardized_log_mse_remaining={total/max(1,seen_loss):.6f}', flush=True)
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

