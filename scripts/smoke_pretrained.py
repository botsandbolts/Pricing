"""Optional actual-checkpoint gradient smoke test. Not a pricing evaluation."""
from pathlib import Path
import json
import gc
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel
from pricing.text import MODEL_ID,fit_text,predict_text,save_text,load_text


def main():
    torch.set_num_threads(8)
    revision='ff629add2ff80256e42173ec84299eed1c1b6393'
    source=MODEL_ID
    tokenizer=AutoTokenizer.from_pretrained(source,revision=revision)
    initial=AutoModel.from_pretrained(source,revision=revision)
    assert initial.config.hidden_size==768
    before=initial.encoder.layer[0].attention.self.query.weight.detach().clone()
    del initial;gc.collect()
    texts=['Reliable sedan with maintenance records, clean interior, good tires and no known mechanical problems.',
           'Older hatchback with high mileage, worn paint and a transmission fault. Needs substantial repair work.']
    model=fit_text(texts,[12000.,2500.],tokenizer=tokenizer,source=source,revision=revision,
                   epochs=1,batch_size=1,accumulation=2,device='cpu',seed=42)
    after=model.backbone.encoder.layer[0].attention.self.query.weight.detach()
    delta=float((after-before).abs().max())
    assert delta>0
    predictions=predict_text(model,tokenizer,texts,'cpu')
    assert np.isfinite(predictions).all()
    path=Path('training-cache/pretrained-smoke')
    save_text(model,tokenizer,path,{'source':MODEL_ID,'revision':revision,'scope':'2-example gradient smoke only'})
    del model;gc.collect()
    restored,tok=load_text(path)
    np.testing.assert_allclose(predict_text(restored,tok,texts,'cpu'),predictions,rtol=1e-5)
    report={'status':'passed','scope':'two synthetic examples, one optimizer step; not a pricing evaluation',
            'source':MODEL_ID,'revision':revision,'dimensions':768,
            'first_attention_query_max_weight_delta':delta,'save_load_prediction_parity':True}
    (path/'smoke-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
