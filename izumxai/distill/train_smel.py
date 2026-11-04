"""Continue dense IzumxAI with SMEL FT's cached-prefix and frozen-head CE.

This is a dense extension, without the MoE router/expert selection. Uses the
actual pinned smel.loss implementation and native Transformers Qwen3 layers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch

from .common import build_labeled_example, configure_vram_budget, load_tokenizer, pick_dtype, print_vram
from .smel_dense import (SMEL_COMMIT, SMEL_PATH, cache_fingerprint, capture_prefix,
                         cached_response_loss, drop_prefix)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_parity(values):
    keys=('full_loss','cached_loss','relative_lora_gradient_error')
    if not all(math.isfinite(values[key]) for key in keys):
        raise RuntimeError('nonfinite cached-prefix equivalence result')
    if values['relative_lora_gradient_error'] > 0.03 or abs(values['full_loss']-values['cached_loss']) > 0.02:
        raise RuntimeError('cached-prefix equivalence check failed')


def read_examples(path, tokenizer, max_len):
    out = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ex = build_labeled_example(tokenizer, row['messages'], max_len)
        if ex is None:
            raise ValueError(f'empty/overlength example in {path}; no silent truncation')
        out.append({**ex, 'kind': row.get('task_type', 'unknown')})
    if not out:
        raise ValueError(f'empty dataset: {path}')
    return out


def feature_cache(model, examples, signature, cut, directory, name):
    digest = cache_fingerprint(signature, examples, cut)
    path = Path(directory) / f'{name}_{digest}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['fingerprint'] != digest or len(saved['examples']) != len(examples):
            raise ValueError('cache identity/length mismatch')
        for cached, original in zip(saved['examples'], examples):
            if cached['input_ids'] != original['input_ids'] or cached['labels'] != original['labels']:
                raise ValueError('cache tokens/labels mismatch')
            if tuple(cached['hidden'].shape) != (len(original['input_ids']), model.config.hidden_size):
                raise ValueError('cache hidden shape mismatch')
        print(f'cache hit {name}: {len(examples)} examples', flush=True)
        return saved['examples'], str(path)
    device = next(model.parameters()).device
    cached = []
    start = time.monotonic()
    for index, ex in enumerate(examples, 1):
        ids = torch.tensor([ex['input_ids']], device=device)
        with torch.autocast(device.type, dtype=pick_dtype(), enabled=device.type == 'cuda'):
            hidden = capture_prefix(model, ids, cut)
        cached.append({**ex, 'hidden': hidden[0].cpu()})
        if index % 32 == 0 or index == len(examples):
            print(f'cache {name} {index}/{len(examples)} | {time.monotonic()-start:.1f}s', flush=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'fingerprint': digest, 'signature': signature, 'examples': cached}, temporary)
    temporary.replace(path)
    return cached, str(path)


def tensors(example, device):
    hidden = example['hidden'].unsqueeze(0).to(device).detach()
    labels = torch.tensor([example['labels']], device=device)
    return hidden, labels, torch.ones(labels.shape, device=device, dtype=torch.long)


@torch.no_grad()
def evaluate(model, examples, dtype):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    sums = {}
    try:
        for ex in examples:
            h, labels, mask = tensors(ex, device)
            count = int((labels[:, 1:] != -100).sum())
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda'):
                value = cached_response_loss(model, h, labels, mask).item()
            group = sums.setdefault(ex['kind'], {'loss_sum': 0.0, 'tokens': 0})
            group['loss_sum'] += value * count
            group['tokens'] += count
    finally:
        model.train(was_training)
    by_kind = {kind: item['loss_sum'] / item['tokens'] for kind, item in sums.items()}
    return {'by_kind': by_kind, 'macro_loss': sum(by_kind.values()) / len(by_kind)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='models/izumxai-2b')
    ap.add_argument('--data', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--out', default='runs/izumxai-smel-dense')
    ap.add_argument('--cache', default='data/cache/smel-dense')
    ap.add_argument('--cut', type=int, default=20)
    ap.add_argument('--max-len', type=int, default=512)
    ap.add_argument('--steps', type=int, default=32)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--code-probability', type=float, default=0.7)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--eval-every', type=int, default=8)
    ap.add_argument('--vram-budget-gb', type=float, default=2.8)
    ap.add_argument('--seed', type=int, default=20261103)
    args = ap.parse_args()
    if min(args.steps, args.grad_accum, args.rank, args.eval_every, args.max_len) < 1:
        ap.error('steps, accumulation, rank, intervals and length must be positive')
    if not 0 < args.code_probability < 1 or args.lr <= 0:
        ap.error('code probability must be in (0,1), LR positive')
    if pick_dtype() != torch.bfloat16:
        ap.error('CUDA BF16 support required')
    configure_vram_budget(args.vram_budget_gb)
    source = Path(args.model).resolve()
    if not source.is_dir():
        ap.error('a local dense Qwen3 checkpoint is required')
    out = Path(args.out)
    if (out/'final').exists():
        ap.error('output already contains a completed adapter')
    out.mkdir(parents=True, exist_ok=True)
    smel_commit = subprocess.check_output(['git', '-C', str(SMEL_PATH), 'rev-parse', 'HEAD'], text=True).strip()
    if smel_commit != SMEL_COMMIT:
        ap.error(f'SMEL FT revision mismatch: expected {SMEL_COMMIT}')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import transformers
    import bitsandbytes

    tok = load_tokenizer(str(source))
    train = read_examples(args.data, tok, args.max_len)
    val = read_examples(args.val, tok, args.max_len)
    def prompts(rows):
        return {tuple(ex['input_ids'][:next(i for i,v in enumerate(ex['labels']) if v != -100)]) for ex in rows}
    if prompts(train) & prompts(val):
        ap.error('train/validation prompt overlap')
    signature = {
        'weights': {p.name: file_hash(p) for p in sorted(source.glob('*.safetensors'))},
        'config': file_hash(source/'config.json'), 'chat_template': tok.chat_template,
        'tokenizer': {p.name:file_hash(p) for p in source.glob('*tokenizer*.json')},
        'dtype':'bfloat16', 'quantization':'NF4 double quant BF16',
        'norms':'native Qwen3 FP32 weights', 'backend':'sdpa',
        'torch':torch.__version__, 'transformers':transformers.__version__,
        'bitsandbytes':bitsandbytes.__version__,
        'device_capability':torch.cuda.get_device_capability(),
    }
    print('loading dense IzumxAI NF4; frozen BF16 vocabulary, FP32 norms', flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(source), dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={'':0}, attn_implementation='sdpa')
    if model.config.model_type != 'qwen3' or not 0 <= args.cut < model.config.num_hidden_layers:
        ap.error('only native dense Qwen3 with a nonempty suffix is supported')
    # k-bit preparation's blanket FP32 cast of the vocabulary caused the old
    # transient peak. Freeze directly and upcast only native normalization weights.
    model.requires_grad_(False)
    for module in model.modules():
        if type(module).__name__ == 'Qwen3RMSNorm':
            module.to(dtype=torch.float32)
    model.config.use_cache = False
    start = time.monotonic()
    train, train_cache = feature_cache(model, train, signature, args.cut, args.cache, 'train')
    val, val_cache = feature_cache(model, val, signature, args.cut, args.cache, 'val')
    cache_seconds = time.monotonic() - start
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank*2, lora_dropout=0.0, bias='none',
        task_type='CAUSAL_LM', target_modules=['gate_proj','up_proj','down_proj'],
        layers_to_transform=list(range(args.cut,model.config.num_hidden_layers))))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    # One real-weight gradient comparison before releasing the frozen prefix.
    device = next(model.parameters()).device
    probe = max(train, key=lambda ex:len(ex['input_ids']))
    ids = torch.tensor([probe['input_ids']],device=device)
    labels = torch.tensor([probe['labels']],device=device)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        embeds = model.get_base_model().get_input_embeddings()(ids)
    torch.cuda.synchronize(); before=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        full_loss = cached_response_loss(model,embeds,labels,mask)
    full_loss.backward()
    torch.cuda.synchronize(); full_seconds=time.monotonic()-before
    gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    drop_prefix(model.get_base_model(), args.cut)
    torch.cuda.empty_cache()
    h, labels, mask = tensors(probe,device)
    torch.cuda.synchronize(); before=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        cached_loss = cached_response_loss(model,h.requires_grad_(),labels,mask)
    cached_loss.backward()
    torch.cuda.synchronize(); cached_seconds=time.monotonic()-before
    error=sum((p.grad.float().cpu()-gradients[n].float()).square().sum().item()
              for n,p in model.named_parameters() if p.requires_grad)
    denom=sum(g.float().square().sum().item() for g in gradients.values())
    relative_error=math.sqrt(error/max(denom,1e-30))
    parity={'full_loss':full_loss.item(),'cached_loss':cached_loss.item(),
            'relative_lora_gradient_error':relative_error,
            'uncached_probe_seconds':full_seconds,'cached_probe_seconds':cached_seconds,
            'timing_scope':'one cold full pass and one cached pass, not a throughput benchmark'}
    validate_parity(parity)
    (out/'parity.json').write_text(json.dumps(parity,indent=2),encoding='utf-8')
    print('real-weight prefix parity:',parity,flush=True)
    model.zero_grad(set_to_none=True)
    del gradients, ids, embeds, h, full_loss, cached_loss
    trainable=[p for p in model.parameters() if p.requires_grad]
    model.print_trainable_parameters()
    from bitsandbytes.optim import PagedAdamW8bit
    optimizer=PagedAdamW8bit(trainable,lr=args.lr)
    code=[x for x in train if x['kind']=='code']
    replay=[x for x in train if x['kind']!='code']
    if not code or not replay:
        ap.error('both code and replay examples are required')
    rng=random.Random(args.seed)
    history=[]; best=float('inf'); best_step=0
    def validate(step):
        nonlocal best,best_step
        values={'step':step,**evaluate(model,val,torch.bfloat16)}
        history.append(values)
        if values['macro_loss'] < best:
            best=values['macro_loss']; best_step=step
            model.save_pretrained(out/'best'); tok.save_pretrained(out/'best')
        (out/'validation.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print('validation:',values,flush=True)
        torch.cuda.empty_cache()
    validate(0)
    started=time.monotonic(); tokens=0
    for step in range(1,args.steps+1):
        window=[rng.choice(code if rng.random()<args.code_probability else replay)
                for _ in range(args.grad_accum)]
        target_count=sum(sum(t!=-100 for t in ex['labels'][1:]) for ex in window)
        summed_loss=0.0
        for ex in window:
            h,labels,mask=tensors(ex,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=cached_response_loss(model,h.requires_grad_(),labels,mask,
                                          denominator=target_count)
            loss.backward(); summed_loss+=loss.item(); tokens+=len(ex['input_ids'])
        torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True)
        progress=(step-1)/max(1,args.steps-1)
        lr=args.lr*min(1,step/2)*(0.1+0.9*0.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if step%4==0:
            print(f'step {step}/{args.steps} | token NLL {summed_loss:.4f} | {tokens/(time.monotonic()-started):.0f} tok/s',flush=True)
        if step%args.eval_every==0 or step==args.steps:
            validate(step)
    model.save_pretrained(out/'final'); tok.save_pretrained(out/'final')
    meta={'base_model':str(source),'method':'SMEL FT dense extension: cached frozen prefix + upstream chunked CE',
          'moe_router_selection':False,'smel_commit':smel_commit,
          'smel_loss_sha256':file_hash(SMEL_PATH/'smel/loss.py'),
          'cut':args.cut,'lora_rank':args.rank,'targets':['gate_proj','up_proj','down_proj'],
          'steps':args.steps,'lr':args.lr,'seed':args.seed,'grad_accum':args.grad_accum,
          'code_probability':args.code_probability,'tokens_seen':tokens,
          'train_examples':len(train),'val_examples':len(val),
          'cache_seconds':cache_seconds,'training_seconds':time.monotonic()-started,
          'train_cache':train_cache,'val_cache':val_cache,'cache_signature':signature,
          'best_step':best_step,'validation':history,
          'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
          'allocator_budget_gib':args.vram_budget_gb}
    (out/'train_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('done; best step',best_step,flush=True); print_vram('SMEL dense complete')


if __name__=='__main__':
    main()
