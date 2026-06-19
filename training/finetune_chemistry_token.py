"""Vocab-extension fine-tune of the RNA-only chinchilla encoder to add
[RNA] / [DNA] chemistry tokens.

Inputs:
  --encoder-ckpt: existing 14M chinchilla encoder ckpt (vocab=6)
  --config: training config with mixed RNA + DNA sources (chemistry-tagged)
  --steps: short fine-tune (default 4000)

Process:
  1. Load existing encoder ckpt (vocab=6 token embedding)
  2. Build new encoder with vocab=8 (adds 2 rows for [RNA], [DNA] tokens)
  3. Copy old embedding rows 0–5 into new ckpt; rows 6, 7 random-init
  4. Continue training with low LR (5e-5) on chemistry-tagged data
  5. Save fine-tuned ckpt

Backward-compat note: existing decoder ckpt's vocab unchanged — decoder uses its
own embedding table and never sees chemistry tokens. Only the encoder side
expands.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aptamer_dataset import (ContrastiveAptamerDataset, N_TOKENS, PAD_ID,
                              RNA_TOK_ID, DNA_TOK_ID, encode)
from aptamer_encoder import AptamerEncoder, info_nce_loss


def extend_embedding(old_weight: torch.Tensor, new_vocab: int) -> torch.Tensor:
    """Copy [old_vocab, d] embedding into [new_vocab, d], random-init extra rows.
    New rows initialized with small Gaussian (std=0.02, like BERT init)."""
    old_vocab, d = old_weight.shape
    if new_vocab == old_vocab:
        return old_weight
    if new_vocab < old_vocab:
        raise ValueError(f'new_vocab ({new_vocab}) < old_vocab ({old_vocab})')
    new_weight = torch.empty(new_vocab, d, dtype=old_weight.dtype, device=old_weight.device)
    torch.nn.init.normal_(new_weight, mean=0.0, std=0.02)
    new_weight[:old_vocab] = old_weight
    return new_weight


def load_and_extend_encoder(args, cfg, device):
    ck = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    enc_cfg = ck.get('config', {})
    model_cfg = dict(enc_cfg.get('model', {}))
    model_cfg.pop('grad_checkpoint', None)
    model_cfg.setdefault('max_len', cfg['data']['max_len'])
    model_cfg.setdefault('regression_head', enc_cfg.get('model', {}).get('regression_head', False))

    # Build model with NEW vocab (N_TOKENS = 8)
    model = AptamerEncoder(**model_cfg).to(device)
    print(f'  built encoder: vocab={N_TOKENS}, d_model={model_cfg["d_model"]}, '
          f'layers={model_cfg["num_layers"]}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')

    # Load ckpt state dict; extend embedding if old vocab smaller
    state = ck['encoder']
    if 'embed.weight' in state and state['embed.weight'].shape[0] < N_TOKENS:
        old_vocab = state['embed.weight'].shape[0]
        print(f'  extending embed.weight from vocab {old_vocab} → {N_TOKENS} (random-init new rows for chemistry tokens)')
        state['embed.weight'] = extend_embedding(state['embed.weight'], N_TOKENS)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing: print(f'  missing keys: {missing}')
    if unexpected: print(f'  unexpected keys: {unexpected}')
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--encoder-ckpt', required=True, type=Path)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=5e-5,
                    help='low LR for fine-tune (default 5e-5)')
    ap.add_argument('--run-id', default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    cfg['training']['steps'] = args.steps
    cfg['training']['lr'] = args.lr
    if args.run_id:
        cfg['run_id'] = args.run_id
    run_id = cfg['run_id']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}  run_id: {run_id}')

    # Encoder (vocab-extended from ckpt)
    model = load_and_extend_encoder(args, cfg, device)

    # Dataset (sources tagged with chemistry='RNA' or 'DNA')
    ds = ContrastiveAptamerDataset(
        sources=cfg['data']['sources'],
        max_len=cfg['data']['max_len'],
        test_exclusion_parquet=cfg['data'].get('test_exclusion_parquet'),
        shuffle_buffer=cfg['data'].get('shuffle_buffer', 50_000),
        seed=cfg.get('seed', 0),
        target_embeddings_parquet=cfg['data'].get('target_embeddings_parquet'),
    )
    target_emb_table = (torch.from_numpy(ds._target_emb_table).to(device)
                        if ds._target_emb_table is not None else None)

    # Sliding window curriculum (re-use existing config)
    phases = cfg.get('curriculum', {}).get('phases') if cfg.get('curriculum', {}).get('enabled') else None
    if phases:
        # For fine-tune, apply the FINAL phase (no curriculum re-training)
        p = phases[-1]
        if 'pos_frac' in p and 'neg_frac' in p:
            ds.set_active_round_frac_window(*p['pos_frac'], *p['neg_frac'])
            print(f'  curriculum (final phase): pos {p["pos_frac"]}, neg {p["neg_frac"]}')

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'],
                             betas=tuple(cfg['training']['betas']),
                             weight_decay=cfg['training']['weight_decay'])
    total_steps = cfg['training']['steps']
    warmup = min(cfg['training'].get('warmup_steps', 200), total_steps // 4)
    def lr_lambda(s):
        if s < warmup: return s / max(1, warmup)
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = cfg['training'].get('amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    run_dir = Path.home() / 'Desktop/autoRNA_data/tumbleweed/training_runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    train_log = open(run_dir / 'train.jsonl', 'a')

    print(f'\n=== Fine-tune {total_steps} steps, LR={args.lr}, vocab {N_TOKENS} ===')
    t0 = time.time()
    losses = []
    it = iter(ds)
    B = cfg['training']['batch_size']

    for step in range(1, total_steps + 1):
        batch_data = [next(it) for _ in range(B)]
        batch = ds.collate(batch_data)
        anchor_ids = batch['anchor_ids'].to(device, non_blocking=True)
        positive_ids = batch['positive_ids'].to(device, non_blocking=True)
        negative_ids = batch['negative_ids'].to(device, non_blocking=True)
        anc_ti = batch['anchor_target_idx'].to(device, non_blocking=True)
        pos_ti = batch['positive_target_idx'].to(device, non_blocking=True)
        neg_ti = batch['negative_target_idx'].to(device, non_blocking=True)
        anc_tgt = target_emb_table[anc_ti] if target_emb_table is not None else None
        pos_tgt = target_emb_table[pos_ti] if target_emb_table is not None else None
        neg_tgt = target_emb_table[neg_ti] if target_emb_table is not None else None

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            anc = model(anchor_ids, target_emb=anc_tgt)
            pos = model(positive_ids, target_emb=pos_tgt)
            neg = model(negative_ids, target_emb=neg_tgt)
            loss = info_nce_loss(anc, pos, neg, temperature=cfg.get('loss', {}).get('temperature', 0.2))

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(float(loss.item()))

        if step % cfg['log']['every_steps'] == 0:
            n = cfg['log']['every_steps']
            avg = sum(losses[-n:]) / n
            entry = dict(step=step, loss=avg, lr=sched.get_last_lr()[0], ts=time.time())
            print(f'  step {step:>5}  loss={avg:.4f}  lr={entry["lr"]:.2e}  ({(time.time()-t0):.0f}s)')
            train_log.write(json.dumps(entry) + '\n'); train_log.flush()

    # Save
    ck_path = run_dir / f'encoder_chemtok_step{total_steps}.pt'
    torch.save({
        'step': total_steps,
        'encoder': model.state_dict(),
        'config': cfg,
        'base_encoder_ckpt': str(args.encoder_ckpt),
        'vocab_size': N_TOKENS,
        'chemistry_tokens': {'RNA': RNA_TOK_ID, 'DNA': DNA_TOK_ID},
    }, ck_path)
    print(f'\nsaved {ck_path}')
    train_log.close()


if __name__ == '__main__':
    main()
