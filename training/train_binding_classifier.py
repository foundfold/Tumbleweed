"""Train a target-conditional binding classifier on top of a FROZEN encoder.

Supports LOO holdout: pass --holdout-targets MECP2 to exclude all MECP2 positive
samples from training, so the classifier learns "is this seq a binder for T"
purely from the OTHER 21 training targets — then we can evaluate zero-shot on T.

Three classes of training samples per batch:
  - POSITIVES: (anchor_seq, anchor's true target_protein), label=1
    Drawn from late-round SELEX, excluding holdout targets.
  - XTGT NEGS: (anchor_seq, randomly chosen DIFFERENT training target), label=0
    Forces classifier to use target info, not just sequence patterns.
  - R0 NEGS: (R0 seq, random training target), label=0
    Naive-library sequences shouldn't bind anything.

Default mix: 50% positives, 25% xtgt negs, 25% R0 negs.

Usage:
  python3 training/train_binding_classifier.py \\
    --encoder-ckpt <path> \\
    --holdout-targets MECP2 \\
    --config configs/binding_clf_loo.yaml \\
    --steps 4000
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

# CUDA 13.0 cuBLAS instability — disable TF32, use bf16 autocast.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aptamer_dataset import ContrastiveAptamerDataset, encode, PAD_ID
from aptamer_encoder import AptamerEncoder
from aptamer_binding_classifier import BindingClassifier, bce_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--encoder-ckpt', required=True, type=Path)
    ap.add_argument('--holdout-targets', default='',
                    help='Comma-separated targets to exclude from positives (LOO)')
    ap.add_argument('--steps', type=int, default=None,
                    help='Override config steps')
    ap.add_argument('--run-id', default=None, help='Override run_id (default: from config + holdout suffix)')
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    holdouts = {t.strip() for t in args.holdout_targets.split(',') if t.strip()}
    if args.steps is not None:
        cfg['training']['steps'] = args.steps

    run_id = args.run_id or (cfg['run_id'] + ('_LOO_' + '_'.join(sorted(holdouts)) if holdouts else ''))

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print(f'device: {device}')
    print(f'run_id: {run_id}')
    if holdouts:
        print(f'HOLDOUT targets (excluded from positives): {sorted(holdouts)}')

    # 1. Load + freeze encoder
    print(f'loading encoder: {args.encoder_ckpt}')
    ck = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    enc_cfg = ck.get('config', {})
    model_cfg = dict(enc_cfg.get('model', {}))
    model_cfg.pop('grad_checkpoint', None)
    model_cfg.setdefault('max_len', cfg['data']['max_len'])
    encoder = AptamerEncoder(**model_cfg).to(device).eval()
    encoder.load_state_dict(ck['encoder'])
    for p in encoder.parameters():
        p.requires_grad = False
    d_in = model_cfg['d_model']
    print(f'  encoder {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params (frozen, d_model={d_in})')

    # 2. Classifier head
    clf = BindingClassifier(d_in=d_in, d_hidden=cfg['model']['d_hidden'],
                             dropout=cfg['model']['dropout']).to(device)
    print(f'  classifier {sum(p.numel() for p in clf.parameters())/1e6:.2f}M params (trainable)')

    # 3. Dataset
    ds = ContrastiveAptamerDataset(
        sources=cfg['data']['sources'],
        max_len=cfg['data']['max_len'],
        test_exclusion_parquet=cfg['data'].get('test_exclusion_parquet'),
        shuffle_buffer=cfg['data'].get('shuffle_buffer', 50_000),
        seed=cfg.get('seed', 0),
        target_embeddings_parquet=cfg['data'].get('target_embeddings_parquet'),
    )
    target_emb_table = torch.from_numpy(ds._target_emb_table).to(device) if ds._target_emb_table is not None else None

    # Build set of training-target indices (excluding holdouts) — for sampling random targets
    name_to_idx = ds._target_protein_to_idx
    allowed_target_names = [n for n in name_to_idx.keys() if n not in holdouts]
    allowed_target_idxs = np.array([name_to_idx[n] for n in allowed_target_names])
    holdout_idxs = {name_to_idx[t] for t in holdouts if t in name_to_idx}
    print(f'  {len(allowed_target_idxs)} training-target IDs available; '
          f'{len(holdout_idxs)} held out')

    # 4. Optimizer
    opt = torch.optim.AdamW(clf.parameters(), lr=cfg['training']['lr'],
                            betas=tuple(cfg['training']['betas']),
                            weight_decay=cfg['training']['weight_decay'])
    total_steps = cfg['training']['steps']
    warmup = cfg['training']['warmup_steps']
    def lr_lambda(s):
        if s < warmup: return s / max(1, warmup)
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = cfg['training'].get('amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 5. Run dir
    run_dir = Path.home() / 'Desktop/autoRNA_data/tumbleweed/training_runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    train_log = open(run_dir / 'train.jsonl', 'a')
    rng = np.random.default_rng(cfg.get('seed', 0))

    # 6. Training loop — sample anchor+positive from contrastive dataset,
    #    skip if anchor's target is in holdouts, build target-shuffle + R0 negatives
    print(f'\n=== Training {total_steps} steps ===')
    t0 = time.time()
    losses, accs = [], []
    step = 0
    it = iter(ds)
    B = cfg['training']['batch_size']
    while step < total_steps:
        anchor_seqs, anchor_targets = [], []
        # Sample B positives (anchor seqs with non-holdout targets)
        while len(anchor_seqs) < B:
            sample = next(it)
            tgt_idx = int(sample['anchor_target_idx'])
            # Skip null target (idx=0) and holdouts; need a real training target
            if tgt_idx == 0 or tgt_idx in holdout_idxs:
                continue
            anchor_seqs.append(sample['anchor_ids'])
            anchor_targets.append(tgt_idx)
        # Sample B negatives (R0 seqs) — use ds.sample_negatives
        neg_ids_np, _, _ = ds.sample_negatives(B)
        # Target-shuffle: half the anchors get assigned a DIFFERENT random allowed target.
        # The other half stay matched (positives).
        n_half = B // 2
        n_xtgt = (B - n_half) // 2
        n_r0 = B - n_half - n_xtgt
        # POS: first n_half anchors with their TRUE targets, label=1
        pos_seqs = anchor_seqs[:n_half]
        pos_targets = anchor_targets[:n_half]
        # XTGT NEG: next n_xtgt anchors with WRONG target (sample random allowed, must differ)
        xtgt_seqs = anchor_seqs[n_half:n_half + n_xtgt]
        xtgt_targets = []
        for true_t in anchor_targets[n_half:n_half + n_xtgt]:
            choices = allowed_target_idxs[allowed_target_idxs != true_t]
            xtgt_targets.append(int(rng.choice(choices)))
        # R0 NEG: R0 seqs with random allowed targets
        r0_seqs = [neg_ids_np[i] for i in range(n_r0)]
        r0_targets = [int(rng.choice(allowed_target_idxs)) for _ in range(n_r0)]
        # Combine
        all_seqs = pos_seqs + xtgt_seqs + r0_seqs
        all_targets = pos_targets + xtgt_targets + r0_targets
        all_labels = ([1] * n_half) + ([0] * n_xtgt) + ([0] * n_r0)

        ids = torch.from_numpy(np.stack(all_seqs)).to(device)
        tgt_idx_t = torch.tensor(all_targets, dtype=torch.long, device=device)
        labels = torch.tensor(all_labels, dtype=torch.float32, device=device)

        with torch.no_grad():
            tgt_emb = target_emb_table[tgt_idx_t] if target_emb_table is not None else None
            pooled = encoder.encode(ids, target_emb=tgt_emb)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = clf(pooled)
            loss = bce_loss(logits, labels)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(clf.parameters(), cfg['training']['grad_clip'])
        scaler.step(opt)
        scaler.update()
        sched.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == labels).float().mean().item()
        losses.append(float(loss.item())); accs.append(acc)
        step += 1

        if step % cfg['log']['every_steps'] == 0:
            avg_loss = sum(losses[-cfg['log']['every_steps']:]) / cfg['log']['every_steps']
            avg_acc = sum(accs[-cfg['log']['every_steps']:]) / cfg['log']['every_steps']
            entry = dict(step=step, loss=avg_loss, acc=avg_acc, lr=sched.get_last_lr()[0],
                         ts=time.time())
            print(f'  step {step:>5}  loss={avg_loss:.4f}  acc={avg_acc:.3f}  '
                  f'lr={entry["lr"]:.2e}  {(time.time()-t0):.0f}s')
            train_log.write(json.dumps(entry) + '\n'); train_log.flush()

    # 7. Save final ckpt
    ck_path = run_dir / f'classifier_step{total_steps}.pt'
    torch.save({
        'step': total_steps,
        'classifier': clf.state_dict(),
        'config': cfg,
        'encoder_ckpt': str(args.encoder_ckpt),
        'holdout_targets': sorted(holdouts),
    }, ck_path)
    print(f'\nckpt → {ck_path}')
    train_log.close()


if __name__ == '__main__':
    main()
