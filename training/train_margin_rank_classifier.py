"""Train a target-conditional binding classifier with BCE + margin-rank loss.

Same architecture as BindingClassifier (3-layer MLP on frozen encoder pooled).
Difference from train_binding_classifier.py:
  - Loads r_mid sources alongside r_late + r0
  - Per batch: builds (R_top, R_mid) PAIRS where both come from the SAME target
    (so the model has an explicit "near-miss" signal to rank against the top)
  - Loss = BCE(R_top=1, R_mid=0, R0=0, xtgt=0) + rank_weight * max(0, margin − (s(R_top) − s(R_mid)))

The ranking loss is added only over targets that have R_mid data. R_top examples
from R_mid-less targets still contribute the BCE-positive term, so coverage isn't lost.

Holdout (LOO): if --holdout-targets MECP2, skip MECP2 positives entirely.

Usage:
  python3 training/train_margin_rank_classifier.py \\
    --config training/configs/binding_clf_marginrank.yaml \\
    --encoder-ckpt <path> \\
    --holdout-targets MECP2 \\
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

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aptamer_dataset import encode
from aptamer_encoder import AptamerEncoder
from aptamer_binding_classifier import BindingClassifier


def load_seq_pool(parquet_path: Path, sequence_col: str = 'sequence',
                  max_len: int = 128, test_exclude: set | None = None,
                  max_rows: int = 50_000, rng=None) -> np.ndarray:
    """Stream-sample up to max_rows sequences via duckdb (avoids loading full
    parquet into RAM — critical for huge SELEX files like alphasyn 7.6M rows).
    Returns [N, max_len] uint8 token IDs."""
    import duckdb
    # First check column existence
    try:
        cols_q = duckdb.query(f"DESCRIBE SELECT * FROM '{parquet_path}'").df()
        cols_set = set(cols_q['column_name'].tolist())
    except Exception:
        cols_set = {sequence_col}
    if sequence_col not in cols_set:
        for c in ('sequence', 'aptamer_sequence_RNA'):
            if c in cols_set:
                sequence_col = c
                break
    # Sample via duckdb — bounded memory regardless of file size
    seed = int((rng or np.random.default_rng(0)).integers(0, 2**31))
    try:
        df = duckdb.query(
            f"SELECT {sequence_col} FROM '{parquet_path}' USING SAMPLE {max_rows} ROWS (reservoir, {seed})"
        ).df()
    except Exception:
        # File smaller than max_rows or sampling unsupported — just load fully
        df = duckdb.query(f"SELECT {sequence_col} FROM '{parquet_path}'").df()
    seqs = df[sequence_col].astype(str).str.upper().str.replace('T', 'U').tolist()
    del df
    if test_exclude is not None:
        seqs = [s for s in seqs if s not in test_exclude]
    if not seqs:
        return np.zeros((0, max_len), dtype=np.uint8)
    arr = np.stack([encode(s, max_len) for s in seqs])
    return arr.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--encoder-ckpt', required=True, type=Path)
    ap.add_argument('--holdout-targets', default='')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--run-id', default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    holdouts = {t.strip() for t in args.holdout_targets.split(',') if t.strip()}
    if args.steps is not None:
        cfg['training']['steps'] = args.steps
    run_id = args.run_id or (cfg['run_id'] + ('_LOO_' + '_'.join(sorted(holdouts)) if holdouts else ''))

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print(f'device: {device}  run_id: {run_id}')
    if holdouts: print(f'HOLDOUTS: {sorted(holdouts)}')

    # 1. Encoder (frozen)
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
    print(f'  encoder {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M frozen  d_model={d_in}')

    # 2. Target embedding table
    te = pd.read_parquet(cfg['data']['target_embeddings_parquet'])
    target_names = te['target_protein'].tolist()
    name_to_idx = {n: i for i, n in enumerate(target_names)}
    target_embs = np.stack([np.asarray(e, dtype=np.float32) for e in te['embedding']])
    # z-score normalize
    target_embs = (target_embs - target_embs.mean(axis=0, keepdims=True)) / (target_embs.std(axis=0, keepdims=True) + 1e-6)
    target_emb_table = torch.from_numpy(target_embs.astype(np.float32)).to(device)
    print(f'  target embeddings: {target_emb_table.shape}')

    # 3. Test-set exclusion
    test_excl = None
    if cfg['data'].get('test_exclusion_parquet'):
        tdf = pd.read_parquet(cfg['data']['test_exclusion_parquet'])
        test_excl = set(tdf['sequence'].astype(str).str.upper().str.replace('T', 'U').tolist())
        print(f'  test-set exclusion: {len(test_excl)} seqs')

    # 4. Load source pools by class+target
    r_late_by_target: dict[str, np.ndarray] = {}  # target_name → [N, max_len]
    r_mid_by_target: dict[str, np.ndarray] = {}
    r0_pool: list[np.ndarray] = []
    max_len = cfg['data']['max_len']
    for src in cfg['data']['sources']:
        p = Path(src['parquet']).expanduser()
        if not p.exists():
            print(f'  WARN missing {p}, skip')
            continue
        seqcol = src.get('sequence_col', 'sequence')
        arr = load_seq_pool(p, seqcol, max_len, test_excl,
                            max_rows=cfg['data'].get('max_rows_per_source', 50_000),
                            rng=np.random.default_rng(cfg.get('seed', 0) + hash(src['name']) % 10000))
        if len(arr) == 0:
            continue
        cls = src.get('class', 'r_late')
        tgt = src.get('target_protein')
        if cls == 'r0':
            r0_pool.append(arr)
        elif cls in ('r_late', 'r_mid') and tgt:
            if tgt in holdouts:
                print(f'  HOLDOUT skip: {src["name"]} ({tgt}, {cls})')
                continue
            bucket = r_late_by_target if cls == 'r_late' else r_mid_by_target
            if tgt in bucket:
                bucket[tgt] = np.concatenate([bucket[tgt], arr], axis=0)
            else:
                bucket[tgt] = arr
    r0_all = np.concatenate(r0_pool, axis=0) if r0_pool else np.zeros((0, max_len), dtype=np.uint8)
    print(f'  R_late targets: {sorted(r_late_by_target.keys())}')
    print(f'  R_mid targets:  {sorted(r_mid_by_target.keys())}')
    print(f'  R0 pool: {len(r0_all)} seqs')

    paired_targets = [t for t in r_late_by_target if t in r_mid_by_target]
    print(f'  paired (R_late+R_mid) targets for ranking loss: {paired_targets}')

    late_only_targets = [t for t in r_late_by_target if t not in r_mid_by_target]
    print(f'  R_late only (no ranking term): {late_only_targets}')

    allowed_target_names = list(r_late_by_target.keys())  # excludes holdouts already
    allowed_target_idxs = np.array([name_to_idx[t] for t in allowed_target_names if t in name_to_idx])

    # 5. Classifier
    clf = BindingClassifier(d_in=d_in, d_hidden=cfg['model']['d_hidden'],
                             dropout=cfg['model']['dropout']).to(device)
    print(f'  classifier {sum(p.numel() for p in clf.parameters())/1e6:.2f}M trainable')

    # 6. Optim
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

    run_dir = Path.home() / 'Desktop/autoRNA_data/tumbleweed/training_runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    train_log = open(run_dir / 'train.jsonl', 'a')
    rng = np.random.default_rng(cfg.get('seed', 0))

    # 7. Training loop
    B = cfg['training']['batch_size']
    # Per batch composition:
    #   n_pair = B // 4 PAIRS → 2*n_pair entries (R_top + R_mid)
    #   remaining: half R_top from late-only targets (label 1) + xtgt (label 0)
    #             + R0 (label 0)
    n_pair = B // 4
    n_xtgt = B // 4
    n_r0 = B // 4
    n_late_solo = B - 2 * n_pair - n_xtgt - n_r0  # extra R_top positives
    rank_margin = cfg['training'].get('rank_margin', 0.3)
    rank_weight = cfg['training'].get('rank_weight', 0.5)

    print(f'\n=== Training {total_steps} steps (B={B}: {n_pair} pairs + {n_late_solo} solo-pos + {n_xtgt} xtgt + {n_r0} r0) ===')
    print(f'  rank_margin={rank_margin}  rank_weight={rank_weight}')
    t0 = time.time()
    losses, bce_losses, rank_losses, accs = [], [], [], []

    for step in range(1, total_steps + 1):
        # --- Build batch ---
        top_seqs, top_tgts = [], []
        mid_seqs, mid_tgts = [], []
        # Sample n_pair (R_top, R_mid) pairs from paired_targets
        if paired_targets and n_pair > 0:
            tgt_pick = rng.choice(paired_targets, size=n_pair, replace=True)
            for t in tgt_pick:
                lp, mp = r_late_by_target[t], r_mid_by_target[t]
                top_seqs.append(lp[rng.integers(0, len(lp))])
                mid_seqs.append(mp[rng.integers(0, len(mp))])
                top_tgts.append(name_to_idx[t])
                mid_tgts.append(name_to_idx[t])

        # Solo R_top positives (any target that has r_late)
        solo_seqs, solo_tgts = [], []
        if n_late_solo > 0 and allowed_target_names:
            tgt_pick = rng.choice(allowed_target_names, size=n_late_solo, replace=True)
            for t in tgt_pick:
                lp = r_late_by_target[t]
                solo_seqs.append(lp[rng.integers(0, len(lp))])
                solo_tgts.append(name_to_idx[t])

        # XTGT negatives: R_top seqs assigned WRONG target
        xtgt_seqs, xtgt_tgts = [], []
        if n_xtgt > 0 and allowed_target_names and len(allowed_target_names) > 1:
            for _ in range(n_xtgt):
                true_t = rng.choice(allowed_target_names)
                lp = r_late_by_target[true_t]
                wrong_t = true_t
                while wrong_t == true_t:
                    wrong_t = rng.choice(allowed_target_names)
                xtgt_seqs.append(lp[rng.integers(0, len(lp))])
                xtgt_tgts.append(name_to_idx[wrong_t])

        # R0 negatives
        r0_seqs, r0_tgts = [], []
        if n_r0 > 0 and len(r0_all) > 0:
            idx = rng.integers(0, len(r0_all), size=n_r0)
            r0_seqs = list(r0_all[idx])
            r0_tgts = list(rng.choice(allowed_target_idxs, size=n_r0))

        all_seqs = top_seqs + mid_seqs + solo_seqs + xtgt_seqs + r0_seqs
        all_tgts = top_tgts + mid_tgts + solo_tgts + xtgt_tgts + r0_tgts
        if not all_seqs:
            continue
        ids = torch.from_numpy(np.stack(all_seqs).astype(np.int64)).to(device)
        tgt_idx_t = torch.tensor(all_tgts, dtype=torch.long, device=device)

        with torch.no_grad():
            tgt_emb = target_emb_table[tgt_idx_t]
            pooled = encoder.encode(ids, target_emb=tgt_emb)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = clf(pooled)  # [N]
            # Slice
            i = 0
            top_logits = logits[i:i + len(top_seqs)]; i += len(top_seqs)
            mid_logits = logits[i:i + len(mid_seqs)]; i += len(mid_seqs)
            solo_logits = logits[i:i + len(solo_seqs)]; i += len(solo_seqs)
            xtgt_logits = logits[i:i + len(xtgt_seqs)]; i += len(xtgt_seqs)
            r0_logits = logits[i:i + len(r0_seqs)]; i += len(r0_seqs)

            # BCE: R_top=1, R_mid=0 (weak), R0=0, xtgt=0, solo R_top=1
            pos_part = torch.cat([top_logits, solo_logits])
            neg_part = torch.cat([r0_logits, xtgt_logits, mid_logits])  # mid as label-0 here too
            bce = F.binary_cross_entropy_with_logits(
                torch.cat([pos_part, neg_part]),
                torch.cat([torch.ones_like(pos_part), torch.zeros_like(neg_part)]),
            )
            # Margin-rank: s(top) > s(mid) + margin (paired)
            if top_logits.numel() > 0 and mid_logits.numel() == top_logits.numel():
                rank = F.relu(rank_margin - (top_logits - mid_logits)).mean()
            else:
                rank = torch.tensor(0.0, device=device)
            loss = bce + rank_weight * rank

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(clf.parameters(), cfg['training']['grad_clip'])
        scaler.step(opt)
        scaler.update()
        sched.step()

        with torch.no_grad():
            labels = torch.cat([torch.ones_like(pos_part), torch.zeros_like(neg_part)])
            preds = (torch.sigmoid(torch.cat([pos_part, neg_part])) > 0.5).float()
            acc = (preds == labels).float().mean().item()
        losses.append(float(loss.item()))
        bce_losses.append(float(bce.item()))
        rank_losses.append(float(rank.item()))
        accs.append(acc)

        if step % cfg['log']['every_steps'] == 0:
            n = cfg['log']['every_steps']
            entry = dict(step=step,
                         loss=sum(losses[-n:]) / n,
                         bce=sum(bce_losses[-n:]) / n,
                         rank=sum(rank_losses[-n:]) / n,
                         acc=sum(accs[-n:]) / n,
                         lr=sched.get_last_lr()[0],
                         ts=time.time())
            print(f'  step {step:>5}  loss={entry["loss"]:.4f}  bce={entry["bce"]:.4f}  '
                  f'rank={entry["rank"]:.4f}  acc={entry["acc"]:.3f}  ({(time.time()-t0):.0f}s)')
            train_log.write(json.dumps(entry) + '\n'); train_log.flush()

    # 8. Save
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
