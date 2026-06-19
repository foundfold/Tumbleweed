"""Production-ready contrastive training for SELEX-aware aptamer foundation models.

Features:
  - YAML config (training/configs/default.yaml)
  - Streaming dataset (1.1B-read corpus, never loaded in full)
  - DDP for multi-GPU (auto-detected; single-process when only 1 GPU available)
  - Mixed precision (torch.amp)
  - Gradient checkpointing on the transformer stack (saved memory)
  - Eval every N steps: R@1/R@10 on InstructNA LOX1+CXCL5 panel, Kd-spearman
  - Checkpoint resume + rolling keep-last-N
  - JSON-lines training log + stdout

Usage:
  python3 train_contrastive.py                            # MPS / single-GPU
  python3 train_contrastive.py --config configs/default.yaml
  torchrun --nproc-per-node=4 train_contrastive.py        # DDP on 4 GPUs
  python3 train_contrastive.py --resume runs/<run_id>/ckpt_step100000.pt

Run dir layout:
  ~/Desktop/autoRNA_data/tumbleweed/training_runs/<run_id>/
    config.yaml      copy of the launch config
    train.jsonl      one JSON object per logged step (step, loss, lr, ts)
    eval.jsonl       one JSON object per eval (step, R@1, R@10, kd_spearman)
    ckpt_step{N}.pt  checkpoint with encoder state + optimizer state + RNG state
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, IterableDataset

# CUDA 13.0 cuBLAS instability workaround: disable TF32 globally; force bf16 autocast.
# Safe to leave on for all configs — bf16 has same exponent range as fp32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aptamer_dataset import ContrastiveAptamerDataset, encode, PAD_ID, MASK_ID, NUC_IDS, N_TOKENS
from aptamer_encoder import AptamerEncoder, info_nce_loss
from eval_benchmarks import run_all as run_all_benchmarks
from kmer_features import build_kmer_features, kmer_feature_dim


# v18: token-id → base-letter map. Aligned with aptamer_dataset.py:
#   0='A', 1='C', 2='G', 3='U', 4=PAD, 5=MASK, 6=[RNA], 7=[DNA], 8='T'
ID2BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 8: 'T'}


def _ids_to_seqs(ids: torch.Tensor) -> list[str]:
    """Decode (B, L) input_ids tensor → list of A/C/G/T strings (RNA U→T)."""
    ids_np = ids.detach().cpu().numpy()
    seqs = []
    for row in ids_np:
        chars = []
        for v in row:
            b = ID2BASE.get(int(v))
            if b is not None:
                chars.append(b)
        seqs.append(''.join(chars).replace('U', 'T'))
    return seqs


def compute_kmer_batch(ids: torch.Tensor, device, dtype=torch.float32) -> torch.Tensor:
    """v18: produce kmer feature tensor for a batch of input_ids.
    Returns (B, kmer_feature_dim()) tensor on `device`.
    """
    seqs = _ids_to_seqs(ids)
    feats = build_kmer_features(seqs)               # (B, D)
    return torch.from_numpy(feats).to(device=device, dtype=dtype)


def bert_mask_torch(ids: torch.Tensor, p: float, generator: torch.Generator | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized BERT-style 80/10/10 masking on a (B, L) LongTensor.
    Returns (masked_ids, active_mask). Loss is computed only where active=True.
    """
    not_pad = ids != PAD_ID
    rand = torch.rand(ids.shape, device=ids.device, generator=generator)
    active = not_pad & (rand < p)
    sub_rand = torch.rand(ids.shape, device=ids.device, generator=generator)
    is_mask = active & (sub_rand < 0.8)
    is_rand = active & (sub_rand >= 0.8) & (sub_rand < 0.9)
    masked_ids = ids.clone()
    masked_ids[is_mask] = MASK_ID
    # Random replacement: pick from NUC_IDS uniformly
    nuc_pool = torch.tensor(NUC_IDS, device=ids.device, dtype=ids.dtype)
    n_rand = int(is_rand.sum())
    if n_rand > 0:
        rand_picks = nuc_pool[torch.randint(0, len(NUC_IDS), (n_rand,), device=ids.device, generator=generator)]
        masked_ids[is_rand] = rand_picks
    return masked_ids, active


# ============================================================ Distributed init ===
def setup_ddp() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank). rank 0 if no DDP."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get('LOCAL_RANK', 0))
    return 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ================================================================ Dataset wrap ===
class _IterableWrap(IterableDataset):
    """Wrap our custom dataset (which is its own iterator)."""
    def __init__(self, ds, max_steps: int):
        self.ds = ds
        self.max_steps = max_steps

    def __iter__(self):
        it = iter(self.ds)
        for _ in range(self.max_steps):
            yield next(it)


# ====================================================================== Eval ===
@torch.no_grad()
def eval_retrieval(model, device, panel_parquet, max_len: int,
                   n_distractors: int = 1000) -> dict:
    """R@1/R@10 + Kd-Spearman on the InstructNA LOX1+CXCL5 panel.

    Setup:
      - 40 panel sequences (20 LOX1 + 20 CXCL5)
      - For each, query against the panel itself + N_DIST random R0 sequences
      - Score = cosine similarity in our learned space
      - R@1: fraction of panel anchors whose top-1 retrieved is another panel item
      - kd_spearman: Spearman ρ between mean-similarity-to-strong-binders and (–Kd)
        Lower Kd should rank higher.
    """
    import pandas as pd
    from scipy.stats import spearmanr

    panel = pd.read_parquet(panel_parquet)
    home = Path.home()
    r0 = pd.read_parquet(home / 'Desktop/autoRNA_data/tumbleweed/synthetic_r0/InstructNA_LOX1/synthetic_r0.parquet').head(n_distractors)

    def embed(seqs):
        ids = torch.tensor([encode(s, max_len) for s in seqs], device=device)
        model.eval()
        return model(ids)

    panel_emb = embed(panel['sequence'].tolist())
    r0_emb = embed(r0['sequence'].tolist())

    # R@k retrieval — anchor = panel item, gallery = panel ∪ R0
    all_emb = torch.cat([panel_emb, r0_emb], dim=0)
    sims = panel_emb @ all_emb.t()
    # zero diagonal (don't retrieve self)
    diag = torch.arange(len(panel_emb), device=device)
    sims[diag, diag] = -1e9
    topk = sims.argsort(dim=1, descending=True)[:, :10]
    is_panel = topk < len(panel_emb)
    r_at_1 = is_panel[:, 0].float().mean().item()
    r_at_10 = is_panel.any(dim=1).float().mean().item()

    # Kd-Spearman: rank panel items by mean-sim-to-other-strong-binders vs (–Kd)
    strong = panel['kd_class'] == 'strong'
    strong_emb = panel_emb[torch.tensor(strong.values, device=device)]
    panel_to_strong = (panel_emb @ strong_emb.t()).mean(dim=1).cpu().numpy()
    kd_known = panel[panel['kd_nm'].notna()]
    if len(kd_known) >= 5:
        idxs = kd_known.index.to_numpy()
        rho, _ = spearmanr(panel_to_strong[idxs], -kd_known['kd_nm'].to_numpy())
    else:
        rho = float('nan')

    return dict(r_at_1=r_at_1, r_at_10=r_at_10, kd_spearman=float(rho))


# ============================================================ Train loop ===
def train(cfg: dict, resume: Path | None, max_seconds: float | None = None):
    rank, world, local_rank = setup_ddp()
    main = is_main(rank)
    device = (torch.device(f'cuda:{local_rank}') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    if main:
        print(f'rank {rank}/{world}  device {device}')

    seed = cfg.get('seed', 0) + rank
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    if main:
        print('\n=== Dataset ===')
    ds = ContrastiveAptamerDataset(
        sources=cfg['data']['sources'],
        max_len=cfg['data']['max_len'],
        test_exclusion_parquet=cfg['data'].get('test_exclusion_parquet'),
        shuffle_buffer=cfg['data'].get('shuffle_buffer', 50_000),
        seed=seed,
        target_embeddings_parquet=cfg['data'].get('target_embeddings_parquet'),
    )

    # Move target embedding tables to GPU once at startup. Per-batch lookups by
    # index avoid shipping any embedding arrays through worker IPC.
    target_emb_table_gpu = None
    target_mask_table_gpu = None
    if ds._target_emb_table is not None:
        target_emb_table_gpu = torch.from_numpy(ds._target_emb_table).to(device)
        if main:
            print(f'  target table on {device}: shape={tuple(target_emb_table_gpu.shape)} '
                  f'({target_emb_table_gpu.element_size() * target_emb_table_gpu.nelement() / 1e6:.0f} MB)')
        if ds._target_mask_table is not None:
            target_mask_table_gpu = torch.from_numpy(ds._target_mask_table).to(device)

    # Curriculum: list of phases. Two supported formats per phase:
    #   1. Tag mode  : {start_step, neg_tags: [...]}            (legacy)
    #   2. Window mode: {start_step, pos_frac: [lo, hi],
    #                                neg_frac: [lo, hi]}        (sliding window)
    phases = cfg.get('curriculum', {}).get('phases') if cfg.get('curriculum', {}).get('enabled') else None
    def phase_for(step):
        idx = 0
        for i, p in enumerate(phases):
            if step >= p['start_step']: idx = i
        return idx

    def apply_phase(p):
        if 'pos_frac' in p and 'neg_frac' in p:
            pos_lo, pos_hi = p['pos_frac']
            neg_lo, neg_hi = p['neg_frac']
            ds.set_active_round_frac_window(pos_lo, pos_hi, neg_lo, neg_hi)
        else:
            ds.set_active_neg_tags(p.get('neg_tags'))

    def build_loader():
        iterable = _IterableWrap(ds, max_steps=cfg['training']['steps'] * cfg['training']['batch_size'])
        return DataLoader(
            iterable,
            batch_size=cfg['training']['batch_size'],
            num_workers=cfg['data'].get('num_workers', 0),
            collate_fn=ds.collate,
            pin_memory=device.type == 'cuda',
        )

    if phases:
        cur_phase = phase_for(0)
        if main: print(f'\n=== Curriculum: {len(phases)} phases, starting phase {cur_phase} ===')
        apply_phase(phases[cur_phase])
    else:
        cur_phase = -1
    loader = build_loader()

    if main:
        print('\n=== Model ===')
    enc = AptamerEncoder(**cfg['model']).to(device)
    if main:
        print(f'  {sum(p.numel() for p in enc.parameters())/1e6:.2f}M params')

    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        enc = DDP(enc, device_ids=[local_rank] if device.type == 'cuda' else None,
                  find_unused_parameters=False)

    opt = torch.optim.AdamW(
        enc.parameters(),
        lr=cfg['training']['lr'],
        betas=tuple(cfg['training']['betas']),
        weight_decay=cfg['training']['weight_decay'],
    )

    total_steps = cfg['training']['steps']
    warmup = cfg['training']['warmup_steps']
    def lr_lambda(s):
        if s < warmup: return s / max(1, warmup)
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    use_amp = cfg['training']['amp'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    run_dir = (Path.home() / 'Desktop/autoRNA_data/tumbleweed/training_runs'
               / cfg['run_id'])
    if main:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f, sort_keys=False)
        train_log = open(run_dir / 'train.jsonl', 'a')
        eval_log = open(run_dir / 'eval.jsonl', 'a')

    start_step = 0
    if resume and resume.exists():
        ck = torch.load(resume, map_location=device)
        (enc.module if world > 1 else enc).load_state_dict(ck['encoder'])
        opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        start_step = ck['step'] + 1
        if main:
            print(f'\nresumed from step {start_step} of {resume.name}')

    if main:
        print(f'\n=== Training ({total_steps} steps from {start_step}) ===')
    enc.train()
    t0 = time.time()
    loss_window: list[float] = []
    step = start_step
    if phases:
        cur_phase = phase_for(start_step)
        apply_phase(phases[cur_phase])
        loader = build_loader()
    data_iter = iter(loader)
    reg_enabled = bool(cfg['model'].get('regression_head', False))
    reg_weight = float(cfg['training'].get('reg_loss_weight', 0.5))
    mlm_enabled = bool(cfg['model'].get('mlm_head', False))
    mlm_weight = float(cfg['training'].get('mlm_weight', 0.0))
    mlm_mask_ratio = float(cfg['training'].get('mlm_mask_ratio', 0.15))

    # ─── Kd regression head (v12+) ──────────────────────────────────────────
    # Loads the small leave-target-out Kd parquet, pre-encodes sequences and
    # pre-resolves target indices on startup so each step's Kd batch sample is
    # just an index lookup + a single forward through the encoder's kd_head.
    kd_enabled = bool(cfg['model'].get('kd_head', False))
    kd_weight = float(cfg['training'].get('kd_weight', 0.0))
    kd_batch_size = int(cfg['training'].get('kd_batch_size', 32))
    kd_data = None
    if kd_enabled and kd_weight > 0:
        kd_parquet = cfg['data'].get('kd_train_parquet')
        if kd_parquet is None:
            raise ValueError('kd_head=True but data.kd_train_parquet not set in config')
        kd_path = Path(os.path.expanduser(kd_parquet))
        kd_df = pd.read_parquet(kd_path)
        if main:
            print(f'\n=== Kd supervision: {len(kd_df)} labeled rows from {kd_path.name} ===')
            print(f'  chemistry split: {dict(kd_df["chemistry_norm"].value_counts())}')
            print(f'  unique targets: {kd_df["target_canonical"].nunique()}')

        # Use the dataset's existing target_protein → idx mapping
        name_to_idx = ds._target_protein_to_idx
        kd_df = kd_df[kd_df['target_canonical'].map(lambda t: t in name_to_idx)].copy()
        if main:
            print(f'  rows w/ ESM-2 target idx resolved: {len(kd_df)}')
        kd_df['target_idx'] = kd_df['target_canonical'].map(name_to_idx)

        # Pre-encode each sequence using its chemistry (DNA→[DNA], RNA→[RNA])
        max_len = int(cfg['model'].get('max_len', 128))
        ids_list = []
        for _, r in kd_df.iterrows():
            ids_list.append(encode(r['sequence'], max_len, chemistry=r['chemistry_norm']))
        ids_arr = np.stack(ids_list).astype(np.int64)
        kd_ids = torch.from_numpy(ids_arr).to(device)
        kd_tgt_idx = torch.from_numpy(kd_df['target_idx'].values.astype(np.int64)).to(device)
        kd_labels = torch.from_numpy(kd_df['log_kd_norm'].values.astype(np.float32)).to(device)
        kd_data = (kd_ids, kd_tgt_idx, kd_labels)
        if main:
            print(f'  pre-encoded {len(kd_df)} Kd rows; per-step sample batch = {kd_batch_size}')
    while step < total_steps:
        # Curriculum phase change: rebuild loader so workers pick up new neg pool.
        if phases:
            want_phase = phase_for(step)
            if want_phase != cur_phase:
                if main:
                    p = phases[want_phase]
                    label = (f'pos={p["pos_frac"]} neg={p["neg_frac"]}'
                             if 'pos_frac' in p else f'neg_tags={p.get("neg_tags")}')
                    print(f'  ── curriculum step {step}: phase {cur_phase} → {want_phase}  {label} ──')
                apply_phase(phases[want_phase])
                loader = build_loader()
                data_iter = iter(loader)
                cur_phase = want_phase
        batch = next(data_iter)
        anchor_ids = batch['anchor_ids'].to(device, non_blocking=True)
        positive_ids = batch['positive_ids'].to(device, non_blocking=True)
        negative_ids = batch['negative_ids'].to(device, non_blocking=True)
        # Lookup target embeddings (and masks) on-GPU by index — collate only
        # passes small int tensors. Saves ~95% of IPC bandwidth vs shipping arrays.
        unwrapped = enc.module if world > 1 else enc
        if unwrapped.target_cond_mode is None or target_emb_table_gpu is None:
            anc_te = pos_te = neg_te = None
            anc_tm = pos_tm = neg_tm = None
        else:
            anc_ti = batch['anchor_target_idx'].to(device, non_blocking=True)
            pos_ti = batch['positive_target_idx'].to(device, non_blocking=True)
            neg_ti = batch['negative_target_idx'].to(device, non_blocking=True)
            anc_te = target_emb_table_gpu[anc_ti]
            pos_te = target_emb_table_gpu[pos_ti]
            neg_te = target_emb_table_gpu[neg_ti]
            if target_mask_table_gpu is not None:
                anc_tm = target_mask_table_gpu[anc_ti]
                pos_tm = target_mask_table_gpu[pos_ti]
                neg_tm = target_mask_table_gpu[neg_ti]
            else:
                anc_tm = pos_tm = neg_tm = None

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            if reg_enabled:
                # v18: compute kmer features when encoder has kmer-token enabled
                if unwrapped.kmer_token_dim > 0:
                    anc_kf = compute_kmer_batch(anchor_ids, device)
                    pos_kf = compute_kmer_batch(positive_ids, device)
                    neg_kf = compute_kmer_batch(negative_ids, device)
                else:
                    anc_kf = pos_kf = neg_kf = None
                anc, anc_r = unwrapped.forward_with_reg(anchor_ids, target_emb=anc_te, target_mask=anc_tm, kmer_feats=anc_kf)
                pos, pos_r = unwrapped.forward_with_reg(positive_ids, target_emb=pos_te, target_mask=pos_tm, kmer_feats=pos_kf)
                neg, neg_r = unwrapped.forward_with_reg(negative_ids, target_emb=neg_te, target_mask=neg_tm, kmer_feats=neg_kf)
                nce = info_nce_loss(anc, pos, neg, temperature=cfg['training']['tau'])
                anc_t = batch['anchor_round_frac'].to(device, non_blocking=True)
                pos_t = batch['positive_round_frac'].to(device, non_blocking=True)
                neg_t = batch['negative_round_frac'].to(device, non_blocking=True)
                mse = F.mse_loss(
                    torch.cat([anc_r, pos_r, neg_r], dim=0),
                    torch.cat([anc_t, pos_t, neg_t], dim=0),
                )
                loss = nce + reg_weight * mse
            else:
                # v18: when encoder has kmer-token, compute per-sequence kmer features
                if unwrapped.kmer_token_dim > 0:
                    anc_kf = compute_kmer_batch(anchor_ids, device)
                    pos_kf = compute_kmer_batch(positive_ids, device)
                    neg_kf = compute_kmer_batch(negative_ids, device)
                else:
                    anc_kf = pos_kf = neg_kf = None
                anc = enc(anchor_ids, target_emb=anc_te, target_mask=anc_tm, kmer_feats=anc_kf)
                pos = enc(positive_ids, target_emb=pos_te, target_mask=pos_tm, kmer_feats=pos_kf)
                neg = enc(negative_ids, target_emb=neg_te, target_mask=neg_tm, kmer_feats=neg_kf)
                nce = info_nce_loss(anc, pos, neg, temperature=cfg['training']['tau'])
                mse = torch.zeros((), device=device)
                loss = nce

            # MLM auxiliary loss (v9+): force seq content into encoder hidden.
            # Sequence-only forward (target_emb=None) over masked anchor batch.
            if mlm_enabled and mlm_weight > 0.0:
                masked_ids, active = bert_mask_torch(anchor_ids, mlm_mask_ratio)
                logits = unwrapped.forward_mlm_logits(masked_ids)  # (B, L, V)
                if active.any():
                    mlm_loss = F.cross_entropy(
                        logits[active], anchor_ids[active], reduction='mean')
                else:
                    mlm_loss = torch.zeros((), device=device)
                loss = loss + mlm_weight * mlm_loss
            else:
                mlm_loss = torch.zeros((), device=device)

            # Kd regression aux loss (v12+): direct supervised signal from the
            # small Kd-labeled pool. Sample a Kd batch every step.
            if kd_enabled and kd_weight > 0.0 and kd_data is not None:
                kd_all_ids, kd_all_tgt_idx, kd_all_labels = kd_data
                n_kd = kd_all_ids.shape[0]
                # Sample with replacement to keep batch_size fixed even when n_kd < kd_batch_size
                idx = torch.randint(0, n_kd, (kd_batch_size,), device=device)
                k_ids = kd_all_ids[idx]
                k_tgt = (target_emb_table_gpu[kd_all_tgt_idx[idx]]
                         if target_emb_table_gpu is not None else None)
                k_y = kd_all_labels[idx]
                k_pred = unwrapped.forward_kd(k_ids, target_emb=k_tgt)
                kd_loss = F.mse_loss(k_pred, k_y)
                loss = loss + kd_weight * kd_loss
            else:
                kd_loss = torch.zeros((), device=device)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(enc.parameters(), cfg['training']['grad_clip'])
        scaler.step(opt)
        scaler.update()
        sched.step()

        loss_window.append(float(loss.item()))

        if main and (step + 1) % cfg['log']['every_steps'] == 0:
            recent = loss_window[-cfg['log']['every_steps']:]
            avg = sum(recent) / len(recent)
            elapsed = time.time() - t0
            rate = (step + 1 - start_step) / max(1, elapsed)
            entry = dict(step=step + 1, loss=avg, lr=sched.get_last_lr()[0],
                         step_per_s=rate, ts=time.time(),
                         nce=float(nce.item()), mse=float(mse.item()),
                         mlm=float(mlm_loss.item()), kd=float(kd_loss.item()))
            extra = f' nce={entry["nce"]:.3f} mse={entry["mse"]:.4f}' if reg_enabled else ''
            if mlm_enabled:
                extra += f' mlm={entry["mlm"]:.3f}'
            if kd_enabled:
                extra += f' kd={entry["kd"]:.4f}'
            print(f'  step {step+1:>7}  loss={avg:.4f}{extra}  '
                  f'lr={entry["lr"]:.2e}  {rate:.1f} step/s  {elapsed:.0f}s')
            train_log.write(json.dumps(entry) + '\n'); train_log.flush()

        if main and (step + 1) % cfg['eval']['every_steps'] == 0:
            model_for_eval = (enc.module if world > 1 else enc)
            model_for_eval.eval()
            metrics = run_all_benchmarks(
                encode_fn=lambda ids: model_for_eval.encode(ids),
                max_len=cfg['data']['max_len'], device=device,
            )
            metrics['step'] = step + 1
            metrics['ts'] = time.time()
            print(f'  EVAL step {step+1}: '
                  f'panel R@1={metrics.get("panel_r_at_1", 0):.3f} '
                  f'R@10={metrics.get("panel_r_at_10", 0):.3f} '
                  f'ρ={(metrics.get("panel_kd_spearman") or 0):.3f}  '
                  f'A4 r={(metrics.get("a4_pearson") or 0):.3f}  '
                  f'Li AUC={(metrics.get("li_auroc") or 0):.3f}  '
                  f'UTex r={(metrics.get("utexas_pearson") or 0):.3f}  '
                  f'test R@1={metrics.get("test_r_at_1", 0):.3f}  '
                  f'({metrics["eval_seconds"]:.0f}s)')
            eval_log.write(json.dumps(metrics) + '\n'); eval_log.flush()
            enc.train()

        if main and (step + 1) % cfg['ckpt']['every_steps'] == 0:
            ckpt_path = run_dir / f'ckpt_step{step+1}.pt'
            torch.save({
                'step': step,
                'encoder': (enc.module if world > 1 else enc).state_dict(),
                'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict(),
                'config': cfg,
            }, ckpt_path)
            print(f'  ckpt → {ckpt_path}')
            # Rolling keep-last-N
            ckpts = sorted(run_dir.glob('ckpt_step*.pt'),
                           key=lambda p: int(p.stem.split('step')[-1]))
            for old in ckpts[:-cfg['ckpt']['keep_last']]:
                old.unlink()

        step += 1
        if step >= total_steps: break
        if max_seconds is not None and (time.time() - t0) > max_seconds:
            if main: print(f'\n--max-seconds {max_seconds} elapsed at step {step}; stopping.')
            break

    if main:
        train_log.close(); eval_log.close()
        print('\nDone.')
    cleanup_ddp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(__file__).parent / 'configs/default.yaml'))
    ap.add_argument('--resume', type=Path, default=None)
    ap.add_argument('--max-seconds', type=float, default=None,
                    help='Hard cap on wall-time. Trainer breaks between batches when exceeded.')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args.resume, max_seconds=args.max_seconds)


if __name__ == '__main__':
    main()
