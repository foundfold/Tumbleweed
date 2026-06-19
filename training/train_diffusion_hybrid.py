"""Train AptamerDiffusionHybrid: joint MDM denoiser + contrastive scorer.

Single-family smoke version (default = rapt_tg2 r0..r8 for TGM2).
Once this works end-to-end, scale by adding more SELEX families to --source-glob.

Round-derived noise schedule:
  t = 1 - round_k / max_round_in_family    # r0 → t=1 (highly masked), r_max → t=0 (clean)

Per training step:
  1. Sample batch from SELEX families (round-weighted)
  2. For each sequence: compute t from its round → randomly mask t fraction of positions
  3. Forward: model(noisy_ids, target_emb, t) → {logits, proj, target_proj}
  4. L_denoise = EvoFlow-weighted CE on masked positions
  5. L_contrast = InfoNCE(proj, target_proj) within batch
  6. L = L_contrast + λ_diff * L_denoise

Usage:
  python3 training/train_diffusion_hybrid.py --config training/configs/tumbleweed_50m_diffusion_v1.yaml
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aptamer_dataset import N_TOKENS, PAD_ID, MASK_ID, RNA_TOK_ID, DNA_TOK_ID, T_ID, encode
from aptamer_diffusion_hybrid import (
    AptamerDiffusionHybrid, denoise_loss, contrast_loss, round_to_t, _ids_to_seqs,
)
from kmer_features import build_kmer_features


def compute_kmer_batch(ids: torch.Tensor, device, dtype=torch.float32) -> torch.Tensor:
    """(B, L) CLEAN ids → (B, kmer_dim) kmer feature tensor on `device`."""
    feats = build_kmer_features(_ids_to_seqs(ids))
    return torch.as_tensor(feats, dtype=dtype, device=device)


# Tokens that should NEVER be masked (special tokens that carry conditioning info)
NEVER_MASK = {PAD_ID, RNA_TOK_ID, DNA_TOK_ID}


def random_mask(ids: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence mask: with probability t[i], replace each non-special token of row i
    with MASK_ID. Returns (noisy_ids, mask_positions).
    """
    B, L = ids.shape
    device = ids.device
    # Build maskable map: True where token can be masked (not pad/chem-token)
    maskable = torch.ones_like(ids, dtype=torch.bool)
    for nm in NEVER_MASK:
        maskable &= (ids != nm)
    # Per-sample t, broadcast to per-token
    t_b = t.unsqueeze(-1).expand(-1, L)                                   # (B, L)
    sampled = torch.rand(B, L, device=device) < t_b                       # (B, L) bool
    mask_pos = maskable & sampled                                         # only mask allowed positions
    noisy = ids.clone()
    noisy[mask_pos] = MASK_ID
    return noisy, mask_pos


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SelexRoundDataset(Dataset):
    """Load (sequence, round_k, R_max, target_emb_idx, chemistry) tuples from a
    list of SELEX-family parquet sets.

    Family spec:
      {
        'name': 'rapt_tg2',
        'rounds_glob': '~/Desktop/autoRNA_data/tumbleweed/raptranker/PRJDB9110_TG2/processed/DRR*.parquet',
        'target_uniprot': 'P21980',
        'chemistry': 'RNA',
        'r_max': 8,             # max round number in this family
      }
    """
    def __init__(
        self,
        families: list[dict],
        target_emb_lookup: dict,   # uniprot → (B, 1280) numpy
        max_len: int = 96,
        max_rows_per_family: int = 200_000,
        seed: int = 0,
        winners_parquet: str | None = None,   # flat (sequence, uniprot, chem) binders
        winner_round: int = 4,                 # pseudo-round → t = 1 - round/r_max
        winner_r_max: int = 5,                 # default (4,5) → t = 0.2 (mild noise)
        winner_repeat: int = 1,                # upsample winners so batches stay target-diverse
        per_residue: bool = False,             # True → target_emb_lookup holds (P,1280) per-residue ESM-2
        max_protein_residues: int = 1000,      # truncate per-residue target to bound batch memory
        trifp_lookup: dict | None = None,      # v9: uniprot → (trifp_dim,) prediction-fingerprint
    ):
        rng = np.random.default_rng(seed)
        rows = []
        for fam in families:
            from glob import glob
            paths = sorted(glob(os.path.expanduser(fam['rounds_glob'])))
            fam_rows = []
            for p in paths:
                # Try to parse round from filename (e.g. DRR201861, or _r3)
                rnd = _parse_round_from_filename(p)
                if rnd is None:
                    continue
                try:
                    df = pd.read_parquet(p, columns=['sequence'])
                except Exception:
                    continue
                seqs = df['sequence'].astype(str).tolist()
                for s in seqs:
                    fam_rows.append((s, rnd, fam['r_max'], fam['target_uniprot'], fam['chemistry']))
            # Cap PER FAMILY (not per round-file) so high-round-count families don't swamp
            # the winners / InfoNCE target diversity. Subsample across all rounds of the family.
            if len(fam_rows) > max_rows_per_family:
                keep = rng.choice(len(fam_rows), max_rows_per_family, replace=False)
                fam_rows = [fam_rows[i] for i in keep]
            print(f'  family {fam["name"]:<18} {fam["target_uniprot"]}: {len(fam_rows)} rows (cap {max_rows_per_family})')
            rows.extend(fam_rows)
        n_selex = len(rows)

        # Inject Kd-bench winners as target-conditioned binders at a fixed mild t.
        # Upsample (winner_repeat) so they aren't swamped by the high-row-count SELEX families —
        # InfoNCE needs target-diverse batches to clear the ln(N_targets) floor.
        n_winner = 0
        if winners_parquet:
            wp = pd.read_parquet(os.path.expanduser(winners_parquet))
            # Per-row round/r_max when present (augmented noised-winner corpus: each variant
            # enters diffusion at the t implied by its synthetic SELEX round — clean winner low t,
            # noised variant high t). Falls back to the scalar winner_round/winner_r_max otherwise.
            has_per_row = 'round' in wp.columns and 'r_max' in wp.columns
            if has_per_row:
                w_rounds = wp['round'].astype(int).tolist()
                w_rmax = wp['r_max'].astype(int).tolist()
            else:
                w_rounds = [winner_round] * len(wp)
                w_rmax = [winner_r_max] * len(wp)
            w_rows = []
            for s, uid, chem, rnd, rmx in zip(wp['sequence'].astype(str), wp['uniprot'].astype(str),
                                              wp['chem'].astype(str), w_rounds, w_rmax):
                w_rows.append((s, rnd, rmx, uid, chem))
            rows.extend(w_rows * winner_repeat)
            n_winner = len(w_rows) * winner_repeat
            if has_per_row:
                rr = wp['round'].astype(int)
                tmin, tmax = 1 - rr.max() / wp['r_max'].astype(int).max(), 1 - rr.min() / wp['r_max'].astype(int).max()
                print(f'  winners: {len(w_rows)} binders × repeat {winner_repeat} = {n_winner} rows '
                      f'over {wp["uniprot"].nunique()} targets (PER-ROW round {rr.min()}..{rr.max()} → t {tmin:.3f}..{tmax:.3f})')
            else:
                print(f'  winners: {len(w_rows)} binders × repeat {winner_repeat} = {n_winner} rows '
                      f'over {wp["uniprot"].nunique()} targets (round={winner_round}/{winner_r_max} → t={1-winner_round/winner_r_max:.2f})')

        self.rows = rows
        self.max_len = max_len
        self.target_emb_lookup = target_emb_lookup
        self.per_residue = per_residue
        self.max_protein_residues = max_protein_residues
        self.trifp_lookup = trifp_lookup
        self.trifp_dim = len(next(iter(trifp_lookup.values()))) if trifp_lookup else 0
        print(f'SelexRoundDataset: {len(rows)} rows  (SELEX {n_selex} + winners {n_winner}) across {len(families)} families'
              f'{" [PER-RESIDUE target conditioning]" if per_residue else ""}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        seq, rnd, r_max, uid, chem = self.rows[idx]
        ids = encode(seq, self.max_len, chemistry=chem)
        tgt = self.target_emb_lookup.get(uid)
        if self.per_residue:
            # (P, 1280) per-residue ESM-2; 1 zero-residue fallback for missing targets
            if tgt is None:
                tgt = np.zeros((1, 1280), dtype=np.float32)
            tgt = np.asarray(tgt, dtype=np.float32)[:self.max_protein_residues]
            return {
                'ids': torch.from_numpy(ids),
                'target_residues': torch.from_numpy(tgt),
                'uid': uid,
                'round_k': torch.tensor(rnd, dtype=torch.long),
                'r_max': torch.tensor(r_max, dtype=torch.long),
            }
        if tgt is None:
            tgt = np.zeros(1280, dtype=np.float32)
        item = {
            'ids': torch.from_numpy(ids),
            'target_emb': torch.from_numpy(tgt.astype(np.float32)),
            'round_k': torch.tensor(rnd, dtype=torch.long),
            'r_max': torch.tensor(r_max, dtype=torch.long),
        }
        if self.trifp_lookup is not None:
            fp = self.trifp_lookup.get(uid)
            if fp is None:
                fp = np.zeros(self.trifp_dim, dtype=np.float32)
            item['trifp_fp'] = torch.from_numpy(np.array(fp, dtype=np.float32))
        return item


def collate_per_residue(batch: list[dict]) -> dict:
    """Pad variable-length per-residue targets to (B, P_max, 1280) and build a
    target_mask (B, P_max) bool where True = real residue. Also stacks ids/rounds
    and carries uid strings (for the supervised-InfoNCE same-target mask)."""
    B = len(batch)
    ids = torch.stack([b['ids'] for b in batch])
    round_k = torch.stack([b['round_k'] for b in batch])
    r_max = torch.stack([b['r_max'] for b in batch])
    uids = [b['uid'] for b in batch]
    lengths = [b['target_residues'].shape[0] for b in batch]
    P_max = max(lengths)
    D = batch[0]['target_residues'].shape[1]
    tgt = torch.zeros(B, P_max, D, dtype=torch.float32)
    mask = torch.zeros(B, P_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = lengths[i]
        tgt[i, :L] = b['target_residues']
        mask[i, :L] = True
    return {'ids': ids, 'target_emb': tgt, 'target_mask': mask,
            'uid': uids, 'round_k': round_k, 'r_max': r_max}


def _parse_round_from_filename(p: str) -> int | None:
    import re
    name = os.path.basename(p).lower().replace('.parquet', '')
    # Try _rN
    m = re.search(r'_r(\d+)(?:_|$)', name)
    if m:
        return int(m.group(1))
    # Try _RN (capital)
    m = re.search(r'_R(\d+)(?:_|$)', os.path.basename(p).replace('.parquet', ''))
    if m:
        return int(m.group(1))
    # Try DRR ids — assume single round files; map to round by metadata externally
    # For smoke run, default to round 4 if can't parse (mid-trajectory)
    if 'drr' in name or 'err' in name:
        return 4
    return None


# ─── Training loop ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=None,
                    help='override cfg seed; also seeds torch (model init) for a reproducible run')
    ap.add_argument('--run_suffix', type=str, default='',
                    help='appended to run_id so multi-seed runs land in separate ckpt dirs')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.seed is not None:
        cfg['seed'] = args.seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    if args.run_suffix:
        cfg['run_id'] = cfg['run_id'] + args.run_suffix
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'device: {device}')

    # Per-residue conditioning: attention-pool (target_pool='attn') OR cross-attn (target_xattn).
    # Both consume the (P,1280) per-residue ESM-2 bank + target_mask.
    per_residue = cfg['model'].get('target_pool', 'mean') == 'attn' or cfg['model'].get('target_xattn', False)

    # Build target embedding lookup
    emb_path = Path(cfg['data']['target_embeddings_parquet']).expanduser()
    emb_df = pd.read_parquet(emb_path)
    target_emb_lookup = {}
    for _, r in emb_df.iterrows():
        uid = str(r.get('uniprot_id') or r.get('uniprot_acc'))
        if uid and uid not in ('None', 'nan', ''):
            if per_residue:
                # residue_embeddings arrives as an object array of (1280,) rows → stack to (P,1280)
                target_emb_lookup[uid] = np.stack(
                    [np.asarray(x, dtype=np.float32) for x in r['residue_embeddings']]).astype(np.float32)
            else:
                target_emb_lookup[uid] = np.asarray(r['embedding'], dtype=np.float32)
    print(f'target_emb_lookup: {len(target_emb_lookup)} UniProts'
          f'{" (per-residue)" if per_residue else ""}')

    # v9: TriFP prediction-fingerprint lookup (uniprot → predicted-Kd vector over the panel).
    trifp_lookup = None
    if cfg['model'].get('trifp_dim', 0) > 0:
        fp_df = pd.read_parquet(Path(cfg['data']['trifp_fingerprint_parquet']).expanduser())
        trifp_lookup = {str(r['uniprot_id']): np.asarray(r['fingerprint'], dtype=np.float32)
                        for _, r in fp_df.iterrows()}
        dim = len(next(iter(trifp_lookup.values())))
        assert dim == cfg['model']['trifp_dim'], \
            f"trifp_dim {cfg['model']['trifp_dim']} != fingerprint bank dim {dim}"
        print(f'trifp_lookup: {len(trifp_lookup)} UniProts, fingerprint dim {dim}')

    # Build dataset
    ds = SelexRoundDataset(
        cfg['data']['families'], target_emb_lookup,
        max_len=cfg['model'].get('max_len', 96),
        max_rows_per_family=cfg['data'].get('max_rows_per_family', 200_000),
        seed=cfg.get('seed', 0),
        winners_parquet=cfg['data'].get('winners_parquet'),
        winner_round=cfg['data'].get('winner_round', 4),
        winner_r_max=cfg['data'].get('winner_r_max', 5),
        winner_repeat=cfg['data'].get('winner_repeat', 1),
        per_residue=per_residue,
        max_protein_residues=cfg['data'].get('max_protein_residues', 1000),
        trifp_lookup=trifp_lookup,
    )
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'],
                        shuffle=True, num_workers=2, drop_last=True,
                        collate_fn=collate_per_residue if per_residue else None)

    # Build model
    model = AptamerDiffusionHybrid(**cfg['model']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'AptamerDiffusionHybrid: {n_params/1e6:.1f}M params')

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training'].get('weight_decay', 0.05),
        betas=tuple(cfg['training'].get('betas', (0.9, 0.95))),
    )
    n_steps = cfg['training']['steps']
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    lam_diff = cfg['training'].get('lam_diff', 0.5)
    # contrastive_clean_forward: True → separate clean-seq forward at t=0 for the proj head
    # (hybrid_cleanfwd, CNN-consistent). False (default) → reuse the noisy denoiser forward
    # at round-t (hybrid_plain recipe; round-noise is a beneficial SELEX-matched aug).
    clean_fwd = cfg['training'].get('contrastive_clean_forward', False)
    # bf16 autocast supported only on CUDA. MPS / CPU must use fp32.
    use_amp = cfg['training'].get('amp', True) and device.type == 'cuda'

    ckpt_dir = Path(cfg['data']['ckpt_dir']).expanduser() / cfg['run_id']
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = ckpt_dir / 'train.log'
    log = open(log_path, 'w')
    def logp(msg):
        print(msg); log.write(msg + '\n'); log.flush()

    logp(f'=== training {cfg["run_id"]} ===')
    logp(f'  steps={n_steps}, batch={cfg["training"]["batch_size"]}, lr={cfg["training"]["lr"]}, lam_diff={lam_diff}')

    step = 0
    data_iter = iter(loader)
    t0 = time.time()
    while step < n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        ids = batch['ids'].to(device, non_blocking=True)
        tgt = batch['target_emb'].to(device, non_blocking=True)
        tgt_mask = batch['target_mask'].to(device, non_blocking=True) if 'target_mask' in batch else None
        trifp_fp = batch['trifp_fp'].to(device, non_blocking=True) if 'trifp_fp' in batch else None
        round_k = batch['round_k'].to(device)
        r_max = batch['r_max'].to(device)
        t = round_to_t(round_k, r_max).to(device)

        # Sample mask per-sequence according to t
        noisy_ids, mask_pos = random_mask(ids, t)

        # kmer features from the CLEAN sequence (proj head only — see model docstring)
        kmer_feats = compute_kmer_batch(ids, device) if model.kmer_token_dim > 0 else None

        # supervised-InfoNCE: mask within-batch rows that share the same target (false negatives)
        same_tgt = None
        if cfg['training'].get('mask_same_target', False):
            if per_residue:
                uids = batch['uid']
                same_tgt = torch.tensor(
                    [[a == b for b in uids] for a in uids], dtype=torch.bool, device=device)
            else:
                same_tgt = torch.cdist(tgt.float(), tgt.float()) < 1e-6        # (B, B) bool

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(noisy_ids, tgt, t, want_denoise=True, want_proj=True,
                        kmer_feats=kmer_feats, clean_ids=ids if clean_fwd else None,
                        target_mask=tgt_mask, trifp_fp=trifp_fp)
            L_d = denoise_loss(out['logits'], ids, mask_pos, t)
            L_c = contrast_loss(out['proj'], out['target_proj'], same_target_mask=same_tgt)
            L = L_c + lam_diff * L_d

        opt.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training'].get('grad_clip', 1.0))
        opt.step()
        sched.step()

        if step % 50 == 0:
            dt = time.time() - t0
            sps = (step + 1) / max(dt, 1)
            logp(f'  step {step:>6}  L={L.item():.4f}  Ld={L_d.item():.4f}  Lc={L_c.item():.4f}  '
                  f'lr={opt.param_groups[0]["lr"]:.2e}  {sps:.2f} step/s')

        if step % cfg['training'].get('ckpt_every', 2500) == 0 and step > 0:
            torch.save({'step': step, 'model': model.state_dict(), 'config': cfg},
                        ckpt_dir / f'ckpt_step{step}.pt')
            logp(f'  saved ckpt step={step}')

        step += 1

    torch.save({'step': step, 'model': model.state_dict(), 'config': cfg},
                ckpt_dir / f'ckpt_step{step}.pt')
    logp(f'=== DONE — final ckpt step={step} ===')
    log.close()


if __name__ == '__main__':
    main()
