"""Train AptamerRinalmoScorer: v10 heads (chem token + ESM-2 FiLM + contrastive round
teacher) on a FROZEN, PRECOMPUTED RiNALMo-giga backbone.

Prereq: run scripts/precompute_rinalmo_embeddings.py first to cache the frozen
RiNALMo mean-pool embeddings for every corpus sequence. This script never loads
RiNALMo — it reads the cache, so each step is just the light head forward/backward.

Per step:
  1. sample batch of (rinalmo_emb, target_emb, chem, round_k, r_max, uid)
  2. t = round_to_t(round_k, r_max)
  3. out = model(rinalmo_emb, target_emb, chem, t) → {proj, target_proj}
  4. L = contrast_loss(proj, target_proj, same_target_mask)   # supervised InfoNCE
  (no denoiser: scorer-first.)

Usage:
  python3 training/train_rinalmo_scorer.py --config training/configs/tumbleweed_rinalmo_scorer.yaml
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aptamer_rinalmo_scorer import AptamerRinalmoScorer
from aptamer_diffusion_hybrid import contrast_loss, round_to_t

CHEM_ID = {'RNA': 0, 'DNA': 1}


def _parse_round_from_filename(p: str):
    import re
    name = os.path.basename(p).lower().replace('.parquet', '')
    m = re.search(r'_r(\d+)(?:_|$)', name)
    if m:
        return int(m.group(1))
    m = re.search(r'_R(\d+)(?:_|$)', os.path.basename(p).replace('.parquet', ''))
    if m:
        return int(m.group(1))
    if 'drr' in name or 'err' in name:
        return 4
    return None


class RinalmoScorerDataset(Dataset):
    """(rinalmo_emb, target_emb, chem, round_k, r_max, uid) tuples.

    Sequences without a cached RiNALMo embedding or targets without an ESM-2
    embedding are dropped (logged). Winners are injected + upsampled exactly like
    SelexRoundDataset so InfoNCE batches stay target-diverse."""
    def __init__(self, cfg, rinalmo_lookup, target_lookup):
        self.rin = rinalmo_lookup            # (sequence, chem) → (1280,)
        self.tgt = target_lookup             # uniprot → (1280,)
        rng = np.random.default_rng(cfg.get('seed', 0))
        rows = []
        n_miss_rin = n_miss_tgt = 0
        for fam in cfg['data']['families']:
            chem = fam['chemistry']
            uid = fam['target_uniprot']
            paths = sorted(glob(os.path.expanduser(fam['rounds_glob'])))
            fam_rows = []
            for p in paths:
                rnd = _parse_round_from_filename(p)
                if rnd is None:
                    continue
                try:
                    df = pd.read_parquet(p, columns=['sequence'])
                except Exception:
                    continue
                for s in df['sequence'].astype(str):
                    fam_rows.append((s, rnd, fam['r_max'], uid, chem))
            cap = cfg['data'].get('max_rows_per_family', 200_000)
            if len(fam_rows) > cap:
                keep = rng.choice(len(fam_rows), cap, replace=False)
                fam_rows = [fam_rows[i] for i in keep]
            print(f'  family {fam["name"]:<18} {uid}: {len(fam_rows)} rows ({chem})')
            rows.extend(fam_rows)
        n_selex = len(rows)

        wp_path = cfg['data'].get('winners_parquet')
        if wp_path:
            wp = pd.read_parquet(os.path.expanduser(wp_path))
            wr = cfg['data'].get('winner_round', 4)
            wrm = cfg['data'].get('winner_r_max', 5)
            rep = cfg['data'].get('winner_repeat', 1)
            w_rows = [(str(s), wr, wrm, str(u), str(c))
                      for s, u, c in zip(wp['sequence'], wp['uniprot'], wp['chem'])]
            rows.extend(w_rows * rep)
            print(f'  winners: {len(w_rows)} × repeat {rep} = {len(w_rows) * rep} rows '
                  f'over {wp["uniprot"].nunique()} targets')

        kept = []
        for s, rnd, rmax, uid, chem in rows:
            if (s, chem) not in self.rin:
                n_miss_rin += 1
                continue
            if uid not in self.tgt:
                n_miss_tgt += 1
                continue
            kept.append((s, rnd, rmax, uid, chem))
        self.rows = kept
        print(f'RinalmoScorerDataset: {len(kept)} rows kept '
              f'(SELEX {n_selex} + winners; dropped {n_miss_rin} no-RiNALMo, {n_miss_tgt} no-ESM2)')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        s, rnd, rmax, uid, chem = self.rows[idx]
        return {
            'rinalmo_emb': torch.from_numpy(np.asarray(self.rin[(s, chem)], dtype=np.float32)),
            'target_emb': torch.from_numpy(np.asarray(self.tgt[uid], dtype=np.float32)),
            'chem': torch.tensor(CHEM_ID[chem], dtype=torch.long),
            'round_k': torch.tensor(rnd, dtype=torch.long),
            'r_max': torch.tensor(rmax, dtype=torch.long),
            'uid': uid,
        }


def collate(batch):
    return {
        'rinalmo_emb': torch.stack([b['rinalmo_emb'] for b in batch]),
        'target_emb': torch.stack([b['target_emb'] for b in batch]),
        'chem': torch.stack([b['chem'] for b in batch]),
        'round_k': torch.stack([b['round_k'] for b in batch]),
        'r_max': torch.stack([b['r_max'] for b in batch]),
        'uid': [b['uid'] for b in batch],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--run_suffix', type=str, default='')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.seed is not None:
        cfg['seed'] = args.seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    if args.run_suffix:
        cfg['run_id'] = cfg['run_id'] + args.run_suffix
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'device: {device}')

    # cached frozen-RiNALMo embeddings: (sequence, chem) → 1280-d
    rin_df = pd.read_parquet(os.path.expanduser(cfg['data']['rinalmo_cache']))
    rinalmo_lookup = {(str(s), str(c)): np.asarray(e, dtype=np.float32)
                      for s, c, e in zip(rin_df['sequence'], rin_df['chem'], rin_df['embedding'])}
    print(f'rinalmo_lookup: {len(rinalmo_lookup)} cached (sequence, chem) pairs')

    # ESM-2 target embeddings: uniprot → 1280-d
    emb_df = pd.read_parquet(os.path.expanduser(cfg['data']['target_embeddings_parquet']))
    target_lookup = {}
    for _, r in emb_df.iterrows():
        uid = str(r.get('uniprot_id') or r.get('uniprot_acc'))
        if uid and uid not in ('None', 'nan', ''):
            target_lookup[uid] = np.asarray(r['embedding'], dtype=np.float32)
    print(f'target_lookup: {len(target_lookup)} UniProts')

    ds = RinalmoScorerDataset(cfg, rinalmo_lookup, target_lookup)
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True,
                        num_workers=2, drop_last=True, collate_fn=collate)

    model = AptamerRinalmoScorer(**cfg['model']).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'AptamerRinalmoScorer: {n_train/1e6:.2f}M trainable params')

    opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'],
                            weight_decay=cfg['training'].get('weight_decay', 0.05),
                            betas=tuple(cfg['training'].get('betas', (0.9, 0.95))))
    n_steps = cfg['training']['steps']
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    mask_same = cfg['training'].get('mask_same_target', True)

    ckpt_dir = Path(os.path.expanduser(cfg['data']['ckpt_dir'])) / cfg['run_id']
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = open(ckpt_dir / 'train.log', 'w')

    def logp(msg):
        print(msg); log.write(msg + '\n'); log.flush()

    logp(f'=== training {cfg["run_id"]} ===  steps={n_steps} batch={cfg["training"]["batch_size"]}')
    step = 0
    data_iter = iter(loader)
    t0 = time.time()
    while step < n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        rin = batch['rinalmo_emb'].to(device)
        tgt = batch['target_emb'].to(device)
        chem = batch['chem'].to(device)
        t = round_to_t(batch['round_k'].to(device), batch['r_max'].to(device)).to(device)

        same_tgt = None
        if mask_same:
            uids = batch['uid']
            same_tgt = torch.tensor([[a == b for b in uids] for a in uids],
                                    dtype=torch.bool, device=device)

        out = model(rin, tgt, chem, t)
        L = contrast_loss(out['proj'], out['target_proj'], same_target_mask=same_tgt)

        opt.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training'].get('grad_clip', 1.0))
        opt.step()
        sched.step()

        if step % 50 == 0:
            sps = (step + 1) / max(time.time() - t0, 1)
            logp(f'  step {step:>6}  L={L.item():.4f}  lr={opt.param_groups[0]["lr"]:.2e}  {sps:.1f} step/s')
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
