"""Hybrid MLM + contrastive trainer. Uses ContrastiveAptamerDataset (with curriculum)
PLUS applies BERT-style 15% masking to the anchor for the MLM head.

Logs:
  step, loss_total, loss_mlm, loss_contrast, lr, step_per_s
Eval every N: panel R@1/R@10/Kd-ρ + A4 r + Li AUC + UTex r + test R@1
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
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aptamer_dataset import ContrastiveAptamerDataset, _bert_mask, encode
from aptamer_hybrid import AptamerHybridModel, hybrid_loss
from eval_benchmarks import run_all as run_all_benchmarks


def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get('LOCAL_RANK', 0))
    return 0, 1, 0


def is_main(r): return r == 0
def cleanup_ddp():
    if dist.is_initialized(): dist.destroy_process_group()


class _IterableWrap(IterableDataset):
    def __init__(self, ds, max_iters):
        self.ds = ds; self.max_iters = max_iters
    def __iter__(self):
        it = iter(self.ds)
        for _ in range(self.max_iters): yield next(it)


def hybrid_collate(batch, mask_prob: float, rng_seed: int):
    """Wraps ContrastiveAptamerDataset.collate output and adds masked-anchor + labels."""
    import torch
    anchor_ids = np.stack([x['anchor_ids'] for x in batch])
    positive_ids = np.stack([x['positive_ids'] for x in batch])
    # negative ids come from the dataset's r0 sampler — generated in main collate path.
    # We rebuild the collate here so we control the masking too.
    rng = np.random.default_rng(rng_seed)
    masked_anchor = np.empty_like(anchor_ids)
    anchor_labels = np.empty_like(anchor_ids)
    anchor_active = np.zeros_like(anchor_ids, dtype=bool)
    for i, ids in enumerate(anchor_ids):
        mi, li, ai = _bert_mask(ids, mask_prob, rng)
        masked_anchor[i] = mi
        anchor_labels[i] = li
        anchor_active[i] = ai
    return dict(
        anchor_ids=torch.from_numpy(anchor_ids),
        positive_ids=torch.from_numpy(positive_ids),
        masked_anchor=torch.from_numpy(masked_anchor),
        anchor_labels=torch.from_numpy(anchor_labels),
        anchor_active=torch.from_numpy(anchor_active),
    )


def train(cfg, resume=None):
    rank, world, local = setup_ddp()
    main = is_main(rank)
    device = (torch.device(f'cuda:{local}') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    if main: print(f'rank {rank}/{world}  device {device}')

    seed = cfg.get('seed', 0) + rank
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    if main: print('\n=== Dataset ===')
    ds = ContrastiveAptamerDataset(
        sources=cfg['data']['sources'],
        max_len=cfg['data']['max_len'],
        test_exclusion_parquet=cfg['data'].get('test_exclusion_parquet'),
        shuffle_buffer=cfg['data'].get('shuffle_buffer', 50_000),
        seed=seed,
    )

    phases = cfg.get('curriculum', {}).get('phases') if cfg.get('curriculum', {}).get('enabled') else None
    def phase_for(step):
        idx = 0
        for i, p in enumerate(phases):
            if step >= p['start_step']: idx = i
        return idx

    mask_prob = cfg['data'].get('mask_prob', 0.15)

    def build_loader():
        iterable = _IterableWrap(ds, max_iters=cfg['training']['steps'] * cfg['training']['batch_size'])
        return DataLoader(
            iterable,
            batch_size=cfg['training']['batch_size'],
            num_workers=cfg['data'].get('num_workers', 0),
            collate_fn=lambda b: hybrid_collate(b, mask_prob, seed),
            pin_memory=device.type == 'cuda',
        )

    cur_phase = -1
    if phases:
        cur_phase = phase_for(0)
        if main: print(f'\n=== Curriculum: {len(phases)} phases, start phase {cur_phase} ===')
        ds.set_active_neg_tags(phases[cur_phase]['neg_tags'])
    loader = build_loader()

    if main: print('\n=== Model ===')
    model = AptamerHybridModel(**cfg['model']).to(device)
    if main: print(f'  {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')

    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local] if device.type == 'cuda' else None,
                    find_unused_parameters=False)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg['training']['lr'],
        betas=tuple(cfg['training']['betas']),
        weight_decay=cfg['training']['weight_decay'],
    )
    total = cfg['training']['steps']
    warm = cfg['training']['warmup_steps']
    def lr_l(s):
        if s < warm: return s / max(1, warm)
        p = (s - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_l)

    use_amp = cfg['training']['amp'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    alpha = cfg['training'].get('hybrid_alpha', 1.0)
    beta = cfg['training'].get('hybrid_beta', 0.5)
    tau = cfg['training'].get('tau', 0.2)

    run_dir = Path.home() / 'Desktop/autoRNA_data/tumbleweed/training_runs' / cfg['run_id']
    if main:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / 'config.yaml', 'w') as f: yaml.dump(cfg, f, sort_keys=False)
        train_log = open(run_dir / 'train.jsonl', 'a')
        eval_log = open(run_dir / 'eval.jsonl', 'a')

    start_step = 0
    if resume and resume.exists():
        ck = torch.load(resume, map_location=device)
        (model.module if world > 1 else model).load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler'])
        start_step = ck['step'] + 1
        if main: print(f'\nresumed from step {start_step}')

    if main: print(f'\n=== Hybrid training (α={alpha} β={beta} τ={tau}, {total} steps from {start_step}) ===')
    model.train()
    t0 = time.time()
    losses_tot, losses_mlm, losses_con = [], [], []
    step = start_step

    if phases:
        cur_phase = phase_for(start_step)
        ds.set_active_neg_tags(phases[cur_phase]['neg_tags'])
        loader = build_loader()
    data_iter = iter(loader)

    while step < total:
        if phases:
            want_phase = phase_for(step)
            if want_phase != cur_phase:
                if main:
                    print(f'  ── curriculum step {step}: phase {cur_phase} → {want_phase}  '
                          f'neg_tags={phases[want_phase]["neg_tags"]} ──')
                ds.set_active_neg_tags(phases[want_phase]['neg_tags'])
                loader = build_loader()
                data_iter = iter(loader)
                cur_phase = want_phase

        batch = next(data_iter)
        masked_anc = batch['masked_anchor'].to(device, non_blocking=True)
        anc_lab = batch['anchor_labels'].to(device, non_blocking=True)
        anc_act = batch['anchor_active'].to(device, non_blocking=True)
        pos_ids = batch['positive_ids'].to(device, non_blocking=True)
        # negatives come from the dataset's r0 sampler — sample on the fly here
        neg_np = ds.sample_negatives(masked_anc.shape[0])
        neg_ids = torch.from_numpy(neg_np).to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            # Masked anchor → both logits (for MLM) and projection (for contrastive)
            anc_out = model(masked_anc, want_mlm=True, want_proj=True)
            # Positive + negative → projection only
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pos_out = model(pos_ids, want_mlm=False, want_proj=True)
                neg_out = model(neg_ids, want_mlm=False, want_proj=True)
            loss, l_mlm, l_con = hybrid_loss(
                anc_out['logits'], anc_lab, anc_act,
                anc_out['projection'], pos_out['projection'], neg_out['projection'],
                alpha=alpha, beta=beta, tau=tau,
            )

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
        scaler.step(opt); scaler.update(); sched.step()

        losses_tot.append(float(loss.item()))
        losses_mlm.append(float(l_mlm.item()))
        losses_con.append(float(l_con.item()))

        if main and (step + 1) % cfg['log']['every_steps'] == 0:
            n = cfg['log']['every_steps']
            tot = sum(losses_tot[-n:]) / n
            ml = sum(losses_mlm[-n:]) / n
            co = sum(losses_con[-n:]) / n
            elapsed = time.time() - t0
            rate = (step + 1 - start_step) / max(1, elapsed)
            entry = dict(step=step + 1, loss=tot, loss_mlm=ml, loss_con=co,
                         lr=sched.get_last_lr()[0], step_per_s=rate, ts=time.time())
            print(f'  step {step+1:>7}  loss={tot:.4f} (mlm={ml:.4f} con={co:.4f})  '
                  f'lr={entry["lr"]:.2e}  {rate:.1f} step/s  {elapsed:.0f}s')
            train_log.write(json.dumps(entry) + '\n'); train_log.flush()

        if main and (step + 1) % cfg.get('eval', {}).get('every_steps', 99999) == 0:
            model_for_eval = (model.module if world > 1 else model)
            model_for_eval.eval()
            metrics = run_all_benchmarks(
                encode_fn=lambda ids: model_for_eval.encode(ids),
                max_len=cfg['data']['max_len'], device=device,
                mlm_fn=lambda ids: model_for_eval(ids, want_mlm=True, want_proj=False)['logits'],
            )
            metrics['step'] = step + 1; metrics['ts'] = time.time()
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
            model.train()

        if main and (step + 1) % cfg['ckpt']['every_steps'] == 0:
            ckpt_path = run_dir / f'ckpt_step{step+1}.pt'
            torch.save({
                'step': step,
                'model': (model.module if world > 1 else model).state_dict(),
                'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict(),
                'config': cfg,
            }, ckpt_path)
            print(f'  ckpt → {ckpt_path}')
            ckpts = sorted(run_dir.glob('ckpt_step*.pt'),
                           key=lambda p: int(p.stem.split('step')[-1]))
            for old in ckpts[:-cfg['ckpt']['keep_last']]:
                old.unlink()

        step += 1

    if main:
        train_log.close(); eval_log.close()
        print('\nDone.')
    cleanup_ddp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(__file__).parent / 'configs/hybrid_30min.yaml'))
    ap.add_argument('--resume', type=Path, default=None)
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    train(cfg, args.resume)


if __name__ == '__main__':
    main()
