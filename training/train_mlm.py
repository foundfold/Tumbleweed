"""MLM-baseline trainer — mirror of train_contrastive.py for the ablation.

Everything is identical except the objective:
  - Encoder backbone: same AptamerEncoder layout (d_model / nhead / num_layers)
    wrapped with an MLM head (vocab-projection).
  - Loss: cross-entropy on 15%-masked tokens (BERT-style).
  - Dataset: MLMAptamerDataset — same sources, but no round-class split.

Output run dir layout same as the contrastive trainer
(`~/Desktop/autoRNA_data/tumbleweed/training_runs/<run_id>/`).

After both runs finish, evaluate via the same downstream protocols (linear
probe on A4, AptaTrans/Li API, InstructNA panel) to get the apples-to-apples
"contrastive vs MLM" comparison.

Usage:
  python3 train_mlm.py --config configs/mlm_baseline.yaml
  torchrun --nproc-per-node=4 train_mlm.py --config configs/mlm_baseline.yaml
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
from aptamer_dataset import MLMAptamerDataset
from aptamer_mlm import AptamerMLMModel, mlm_loss
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


def train(cfg, resume=None, max_seconds: float | None = None):
    rank, world, local = setup_ddp()
    main = is_main(rank)
    device = (torch.device(f'cuda:{local}') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    if main: print(f'rank {rank}/{world}  device {device}')

    seed = cfg.get('seed', 0) + rank
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    if main: print('\n=== Dataset ===')
    ds = MLMAptamerDataset(
        sources=cfg['data']['sources'],
        max_len=cfg['data']['max_len'],
        test_exclusion_parquet=cfg['data'].get('test_exclusion_parquet'),
        mask_prob=cfg['data'].get('mask_prob', 0.15),
        seed=seed,
    )
    iterable = _IterableWrap(ds, max_iters=cfg['training']['steps'] * cfg['training']['batch_size'])
    loader = DataLoader(
        iterable, batch_size=cfg['training']['batch_size'],
        num_workers=cfg['data'].get('num_workers', 0),
        collate_fn=ds.collate, pin_memory=device.type == 'cuda',
    )

    if main: print('\n=== Model ===')
    model_cfg = dict(cfg['model'])
    model_cfg.pop('embed_dim', None)   # MLM doesn't need a projection embed_dim
    model = AptamerMLMModel(**model_cfg).to(device)
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

    run_dir = Path.home() / 'Desktop/autoRNA_data/tumbleweed/training_runs' / cfg['run_id']
    if main:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / 'config.yaml', 'w') as f: yaml.dump(cfg, f, sort_keys=False)
        train_log = open(run_dir / 'train.jsonl', 'a')

    start_step = 0
    if resume and resume.exists():
        ck = torch.load(resume, map_location=device)
        (model.module if world > 1 else model).load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler'])
        start_step = ck['step'] + 1
        if main: print(f'\nresumed from step {start_step}')

    if main: print(f'\n=== MLM training ({total} steps from {start_step}) ===')
    model.train()
    t0 = time.time()
    losses = []
    step = start_step
    for batch in loader:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        label_ids = batch['label_ids'].to(device, non_blocking=True)
        active = batch['active_mask'].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids)
            loss = mlm_loss(logits, label_ids, active)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
        scaler.step(opt); scaler.update(); sched.step()
        losses.append(float(loss.item()))

        if main and (step + 1) % cfg['log']['every_steps'] == 0:
            recent = losses[-cfg['log']['every_steps']:]
            avg = sum(recent) / len(recent)
            elapsed = time.time() - t0
            rate = (step + 1 - start_step) / max(1, elapsed)
            ppl = math.exp(avg) if avg < 20 else float('inf')
            entry = dict(step=step + 1, loss=avg, perplexity=ppl,
                         lr=sched.get_last_lr()[0], step_per_s=rate, ts=time.time())
            print(f'  step {step+1:>7}  loss={avg:.4f}  ppl={ppl:.2f}  '
                  f'lr={entry["lr"]:.2e}  {rate:.1f} step/s  {elapsed:.0f}s')
            train_log.write(json.dumps(entry) + '\n'); train_log.flush()

        if main and (step + 1) % cfg.get('eval', {}).get('every_steps', 99999) == 0:
            model_for_eval = (model.module if world > 1 else model)
            model_for_eval.eval()
            metrics = run_all_benchmarks(
                encode_fn=lambda ids: model_for_eval.encode(ids),
                max_len=cfg['data']['max_len'], device=device,
                mlm_fn=lambda ids: model_for_eval(ids),  # MLM head outputs vocab logits
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
            with open(run_dir / 'eval.jsonl', 'a') as f:
                f.write(json.dumps(metrics) + '\n')
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
            # Rolling keep
            ckpts = sorted(run_dir.glob('ckpt_step*.pt'),
                           key=lambda p: int(p.stem.split('step')[-1]))
            for old in ckpts[:-cfg['ckpt']['keep_last']]:
                old.unlink()

        step += 1
        if step >= total: break
        if max_seconds is not None and (time.time() - t0) > max_seconds:
            if main: print(f'\n--max-seconds {max_seconds} elapsed at step {step}; stopping.')
            break

    if main:
        train_log.close()
        print('\nDone.')
    cleanup_ddp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(__file__).parent / 'configs/mlm_baseline.yaml'))
    ap.add_argument('--resume', type=Path, default=None)
    ap.add_argument('--max-seconds', type=float, default=None,
                    help='Hard cap on wall-time; trainer breaks between batches when exceeded.')
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    train(cfg, args.resume, max_seconds=args.max_seconds)


if __name__ == '__main__':
    main()
