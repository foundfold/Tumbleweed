"""Train the AptamerDecoder against a FROZEN AptamerEncoder ckpt.

The encoder is loaded from a paper-headline ckpt (e.g. 14M chinchilla) and held
fixed. The decoder is trained to reconstruct the input sequence from the
encoder's pooled embedding (teacher-forced CE). This makes the encoder's latent
space "invertible" enough for downstream BO-based generation.

Sampling strategy: oversample MECP2 + SNCA positives (GREEN scorer targets) so
the decoder is most accurate in the embedding regions we'll actually optimize
during generation.

Run dir: ~/Desktop/autoRNA_data/tumbleweed/training_runs/<run_id>/
  config.yaml, train.jsonl, ckpt_step{N}.pt (decoder weights only)
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
from torch.utils.data import DataLoader, IterableDataset

# CUDA 13.0 cuBLAS instability — disable TF32, use bf16 autocast.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aptamer_dataset import ContrastiveAptamerDataset, encode, PAD_ID
from aptamer_encoder import AptamerEncoder
from aptamer_decoder import AptamerDecoder, decoder_loss, prepare_decoder_io


class _DecoderIterable(IterableDataset):
    """Wrap ContrastiveAptamerDataset.iter() and emit only the anchor seq +
    anchor target embedding (we don't need positive/negative for decoder training).
    """
    def __init__(self, ds, max_steps: int):
        self.ds = ds
        self.max_steps = max_steps

    def __iter__(self):
        it = iter(self.ds)
        for _ in range(self.max_steps):
            yield next(it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--encoder-ckpt', required=True, type=Path,
                    help='Path to the frozen encoder checkpoint')
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print(f'device: {device}')

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
    print(f'  {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M encoder params (frozen), '
          f'cond={encoder.target_cond_mode}')

    # 2. Decoder
    decoder = AptamerDecoder(
        encoder_d_model=model_cfg.get('d_model', 384),
        **cfg['model'],
    ).to(device)
    print(f'  {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M decoder params (trainable)')

    # 3. Dataset (reuse the encoder's training dataset to sample anchor sequences)
    ds = ContrastiveAptamerDataset(
        sources=cfg['data']['sources'],
        max_len=cfg['data']['max_len'],
        test_exclusion_parquet=cfg['data'].get('test_exclusion_parquet'),
        shuffle_buffer=cfg['data'].get('shuffle_buffer', 50_000),
        seed=cfg.get('seed', 0),
        target_embeddings_parquet=cfg['data'].get('target_embeddings_parquet'),
    )
    target_emb_table_gpu = None
    if ds._target_emb_table is not None:
        target_emb_table_gpu = torch.from_numpy(ds._target_emb_table).to(device)
        print(f'  target embedding table on {device}: {target_emb_table_gpu.shape}')

    iterable = _DecoderIterable(ds, max_steps=cfg['training']['steps'] * cfg['training']['batch_size'])
    loader = DataLoader(iterable,
                        batch_size=cfg['training']['batch_size'],
                        num_workers=cfg['data'].get('num_workers', 0),
                        collate_fn=ds.collate,
                        pin_memory=device.type == 'cuda')

    # 4. Optimizer + schedule
    opt = torch.optim.AdamW(
        decoder.parameters(),
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

    use_amp = cfg['training'].get('amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 5. Run dir + logging
    run_dir = (Path.home() / 'Desktop/autoRNA_data/tumbleweed/training_runs'
               / cfg['run_id'])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    train_log = open(run_dir / 'train.jsonl', 'a')

    # 6. Training loop
    print(f'\n=== Training ({total_steps} steps) ===')
    t0 = time.time()
    loss_window: list[float] = []
    step = 0
    for batch in loader:
        anchor_ids = batch['anchor_ids'].to(device, non_blocking=True)
        # Get encoder embedding (frozen, no grad)
        if target_emb_table_gpu is not None and encoder.target_cond_mode is not None:
            anc_ti = batch['anchor_target_idx'].to(device, non_blocking=True)
            anc_te = target_emb_table_gpu[anc_ti]
        else:
            anc_te = None
        with torch.no_grad():
            enc_emb = encoder.encode(anchor_ids, target_emb=anc_te)

        # Decoder forward — teacher-forced reconstruction of anchor_ids
        dec_in, dec_target = prepare_decoder_io(anchor_ids, pad_id=PAD_ID)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = decoder(enc_emb, dec_in)
            loss = decoder_loss(logits, dec_target, pad_id=PAD_ID)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg['training']['grad_clip'])
        scaler.step(opt)
        scaler.update()
        sched.step()

        loss_window.append(float(loss.item()))
        step += 1

        if step % cfg['log']['every_steps'] == 0:
            n = cfg['log']['every_steps']
            avg = sum(loss_window[-n:]) / n
            elapsed = time.time() - t0
            rate = step / max(1, elapsed)
            entry = dict(step=step, loss=avg, lr=sched.get_last_lr()[0],
                         step_per_s=rate, ts=time.time())
            # Recon accuracy: token-level argmax match on this batch (skip PAD)
            with torch.no_grad():
                preds = logits.argmax(-1)
                mask = (dec_target != PAD_ID)
                token_acc = (preds[mask] == dec_target[mask]).float().mean().item()
            entry['token_acc'] = token_acc
            print(f'  step {step:>6}  loss={avg:.4f}  tok_acc={token_acc:.3f}  '
                  f'lr={entry["lr"]:.2e}  {rate:.1f} step/s  {elapsed:.0f}s')
            train_log.write(json.dumps(entry) + '\n'); train_log.flush()

        if step % cfg['ckpt']['every_steps'] == 0:
            ck_path = run_dir / f'decoder_step{step}.pt'
            torch.save({
                'step': step,
                'decoder': decoder.state_dict(),
                'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict(),
                'config': cfg,
                'encoder_ckpt': str(args.encoder_ckpt),
            }, ck_path)
            print(f'  ckpt → {ck_path}')
            # Keep last N
            ckpts = sorted(run_dir.glob('decoder_step*.pt'),
                           key=lambda p: int(p.stem.split('step')[-1]))
            for old in ckpts[:-cfg['ckpt']['keep_last']]:
                old.unlink()
        if step >= total_steps:
            break

    train_log.close()
    print('\nDone.')


if __name__ == '__main__':
    main()
