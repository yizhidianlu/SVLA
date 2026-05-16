#!/usr/bin/env python
"""Pretrain the depth VQ-VAE on LIBERO-extracted depth maps.

Phase 1.7b. Reads `.npz` files produced by `scripts/extract_libero_depth_gt.py`
(one per LIBERO demo), trains a `VQVAE(VQVAEConfig())` (K=256 codes, dim=160,
16x16 latent) per QDepth-VLA §3.2, and saves the codebook + decoder weights
to be loaded into `GeoRelVLA(depth_codebook=...)` at Phase 1.7c training time.

Typical Phase-1 invocation (run on AutoDL A800 in the geo-rel-vla env):

    python scripts/pretrain_vqvae.py \\
        --head depth \\
        --data-dir /autodl-fs/data/svla/data/libero_depth_gt/libero_spatial \\
        --out-dir /autodl-fs/data/svla/runs/vqvae_depth_phase1 \\
        --epochs 6 --batch 256 --lr 1e-5

Cost: ~6 GPU-h on A800 for full LIBERO-Spatial corpus (~500 demos x ~80 frames
= ~40k frames). Per QDepth-VLA §3.2 a smaller corpus already converges, so
6 epochs at batch 256 is plenty.

Saved artefacts:
  <out_dir>/vqvae_step{N}.pt          — full VQ-VAE state_dict (re-loadable)
  <out_dir>/codebook.pt                — just the (K, embedding_dim) codebook
                                         tensor (what GeoRelVLA needs)
  <out_dir>/metrics.json               — per-step train metrics (recon RMSE,
                                         codebook usage, total loss)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _lazy_torch():
    import torch  # noqa: PLC0415
    return torch


# ----- dataset -----------------------------------------------------------


class _DepthShardDataset:
    """Iterate over (T, H, W) depth tensors stored across `.npz` shards.

    We keep the implementation deliberately lightweight (no torch.utils.data
    DataLoader) so the file is independent of torch when --help is invoked.

    `head_mode` controls the output channels:
      - "depth": yield (1, H, W) depth in metres
      - "normal": compute analytic surface-normal from depth and yield (3, H, W)
    """

    def __init__(
        self,
        data_dir: Path,
        target_resolution: int = 256,
        depth_clip_m: float = 5.0,
        shuffle_shards: bool = True,
        seed: int = 0,
        head_mode: str = "depth",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.target_resolution = target_resolution
        self.depth_clip_m = depth_clip_m
        self.shuffle_shards = shuffle_shards
        self.rng = np.random.default_rng(seed)
        self.head_mode = head_mode

        self.shards = sorted(self.data_dir.glob("*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no .npz shards under {self.data_dir}")

    def shard_count(self) -> int:
        return len(self.shards)

    def stream_frames(self) -> Iterator[np.ndarray]:
        """Yield individual (C, H, W) tensors. C=1 for depth, C=3 for normal."""
        order = list(range(len(self.shards)))
        if self.shuffle_shards:
            self.rng.shuffle(order)
        for idx in order:
            try:
                shard = np.load(self.shards[idx], allow_pickle=True)
                depth = shard["depth"].astype(np.float32)        # (T, H, W)
            except Exception:
                continue
            depth = np.clip(depth, 0.0, self.depth_clip_m)
            for t in range(depth.shape[0]):
                if self.head_mode == "depth":
                    yield depth[t : t + 1]                       # (1, H, W)
                elif self.head_mode == "normal":
                    from georel_vla.data.normal_from_depth import depth_to_normal_np  # noqa: PLC0415
                    n = depth_to_normal_np(depth[t])             # (H, W, 3)
                    yield n.transpose(2, 0, 1)                   # (3, H, W)
                else:
                    raise ValueError(f"unknown head_mode {self.head_mode}")

    def yield_batches(self, batch_size: int, shuffle_within: bool = True) -> Iterator[np.ndarray]:
        """Group consecutive frames into batches of (B, 1, H, W)."""
        buf: list[np.ndarray] = []
        for frame in self.stream_frames():
            buf.append(frame)
            if len(buf) >= batch_size:
                arr = np.stack(buf, axis=0)
                if shuffle_within:
                    self.rng.shuffle(arr)
                yield arr
                buf.clear()
        if buf:
            yield np.stack(buf, axis=0)


# ----- training loop -----------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", choices=["depth", "normal", "plane"], default="depth")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Folder of <suite>/*.npz shards (e.g., /autodl-fs/data/svla/data/libero_depth_gt/libero_spatial)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num-embeddings", type=int, default=256)
    p.add_argument("--embedding-dim", type=int, default=160)
    p.add_argument("--latent-grid", type=int, default=16,
                   help="Output latent grid HxW; 16 -> downsample factor 16 on 256x256 input")
    p.add_argument("--depth-clip-m", type=float, default=5.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every-steps", type=int, default=2000)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--max-batches-per-epoch", type=int, default=0,
                   help="Cap iterations per epoch (0 = unlimited; useful for smoke runs)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("pretrain_vqvae")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.head != "depth":
        log.warning("--head=%s currently uses the same VQ-VAE arch as depth; "
                    "tune in_channels for normal (3) / plane (1 categorical) when needed",
                    args.head)

    torch = _lazy_torch()
    from georel_vla.codebooks.vqvae import VQVAE, VQVAEConfig

    in_ch = 3 if args.head == "normal" else 1
    cfg = VQVAEConfig(
        in_channels=in_ch, out_channels=in_ch,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        downsample=256 // args.latent_grid,
    )
    log.info("VQ-VAE config: %s", asdict(cfg))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = VQVAE(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ds = _DepthShardDataset(
        args.data_dir, target_resolution=256,
        depth_clip_m=args.depth_clip_m, seed=args.seed,
        head_mode=args.head,
    )
    log.info("dataset: %d shards under %s", ds.shard_count(), args.data_dir)

    metrics = []
    step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        ep_start = time.time()
        ep_batches = 0
        for batch_np in ds.yield_batches(args.batch):
            x = torch.from_numpy(batch_np).to(args.device, non_blocking=True)
            recon, indices, losses = model(x)
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            step += 1
            ep_batches += 1

            if step % args.log_every_steps == 0:
                # codebook usage = unique-codes / K
                used = int(indices.unique().numel())
                rmse = float(torch.sqrt(losses["recon"]).item())
                row = {
                    "epoch": epoch, "step": step,
                    "loss_total": float(losses["total"].item()),
                    "loss_recon": float(losses["recon"].item()),
                    "loss_codebook": float(losses["codebook"].item()),
                    "loss_commit": float(losses["commitment"].item()),
                    "rmse_m": rmse,
                    "codes_used": used,
                    "codes_used_frac": used / cfg.num_embeddings,
                    "elapsed_sec": time.time() - t_start,
                }
                metrics.append(row)
                log.info(
                    "ep=%d step=%d total=%.4f recon=%.4f rmse=%.3fm codes=%d/%d t=%.0fs",
                    epoch, step, row["loss_total"], row["loss_recon"], rmse,
                    used, cfg.num_embeddings, row["elapsed_sec"],
                )

            if args.save_every_steps and step % args.save_every_steps == 0:
                _save_ckpt(model, args.out_dir, step, metrics)

            if args.max_batches_per_epoch and ep_batches >= args.max_batches_per_epoch:
                break

        log.info("epoch %d done in %.1fs (%d batches)", epoch, time.time() - ep_start, ep_batches)

    # final checkpoint
    _save_ckpt(model, args.out_dir, step, metrics)
    log.info("DONE — total %d steps, %.1fs", step, time.time() - t_start)
    return 0


def _save_ckpt(model, out_dir: Path, step: int, metrics: list) -> None:
    torch = _lazy_torch()
    ckpt_path = out_dir / f"vqvae_step{step}.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "cfg": asdict(model.cfg),
    }, ckpt_path)
    # Pull just the codebook tensor — what GeoRelVLA(depth_codebook=...) wants.
    codebook = model.quantizer.codebook.weight.detach().cpu().clone()
    torch.save(codebook, out_dir / "codebook.pt")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
