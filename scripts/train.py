#!/usr/bin/env python
"""GeoRel-VLA main training script.

Phase 1.7c.1 — depth-side co-training. Reads `.npz` shards from
`scripts/extract_libero_depth_gt.py`, encodes their depth maps through a
**pretrained, frozen** VQ-VAE to get target indices, runs
`GeoRelVLA(rgb, depth_target_indices)`, and steps AdamW on the resulting
`λ_t · L_depth`. The action loss is left as a TODO until Pi0Backbone
.forward_action() is wired in 1.7c.2.

Typical invocation:

    python scripts/train.py \\
        --data-dir /autodl-fs/data/svla/data/libero_depth_gt/libero_spatial \\
        --vqvae-ckpt /autodl-fs/data/svla/runs/vqvae_depth_phase1/vqvae_stepN.pt \\
        --out-dir /autodl-fs/data/svla/runs/georel_vla_phase1 \\
        --backbone stub \\
        --epochs 1 --batch 8

For the full QDepth-VLA recipe replace `--backbone stub` with `pi0` once
Phase 1.7c.2 wires Pi0Backbone.forward_action().
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


class _LiberoShardDataset:
    """Iterate (rgb, depth) frames from `extract_libero_depth_gt.py` shards.

    Re-implements just the streaming bit so training can run without spinning
    up torch.utils.data DataLoader workers (which on Windows with the wrong
    spawn semantics tend to fight MuJoCo). For Phase 1.7c.1 a single-process
    Python generator is plenty.
    """

    def __init__(
        self,
        data_dir: Path,
        depth_clip_m: float = 5.0,
        shuffle_shards: bool = True,
        seed: int = 0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.depth_clip_m = depth_clip_m
        self.shuffle_shards = shuffle_shards
        self.rng = np.random.default_rng(seed)

        self.shards = sorted(self.data_dir.glob("*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no .npz shards under {self.data_dir}")

    def stream(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        order = list(range(len(self.shards)))
        if self.shuffle_shards:
            self.rng.shuffle(order)
        for idx in order:
            shard = np.load(self.shards[idx], allow_pickle=True)
            rgb = shard["rgb"]                                  # (T, H, W, 3) uint8
            depth = shard["depth"].astype(np.float32)            # (T, H, W) float16/32
            np.clip(depth, 0.0, self.depth_clip_m, out=depth)
            for t in range(rgb.shape[0]):
                yield rgb[t], depth[t]

    def batches(self, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        rgb_buf: list[np.ndarray] = []
        depth_buf: list[np.ndarray] = []
        for rgb, depth in self.stream():
            rgb_buf.append(rgb)
            depth_buf.append(depth)
            if len(rgb_buf) >= batch_size:
                yield np.stack(rgb_buf, 0), np.stack(depth_buf, 0)
                rgb_buf.clear()
                depth_buf.clear()
        if rgb_buf:
            yield np.stack(rgb_buf, 0), np.stack(depth_buf, 0)


# ----- training ----------------------------------------------------------


def _to_rgb_tensor(rgb_np: np.ndarray, target_size: int = 224, device: str = "cuda"):
    """uint8 (B, H, W, 3) -> float (B, 3, target_size, target_size) on `device`."""
    torch = _lazy_torch()
    import torch.nn.functional as F

    x = torch.from_numpy(rgb_np).to(device).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != target_size:
        x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return x


def _build_backbone(name: str, device: str):
    if name == "stub":
        from georel_vla.backbones.stub import StubBackbone, StubBackboneConfig
        return StubBackbone(StubBackboneConfig()).to(device)
    if name == "pi0":
        from georel_vla.backbones.pi0 import Pi0Backbone, Pi0BackboneConfig
        bk = Pi0Backbone(Pi0BackboneConfig(device=device))
        bk.load()
        return bk
    raise ValueError(f"unknown backbone {name!r}; expected one of [stub, pi0]")


def _load_vqvae(ckpt_path: Path, device: str):
    torch = _lazy_torch()
    from georel_vla.codebooks.vqvae import VQVAE, VQVAEConfig
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = VQVAEConfig(**state["cfg"]) if isinstance(state.get("cfg"), dict) else VQVAEConfig()
    vq = VQVAE(cfg).to(device)
    vq.load_state_dict(state["model_state_dict"])
    vq.eval()
    for p in vq.parameters():
        p.requires_grad = False
    return vq, cfg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--vqvae-ckpt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--backbone", choices=["stub", "pi0"], default="stub")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lambda-depth", type=float, default=0.01)
    p.add_argument("--lambda-gamma", type=float, default=0.9999,
                   help="QDepth-VLA λ_t = λ_0 · γ^step decay")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every-steps", type=int, default=10)
    p.add_argument("--save-every-steps", type=int, default=2000)
    p.add_argument("--max-batches-per-epoch", type=int, default=0,
                   help="Cap iterations per epoch (0 = unlimited; useful for smoke runs)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("train")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch = _lazy_torch()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log.info("loading frozen VQ-VAE codebook from %s", args.vqvae_ckpt)
    vq, vq_cfg = _load_vqvae(args.vqvae_ckpt, args.device)
    codebook = vq.quantizer.codebook.weight.detach().clone()
    log.info("codebook: K=%d dim=%d (frozen)", codebook.shape[0], codebook.shape[1])

    log.info("building backbone=%s on %s", args.backbone, args.device)
    backbone = _build_backbone(args.backbone, args.device)

    from georel_vla.experts.depth_expert import DepthExpert, DepthExpertConfig
    from georel_vla.model import GeoRelVLA, GeoRelVLAConfig

    cfg = GeoRelVLAConfig(
        depth_codebook_size=vq_cfg.num_embeddings,
        depth_embedding_dim=vq_cfg.embedding_dim,
        depth_latent_h=256 // vq_cfg.downsample,
        depth_latent_w=256 // vq_cfg.downsample,
        lambda_depth=args.lambda_depth,
    )
    expert = DepthExpert(DepthExpertConfig(
        siglip_dim=backbone.siglip_dim, n_img_tokens=backbone.n_image_tokens,
        latent_h=cfg.depth_latent_h, latent_w=cfg.depth_latent_w,
        embedding_dim=cfg.depth_embedding_dim, num_embeddings=cfg.depth_codebook_size,
    )).to(args.device)

    model = GeoRelVLA(cfg, backbone=backbone, depth_expert=expert,
                     depth_codebook=codebook.to(args.device))
    log.info("GeoRelVLA constructed: %s", repr(model))

    # Optimiser sees only trainable params (backbone may have frozen pieces in Phase 1.7c.2)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    ds = _LiberoShardDataset(args.data_dir, seed=args.seed)
    metrics = []
    step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        ep_start = time.time()
        ep_batches = 0
        for rgb_np, depth_np in ds.batches(args.batch):
            rgb = _to_rgb_tensor(rgb_np, args.image_size, args.device)
            # depth: encode on GPU through frozen VQ-VAE -> per-spatial indices
            depth = torch.from_numpy(depth_np).to(args.device).unsqueeze(1)   # (B, 1, H, W)
            with torch.no_grad():
                indices = vq.encode_indices(depth).reshape(depth.size(0), -1)  # (B, N_lat)

            out = model(rgb, depth_target_indices=indices)
            # compute_losses applies λ·γ^step decay internally
            losses = model.compute_losses(
                depth_logits=out["depth_logits"],
                depth_target_indices=indices,
                step=step,
                gamma=args.lambda_gamma,
            )
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            step += 1
            ep_batches += 1

            if step % args.log_every_steps == 0:
                row = {
                    "epoch": epoch, "step": step,
                    "loss_total": float(losses["total"].item()),
                    "loss_depth": float(losses["depth"].item()),
                    "lambda_t": float(losses["lambda_depth_t"].item()),
                    "elapsed_sec": time.time() - t_start,
                }
                metrics.append(row)
                log.info(
                    "ep=%d step=%d total=%.4f depth=%.4f lambda_t=%.4g t=%.0fs",
                    epoch, step, row["loss_total"], row["loss_depth"],
                    row["lambda_t"], row["elapsed_sec"],
                )

            if args.save_every_steps and step % args.save_every_steps == 0:
                _save_ckpt(model, args.out_dir, step, metrics, cfg)

            if args.max_batches_per_epoch and ep_batches >= args.max_batches_per_epoch:
                break

        log.info("epoch %d done %d batches %.1fs", epoch, ep_batches, time.time() - ep_start)

    _save_ckpt(model, args.out_dir, step, metrics, cfg)
    log.info("DONE %d steps %.1fs", step, time.time() - t_start)
    return 0


def _save_ckpt(model, out_dir: Path, step: int, metrics: list, cfg) -> None:
    torch = _lazy_torch()
    ckpt = out_dir / f"georel_vla_step{step}.pt"
    # Save only the trainable parts + the buffered codebook so we can re-load
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "cfg": asdict(cfg),
    }, ckpt)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
