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
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _lazy_torch():
    import torch  # noqa: PLC0415
    return torch


def _ddp_env() -> tuple[int, int, int]:
    """Read torchrun env (RANK / LOCAL_RANK / WORLD_SIZE). Returns (rank, local_rank, world_size)."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _ddp_init(local_rank: int, world_size: int):
    """Initialise torch.distributed if world_size > 1. Returns torch device."""
    torch = _lazy_torch()
    if world_size > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return None


# ----- dataset -----------------------------------------------------------


class _LiberoShardDataset:
    """Iterate (rgb, depth, action, proprio, language) frames from shards.

    Re-implements just the streaming bit so training can run without spinning
    up torch.utils.data DataLoader workers (which on Windows with the wrong
    spawn semantics tend to fight MuJoCo).

    `proprio` and `language` are absent on shards written before extractor v3
    (Phase 1.7c.2 bumped that). When absent, `proprio` defaults to zeros and
    `language` defaults to '' so the dataset survives mixed-vintage shards.
    """

    def __init__(
        self,
        data_dir: Path,
        depth_clip_m: float = 5.0,
        shuffle_shards: bool = True,
        seed: int = 0,
        proprio_dim_fallback: int = 9,   # 3 (eef pos) + 4 (eef quat) + 2 (parallel-jaw gripper qpos)
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.depth_clip_m = depth_clip_m
        self.shuffle_shards = shuffle_shards
        self.rng = np.random.default_rng(seed + rank)  # per-rank stream order
        self.proprio_dim_fallback = proprio_dim_fallback
        self.rank = rank
        self.world_size = world_size

        all_shards = sorted(self.data_dir.glob("*.npz"))
        if not all_shards:
            raise FileNotFoundError(f"no .npz shards under {self.data_dir}")
        # Per-rank shard subset for DDP: deterministic round-robin assignment.
        self.shards = all_shards[rank::world_size] if world_size > 1 else all_shards
        if world_size > 1 and not self.shards:
            raise RuntimeError(f"rank {rank}/{world_size} ended up with 0 shards "
                               f"(total shards={len(all_shards)}); reduce world_size")

    def stream(self):
        """Yield per-frame (rgb, depth, action, proprio, language) tuples.

        Corrupt / partially-written shards (e.g. interrupted extraction that
        left a truncated zip) raise inside np.load. We log+skip them instead
        of crashing the whole training run — losing one shard's frames is
        cheaper than a 20-min restart.
        """
        import json as _json
        import logging as _logging  # noqa: PLC0415
        _log = _logging.getLogger("train")
        order = list(range(len(self.shards)))
        if self.shuffle_shards:
            self.rng.shuffle(order)
        for idx in order:
            shard_path = self.shards[idx]
            try:
                shard = np.load(shard_path, allow_pickle=True)
                rgb = shard["rgb"]                               # (T, H, W, 3) uint8
                depth = shard["depth"].astype(np.float32)        # (T, H, W) float16/32
            except (EOFError, OSError, ValueError, KeyError) as exc:
                _log.warning("skipping corrupt shard %s: %s", shard_path.name, exc)
                continue
            np.clip(depth, 0.0, self.depth_clip_m, out=depth)
            action = shard["action"].astype(np.float32) if "action" in shard.files else np.zeros((rgb.shape[0], 7), np.float32)
            if "proprio" in shard.files:
                proprio = shard["proprio"].astype(np.float32)
            else:
                proprio = np.zeros((rgb.shape[0], self.proprio_dim_fallback), np.float32)
            try:
                lang = _json.loads(str(shard["meta"])).get("language", "") if "meta" in shard.files else ""
            except Exception:
                lang = ""
            for t in range(rgb.shape[0]):
                yield rgb[t], depth[t], action[t], proprio[t], lang

    def batches(self, batch_size: int):
        rgb_buf: list = []
        depth_buf: list = []
        action_buf: list = []
        proprio_buf: list = []
        lang_buf: list = []
        for rgb, depth, action, proprio, lang in self.stream():
            rgb_buf.append(rgb)
            depth_buf.append(depth)
            action_buf.append(action)
            proprio_buf.append(proprio)
            lang_buf.append(lang)
            if len(rgb_buf) >= batch_size:
                yield (np.stack(rgb_buf, 0), np.stack(depth_buf, 0),
                       np.stack(action_buf, 0), np.stack(proprio_buf, 0), list(lang_buf))
                rgb_buf.clear()
                depth_buf.clear()
                action_buf.clear()
                proprio_buf.clear()
                lang_buf.clear()
        if rgb_buf:
            yield (np.stack(rgb_buf, 0), np.stack(depth_buf, 0),
                   np.stack(action_buf, 0), np.stack(proprio_buf, 0), list(lang_buf))


# ----- training ----------------------------------------------------------


def _to_rgb_tensor(rgb_np: np.ndarray, target_size: int = 224, device: str = "cuda"):
    """uint8 (B, H, W, 3) -> float (B, 3, target_size, target_size) on `device`."""
    torch = _lazy_torch()
    import torch.nn.functional as F

    x = torch.from_numpy(rgb_np).to(device).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != target_size:
        x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return x


def _build_backbone(name: str, device: str, action_expert_ckpt: Path | None = None):
    if name == "stub":
        from georel_vla.backbones.stub import StubBackbone, StubBackboneConfig
        return StubBackbone(StubBackboneConfig()).to(device)
    if name == "pi0":
        from georel_vla.backbones.pi0 import Pi0Backbone, Pi0BackboneConfig
        bk = Pi0Backbone(Pi0BackboneConfig(
            device=device,
            action_expert_ckpt=action_expert_ckpt,
        ))
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
    # Stage-B recipe fixes (added after Phase-1.7c-v1 produced clean loss but 0/30 SR).
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Accumulate gradients over this many micro-batches before optimizer.step (effective batch = batch * grad-accum-steps)")
    p.add_argument("--clip-grad-norm", type=float, default=1.0,
                   help="Max grad norm for clip_grad_norm_; 0 disables")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear LR warmup steps; 0 disables")
    p.add_argument("--lr-min", type=float, default=1e-8,
                   help="Cosine LR floor (eta_min)")
    p.add_argument("--flow-sampling", choices=["uniform", "beta"], default="beta",
                   help="Flow-matching time sampler; Pi0 paper recommends beta")
    p.add_argument("--clamp-actions", action="store_true", default=True,
                   help="Clamp training actions to [-1, 1] to match PiZero inference clip")
    p.add_argument("--action-expert-ckpt", type=Path, default=None,
                   help="Path to open-pi-zero published ckpt (e.g. bridge_beta_step19296.pt) for warmstart")
    args = p.parse_args()

    # DDP setup (torchrun-style). Single-GPU runs are unaffected (world_size=1).
    rank, local_rank, world_size = _ddp_env()
    is_main = rank == 0
    if world_size > 1:
        args.device = f"cuda:{local_rank}"
        _ddp_init(local_rank, world_size)

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"%(asctime)s [%(levelname)s] %(name)s [r{rank}/{world_size}]: %(message)s",
    )
    log = logging.getLogger("train")

    if is_main:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        import torch as _t  # noqa: PLC0415
        _t.distributed.barrier()
    torch = _lazy_torch()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    log.info("loading frozen VQ-VAE codebook from %s", args.vqvae_ckpt)
    vq, vq_cfg = _load_vqvae(args.vqvae_ckpt, args.device)
    codebook = vq.quantizer.codebook.weight.detach().clone()
    log.info("codebook: K=%d dim=%d (frozen)", codebook.shape[0], codebook.shape[1])

    log.info("building backbone=%s on %s", args.backbone, args.device)
    backbone = _build_backbone(args.backbone, args.device,
                               action_expert_ckpt=args.action_expert_ckpt)

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

    # DDP wrap.
    train_module = model
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: PLC0415
        # static_graph=True is required because the training loop runs TWO forward
        # passes per iteration (depth head via GeoRelVLA.forward, then action loss via
        # Pi0Backbone.forward_action_loss called directly through train_module.backbone).
        # The default DDP autograd hook fires once per parameter; with two forwards
        # sharing the encoder, some params see two backward signals and DDP raises
        # "marked as ready twice". static_graph lets DDP trace the full multi-forward
        # graph on iter 0 and reuse it. find_unused_parameters=False because static_graph
        # already handles unused-param detection.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    static_graph=True, broadcast_buffers=False)
        train_module = model.module

    # Optimiser sees only trainable params (backbone may have frozen pieces in Phase 1.7c.2)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    # LR schedule: linear warmup over `warmup_steps` then cosine decay to lr_min.
    # Total steps estimated from dataset size + grad accum (refined on first epoch).
    import math  # noqa: PLC0415

    ds = _LiberoShardDataset(args.data_dir, seed=args.seed,
                              rank=rank, world_size=world_size)

    # Pre-flight: count batches in epoch 0 to size the cosine schedule total_steps
    # without rerunning the iterator (peek the npz files for total frame count).
    total_frames = 0
    for shard_path in ds.shards:
        try:
            total_frames += int(np.load(shard_path, allow_pickle=True)["rgb"].shape[0])
        except Exception:
            total_frames += 100  # fallback estimate
    batches_per_epoch = max(1, total_frames // args.batch)
    optim_steps_per_epoch = max(1, batches_per_epoch // max(1, args.grad_accum_steps))
    total_optim_steps = optim_steps_per_epoch * args.epochs
    log.info("estimated %d frames / %d batches/epoch / %d optim steps total (grad_accum=%d)",
             total_frames, batches_per_epoch, total_optim_steps, args.grad_accum_steps)

    def get_lr_mult(optim_step: int) -> float:
        if args.warmup_steps and optim_step < args.warmup_steps:
            return (optim_step + 1) / args.warmup_steps
        progress = (optim_step - args.warmup_steps) / max(1, total_optim_steps - args.warmup_steps)
        progress = max(0.0, min(1.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (args.lr_min / args.lr) + cos * (1.0 - args.lr_min / args.lr)

    # Beta flow-matching time sampler per Pi0 paper (alpha=1.5, beta=1, flip+shift).
    flow_beta_dist = torch.distributions.Beta(
        torch.tensor(1.5, device=args.device), torch.tensor(1.0, device=args.device),
    ) if args.flow_sampling == "beta" else None
    flow_t_max = 1.0 - 1e-3

    metrics = []
    step = 0           # batch counter (incremented every micro-batch)
    optim_step = 0     # optimizer step counter (incremented every grad-accum boundary)
    t_start = time.time()
    opt.zero_grad()
    for epoch in range(args.epochs):
        ep_start = time.time()
        ep_batches = 0
        for rgb_np, depth_np, action_np, proprio_np, lang_list in ds.batches(args.batch):
            rgb = _to_rgb_tensor(rgb_np, args.image_size, args.device)
            depth = torch.from_numpy(depth_np).to(args.device).unsqueeze(1)   # (B, 1, H, W)
            with torch.no_grad():
                indices = vq.encode_indices(depth).reshape(depth.size(0), -1)  # (B, N_lat)

            out = model(rgb, depth_target_indices=indices)

            # depth-side compute_losses always; for pi0 backbone we then add the
            # CFM action loss DIRECTLY (forward_action_loss already returns a scalar
            # loss; passing it through compute_losses' MSE would square it).
            losses = train_module.compute_losses(
                depth_logits=out["depth_logits"],
                depth_target_indices=indices,
                step=step, gamma=args.lambda_gamma,
            )
            if args.backbone == "pi0":
                import torch.nn.functional as Fnn  # noqa: PLC0415
                B = rgb.size(0)
                rgb_u8_native = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).contiguous().to(args.device)
                if rgb_u8_native.shape[-1] != args.image_size:
                    rgb_u8 = Fnn.interpolate(
                        rgb_u8_native.float(), size=(args.image_size, args.image_size),
                        mode="bilinear", align_corners=False,
                    ).clamp(0, 255).to(torch.uint8)
                else:
                    rgb_u8 = rgb_u8_native
                proprios = torch.from_numpy(proprio_np).unsqueeze(1).to(args.device)
                actions = torch.from_numpy(action_np).unsqueeze(1).to(args.device)
                if args.clamp_actions:
                    actions = torch.clamp(actions, -1.0, 1.0)
                horizon = train_module.backbone.pizero.horizon_steps if hasattr(train_module.backbone, "pizero") else 4
                if actions.size(1) != horizon:
                    actions = actions.expand(B, horizon, actions.size(-1)).contiguous()
                if flow_beta_dist is not None:
                    z = flow_beta_dist.sample((B,))
                    t_fm = flow_t_max * (1.0 - z)
                else:
                    t_fm = torch.rand(B, device=args.device) * flow_t_max
                action_loss = train_module.backbone.forward_action_loss(
                    rgb_u8, lang_list, proprios, actions, t_fm,
                )
                losses["action"] = action_loss
                losses["total"] = losses["total"] + action_loss

            # Gradient accumulation: scale loss by 1/N and step optimizer every N micro-batches.
            scaled_loss = losses["total"] / max(1, args.grad_accum_steps)
            scaled_loss.backward()
            step += 1
            ep_batches += 1
            if step % max(1, args.grad_accum_steps) == 0:
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.clip_grad_norm,
                    )
                # Apply LR schedule before step()
                lr_mult = get_lr_mult(optim_step)
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * lr_mult
                opt.step()
                opt.zero_grad()
                optim_step += 1

            if step % args.log_every_steps == 0:
                row = {
                    "epoch": epoch, "step": step, "optim_step": optim_step,
                    "lr": opt.param_groups[0]["lr"],
                    "loss_total": float(losses["total"].item()),
                    "loss_depth": float(losses["depth"].item()),
                    "lambda_t": float(losses["lambda_depth_t"].item()),
                    "elapsed_sec": time.time() - t_start,
                }
                if "action" in losses:
                    row["loss_action"] = float(losses["action"].item())
                metrics.append(row)
                log.info(
                    "ep=%d step=%d opt=%d lr=%.2e total=%.4f depth=%.4f action=%s lambda_t=%.4g t=%.0fs",
                    epoch, step, optim_step, row["lr"],
                    row["loss_total"], row["loss_depth"],
                    f"{row['loss_action']:.4f}" if "loss_action" in row else "n/a",
                    row["lambda_t"], row["elapsed_sec"],
                )

            if args.save_every_steps and step % args.save_every_steps == 0 and is_main:
                _save_ckpt(train_module, args.out_dir, step, metrics, cfg)

            if args.max_batches_per_epoch and ep_batches >= args.max_batches_per_epoch:
                break

        log.info("epoch %d done %d batches %.1fs", epoch, ep_batches, time.time() - ep_start)

    if is_main:
        _save_ckpt(train_module, args.out_dir, step, metrics, cfg)
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
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
