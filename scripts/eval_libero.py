#!/usr/bin/env python
"""LIBERO evaluation — Phase 1.8 / Phase-1 gate.

Runs the policy under test through one or more LIBERO suites/tasks and
reports the success rate. Two policy modes:

* `--policy openvla`  — OpenVLA-7B-finetuned-<suite> baseline; same path as
  PSSA's `run_libero_eval.py` smoke. Used for Phase-0 reproducibility and as
  the "no auxiliary supervision" reference number.
* `--policy georel`    — our GeoRel-VLA checkpoint (Phase 1.7c.2 onwards).
  Until Pi0Backbone.forward_action() is wired, this raises a clear error.

Common flags mirror PSSA's run_libero_eval.py for drop-in comparability:
`--libero-action-fix` (gripper sign + flip) and `--libero-image-fix` (180°
rotate to match OpenVLA-finetune training distribution).

Output: one `metrics.json` per (suite, task) pair under `--out-dir`,
following the same schema PSSA used so existing analysis scripts keep
working.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def _set_egl_env() -> None:
    """Off-screen MuJoCo render needs EGL on headless GPU nodes."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")


_PROPRIO_DIMS_FALLBACK = (3, 4, 1)  # eef pos + eef quat + gripper qpos = 8 dims


# ----- LIBERO env setup --------------------------------------------------


def _make_env(suite: str, task_id: int, resolution: int = 224, with_depth: bool = False):
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = get_benchmark(suite)()
    task = bench.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
        camera_depths=with_depth,
        camera_names=["agentview"],
    )
    init_states = bench.get_task_init_states(task_id)
    return env, task, init_states


# ----- policy adapters ---------------------------------------------------


def _build_policy(name: str, model_id: str, unnorm_key: str, device: str = "cuda") -> Callable:
    """Return a `policy(obs_rgb, language) -> action_np` callable."""
    if name == "openvla":
        from transformers import AutoModelForVision2Seq, AutoProcessor
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="bfloat16",
        ).to(device)

        def policy(obs_rgb: np.ndarray, language: str) -> np.ndarray:
            from PIL import Image
            img = Image.fromarray(obs_rgb)
            prompt = f"In: What action should the robot take to {language.strip().lower()}?\nOut:"
            inputs = proc(prompt, img).to(device, dtype="bfloat16")
            action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            return np.asarray(action, dtype=np.float32)

        return policy

    if name == "georel":
        import torch

        from georel_vla.backbones.pi0 import Pi0Backbone, Pi0BackboneConfig
        bk = Pi0Backbone(Pi0BackboneConfig(device=device, dtype="bf16", load_paligemma=True))
        bk.load()
        # Optionally load a GeoRelVLA fine-tuned ckpt over the freshly-loaded
        # PaliGemma weights. For Phase 1 this is the train.py output; if no
        # path passed, eval runs on the PaliGemma-init backbone (sanity baseline).
        if model_id and Path(model_id).is_file():
            state = torch.load(model_id, map_location=device, weights_only=False)
            bk.pizero.load_state_dict(state["model_state_dict"], strict=False)

        def policy(obs_rgb: np.ndarray, language: str) -> np.ndarray:
            rgb_u8 = torch.from_numpy(obs_rgb).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
            # NOTE: caller must arrange to pass full obs (not just rgb) when using georel;
            # for now, construct zero proprio fallback. Phase 1.7d wires real obs.
            proprios = torch.zeros(1, 1, sum(_PROPRIO_DIMS_FALLBACK), device=device)
            actions = bk.infer_action(rgb_u8, [language], proprios)  # (1, horizon, action_dim)
            return actions[0, 0].cpu().float().numpy()

        return policy

    raise ValueError(f"unknown --policy {name!r}; expected openvla | georel")


def _libero_image_fix(rgb: np.ndarray) -> np.ndarray:
    """PSSA documented: 180° rotate to match OpenVLA-finetune training distribution."""
    return rgb[::-1, ::-1].copy()


def _libero_action_fix(action: np.ndarray) -> np.ndarray:
    """PSSA documented: gripper [0,1]→sign({-1,+1}) + flip sign (OpenVLA 1=close vs LIBERO 1=open)."""
    a = action.copy()
    g = a[-1]
    a[-1] = -1.0 if g > 0.5 else 1.0
    return a


# ----- eval loop ---------------------------------------------------------


def evaluate_task(
    suite: str,
    task_id: int,
    policy: Callable,
    rollouts: int = 50,
    max_steps: int = 200,
    resolution: int = 224,
    apply_image_fix: bool = True,
    apply_action_fix: bool = True,
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    log = log or logging.getLogger("eval_libero")
    env, task, init_states = _make_env(suite, task_id, resolution=resolution)
    log.info("==> bench %s task %d  name=%s", suite, task_id, task.name)
    n_init = len(init_states)
    rollouts = min(rollouts, n_init)

    rollout_records: list[dict[str, Any]] = []
    n_success = 0
    for r in range(rollouts):
        env.reset()
        env.set_init_state(init_states[r % n_init])
        success = False
        steps = 0
        step_times = []
        peak_vram = 0.0
        for steps in range(max_steps):
            obs, *_ = env.step(np.zeros(7, dtype=np.float32))  # initial settle
            if steps > 0:
                break
        for steps in range(1, max_steps + 1):
            rgb = obs["agentview_image"]
            if apply_image_fix:
                rgb = _libero_image_fix(rgb)
            t0 = time.time()
            action = policy(rgb, task.language)
            step_times.append((time.time() - t0) * 1000.0)
            if apply_action_fix:
                action = _libero_action_fix(action)
            obs, _, done, info = env.step(action)
            if isinstance(done, bool) and done:
                success = True
                break
            if isinstance(info, dict) and info.get("success"):
                success = True
                break
            try:
                import torch
                if torch.cuda.is_available():
                    peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e9)
            except ImportError:
                pass

        rollout_records.append({
            "rollout": r,
            "success": bool(success),
            "steps": int(steps),
            "step_ms_avg": float(np.mean(step_times)) if step_times else 0.0,
            "step_ms_p95": float(np.percentile(step_times, 95)) if step_times else 0.0,
            "peak_vram_gb": float(peak_vram),
        })
        n_success += int(success)
        log.info("==> rollout %d -> success=%s steps=%d", r, success, steps)
    env.close()

    return {
        "suite": suite, "task_id": int(task_id), "task_name": task.name,
        "task_language": task.language, "n_rollouts": rollouts, "n_success": n_success,
        "success_rate": n_success / rollouts if rollouts else 0.0,
        "rollouts": rollout_records,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="libero_spatial",
                   choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    p.add_argument("--task-ids", default="0-9", help='"0-9" or "0,3,5"')
    p.add_argument("--rollouts", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--policy", choices=["openvla", "georel"], default="openvla")
    p.add_argument("--model-id", default="openvla/openvla-7b-finetuned-libero-spatial",
                   help="HF model id (openvla policy) or path to GeoRelVLA ckpt (georel policy)")
    p.add_argument("--unnorm-key", default=None,
                   help="OpenVLA unnorm key; defaults to suite name (libero_spatial / libero_10 / ...)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--libero-action-fix", action="store_true", default=True)
    p.add_argument("--libero-image-fix", action="store_true", default=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("eval_libero")
    _set_egl_env()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # parse task ids
    out: list[int] = []
    for part in args.task_ids.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    task_ids = sorted(set(out))
    log.info("running %d task(s) in %s: %s", len(task_ids), args.suite, task_ids)

    unnorm_key = args.unnorm_key or args.suite
    log.info("policy=%s model=%s unnorm_key=%s", args.policy, args.model_id, unnorm_key)
    policy = _build_policy(args.policy, args.model_id, unnorm_key, device=args.device)

    summary = []
    for tid in task_ids:
        metrics = evaluate_task(
            args.suite, tid, policy,
            rollouts=args.rollouts, max_steps=args.max_steps, resolution=args.resolution,
            apply_image_fix=args.libero_image_fix, apply_action_fix=args.libero_action_fix,
            log=log,
        )
        out_path = args.out_dir / f"task{tid:02d}_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log.info("wrote %s SR=%d/%d (%.1f%%)",
                 out_path.name, metrics["n_success"], metrics["n_rollouts"],
                 100 * metrics["success_rate"])
        summary.append({"task_id": tid, "task_name": metrics["task_name"],
                        "n_success": metrics["n_success"], "n_rollouts": metrics["n_rollouts"],
                        "success_rate": metrics["success_rate"]})

    overall = {
        "suite": args.suite, "policy": args.policy, "model_id": args.model_id,
        "n_tasks": len(task_ids),
        "overall_success_rate": (
            sum(s["n_success"] for s in summary)
            / max(1, sum(s["n_rollouts"] for s in summary))
        ),
        "per_task": summary,
    }
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    log.info(
        "DONE — overall SR %.3f over %d tasks (-> %s)",
        overall["overall_success_rate"], len(task_ids), args.out_dir / "summary.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
