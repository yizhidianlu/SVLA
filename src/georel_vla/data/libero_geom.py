"""LIBERO simulator-GT depth (and RGB / proprio / action) extractor.

For Phase 1 VQ-VAE pretraining we need clean depth supervision aligned with
the RGB the policy sees. LIBERO is built on Robosuite + MuJoCo, so depth is
directly queryable from the renderer — no monocular estimator needed (as
QDepth-VLA's ViDA path on OXE was forced to).

This module is the *library* — `LiberoDepthExtractor.replay_demo()` returns
(rgb, depth, action, ...) frame by frame; `extract_to_npz()` writes one
compressed `.npz` per demo, layout:

    <out_dir>/<suite>/task<NN>_<demo_key>.npz   ->
        rgb     : (T, H, W, 3) uint8
        depth   : (T, H, W)    float16 in [0, 1]   (Robosuite's normalised z-buffer)
        action  : (T, action_dim) float32
        meta    : task_name, language, suite, task_id, H, W, camera, action_fix_applied (json)

NB: the 180-degree image flip PSSA documented as `--libero-image-fix` is
**NOT** applied here. We save the raw orientation that Robosuite emits and
leave the flip to the training/eval pipeline (so the saved files are
agnostic to which backbone consumes them).

Imports of `libero` / `robosuite` / `h5py` are deferred to method scope so
this module is importable on machines that do not have MuJoCo (e.g., the CI
Ubuntu runner without GPU).

The companion CLI lives at `scripts/extract_libero_depth_gt.py`.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

# --- defaults sourced from the smoke run + PSSA workspace conventions -------

DEFAULT_LIBERO_ROOT = Path("/root/autodl-tmp/LIBERO")
DEFAULT_DEMOS_ROOT = Path("/root/autodl-tmp/datasets")
DEFAULT_CAMERA = "agentview"
DEFAULT_RESOLUTION = 256                  # LIBERO native render size; resize at training time
DEFAULT_OUT_ROOT = Path("/autodl-fs/data/svla/data/libero_depth_gt")

# Robosuite emits depth in [0,1] as normalised z-buffer; convert to metric on
# the fly only if needed (the VQ-VAE doesn't care as long as we're consistent).


@dataclass
class LiberoExtractorConfig:
    libero_root: Path = DEFAULT_LIBERO_ROOT
    demos_root: Path = DEFAULT_DEMOS_ROOT
    out_root: Path = DEFAULT_OUT_ROOT
    camera: str = DEFAULT_CAMERA
    resolution: int = DEFAULT_RESOLUTION
    max_steps_per_demo: int = 400         # safety cap: LIBERO-Long demos can run ~300+ steps
    max_demos_per_task: int | None = None  # None = all demos in the .hdf5
    stride: int = 1                       # keep every `stride`-th frame; 1 = keep all
    save_action: bool = True
    save_proprio: bool = True             # eef pos+quat+gripper for Phase 1.7c action loss
    proprio_keys: tuple[str, ...] = (     # standard LIBERO obs names; concatenated as proprio
        "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
    )
    compress: bool = True                 # use np.savez_compressed
    skip_existing: bool = True
    seed: int = 0
    # Convert MuJoCo's normalised z-buffer (non-linear, [0,1], dominated by far plane)
    # to metric depth (meters) via robosuite.utils.camera_utils.get_real_depth_map.
    # Without this the VQ-VAE codebook collapses because all values cluster
    # tightly around 1.0 right at the far plane.
    metric_depth: bool = True
    depth_clip_m: float = 5.0             # clip metric depth to [0, depth_clip_m] for fp16 storage
    extra_metadata: dict = field(default_factory=dict)


@dataclass
class LiberoFrame:
    rgb: np.ndarray          # (H, W, 3) uint8
    depth: np.ndarray        # (H, W) float32 — metric meters by default; raw z-buffer if metric_depth=False
    action: np.ndarray       # (action_dim,) float32
    proprio: np.ndarray | None  # (proprio_dim,) float32 if cfg.save_proprio else None
    step: int


class LiberoDepthExtractor:
    """Replay LIBERO demos and yield / save (rgb, depth, action) frames.

    Usage:

        extractor = LiberoDepthExtractor(LiberoExtractorConfig())
        for npz_path in extractor.extract_suite("libero_spatial", task_ids=range(10)):
            print("wrote", npz_path)

    or for in-memory frame iteration without disk IO:

        for frame in extractor.replay_demo("libero_spatial", task_id=0, demo_idx=0):
            ...
    """

    def __init__(self, cfg: LiberoExtractorConfig | None = None) -> None:
        self.cfg = cfg or LiberoExtractorConfig()
        # set EGL env vars so off-screen rendering works on headless GPU nodes
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # --- public API -------------------------------------------------------

    def extract_suite(
        self,
        suite: str,
        task_ids: list[int] | range | None = None,
    ) -> Iterator[Path]:
        """Yield npz file path for every demo of every requested task in `suite`."""
        bench = self._load_benchmark(suite)
        all_task_ids = range(bench.n_tasks) if task_ids is None else list(task_ids)
        for tid in all_task_ids:
            yield from self.extract_task(suite, tid)

    def extract_task(self, suite: str, task_id: int) -> Iterator[Path]:
        """Yield npz path for every demo in `<suite>/<task>_demo.hdf5`.

        PSSA-documented infra fix: instantiate the Robosuite env **once per
        task** and reset between demos instead of recreating per demo. The
        per-demo `OffScreenRenderEnv` recreation leaks EGL render contexts
        (each new render context = 1+ open fds) and the process hangs ~10
        demos in when the EGL pool / fd table is exhausted.
        """
        import h5py

        bench = self._load_benchmark(suite)
        task = bench.get_task(task_id)
        demo_path = self._demo_path(suite, task)
        if not demo_path.is_file():
            raise FileNotFoundError(f"demo file not found: {demo_path}")

        out_dir = self.cfg.out_root / suite
        out_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(demo_path, "r") as f:
            demo_keys = sorted(f["data"].keys())
            if self.cfg.max_demos_per_task is not None:
                demo_keys = demo_keys[: self.cfg.max_demos_per_task]

            init_states = bench.get_task_init_states(task_id)

            # Decide first if there is any work left, to skip the (slow)
            # env construction when every demo is already on disk.
            todo = [
                k for k in demo_keys
                if not (self.cfg.skip_existing and (out_dir / f"task{task_id:02d}_{k}.npz").is_file())
            ]
            for k in demo_keys:
                if k not in todo:
                    yield out_dir / f"task{task_id:02d}_{k}.npz"
            if not todo:
                return

            env = self._make_env(suite, task)
            env.reset()  # ONE-TIME full reset; per-demo we only set_init_state below.
            try:
                # Recreate env every RECREATE_EVERY demos to bound robosuite's
                # deepcopy slowdown (model state accumulates across resets — by
                # demo ~25 a single `_load_model` deepcopy hangs >5 min, so we
                # never let the same env survive that long).
                RECREATE_EVERY = 20
                demos_since_recreate = 0
                for demo_key in todo:
                    demo_idx = demo_keys.index(demo_key)
                    out_path = out_dir / f"task{task_id:02d}_{demo_key}.npz"

                    if demos_since_recreate >= RECREATE_EVERY:
                        try:
                            env.close()
                        except Exception as exc:
                            log.warning("env.close() before recreate raised %s", exc)
                        env = self._make_env(suite, task)
                        env.reset()
                        demos_since_recreate = 0

                    frames = list(self._replay_one_demo_in_env(
                        env, task_id, demo_idx,
                        init_states, demo_actions=f["data"][demo_key]["actions"][:],
                    ))
                    demos_since_recreate += 1
                    if not frames:
                        log.warning("no frames produced for %s demo %s", suite, demo_key)
                        continue

                    self._save_npz(out_path, suite, task, task_id, frames)
                    yield out_path
            finally:
                try:
                    env.close()
                except Exception as exc:  # pragma: no cover
                    log.warning("env.close() raised %s — ignoring", exc)

    def replay_demo(
        self,
        suite: str,
        task_id: int,
        demo_idx: int = 0,
    ) -> Iterator[LiberoFrame]:
        """Yield per-step LiberoFrame for `(suite, task_id, demo_idx)`. No disk IO.

        Single-demo convenience path; creates + closes its own env.
        """
        import h5py

        bench = self._load_benchmark(suite)
        task = bench.get_task(task_id)
        demo_path = self._demo_path(suite, task)
        with h5py.File(demo_path, "r") as f:
            demo_keys = sorted(f["data"].keys())
            if demo_idx >= len(demo_keys):
                raise IndexError(f"demo_idx {demo_idx} >= {len(demo_keys)} demos")
            demo_key = demo_keys[demo_idx]
            init_states = bench.get_task_init_states(task_id)
            env = self._make_env(suite, task)
            try:
                yield from self._replay_one_demo_in_env(
                    env, task_id, demo_idx, init_states,
                    demo_actions=f["data"][demo_key]["actions"][:],
                )
            finally:
                try:
                    env.close()
                except Exception as exc:  # pragma: no cover
                    log.warning("env.close() raised %s — ignoring", exc)

    # --- internals --------------------------------------------------------

    def _load_benchmark(self, suite: str):
        from libero.libero.benchmark import get_benchmark
        return get_benchmark(suite)()

    def _demo_path(self, suite: str, task) -> Path:
        # LIBERO Task namedtuple: `task.name` (stem), `task.bddl_file`
        # (basename with .bddl), `task.problem_folder` (suite). Datasets
        # convention is `<task.name>_demo.hdf5` inside `<demos_root>/<suite>/`.
        return self.cfg.demos_root / suite / f"{task.name}_demo.hdf5"

    def _make_env(self, suite: str, task):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        # Prefer LIBERO's own resolver so tests don't pin to autodl-tmp paths.
        bddl_dir = Path(get_libero_path("bddl_files"))
        bddl_file = bddl_dir / task.problem_folder / task.bddl_file
        if not bddl_file.is_file():
            # Fallback to our configured libero_root tree.
            bddl_file = (
                self.cfg.libero_root / "libero" / "libero" / "bddl_files"
                / task.problem_folder / task.bddl_file
            )
        if not bddl_file.is_file():
            raise FileNotFoundError(f"bddl file not found under either resolver: {bddl_file}")

        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=self.cfg.resolution,
            camera_widths=self.cfg.resolution,
            camera_depths=True,                       # the whole point
            camera_names=[self.cfg.camera],
        )
        env.seed(self.cfg.seed)
        return env

    def _replay_one_demo_in_env(
        self,
        env,
        task_id: int,
        demo_idx: int,
        init_states,
        demo_actions,
    ) -> Iterator[LiberoFrame]:
        """Replay one demo inside a *shared* env (do NOT close env here).

        We skip `env.reset()` here — the caller has already called reset() once
        after env construction. Each successive call to `reset()` in
        robosuite/LIBERO triggers a full model rebuild
        (`_load_robots -> SingleArm.__init__ -> deepcopy`) that gets
        pathologically slow after ~25 demos. `set_init_state()` is sufficient
        to position the simulation for the next replay.
        """
        import numpy as np
        from robosuite.utils import camera_utils

        env.set_init_state(init_states[demo_idx])
        # NOTE: take the sim handle AFTER env.reset() / set_init_state.
        # LIBERO/Robosuite re-instantiate the underlying MjSim on reset, so a
        # reference captured before reset would be stale and `.model` unbound
        # at first use ("'MjSim' object has no attribute 'model'").
        sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
        if self.cfg.metric_depth and sim is None:
            raise RuntimeError(
                "metric_depth=True requires a MuJoCo sim handle (env.sim or env.env.sim)"
            )

        n_steps = min(int(len(demo_actions)), int(self.cfg.max_steps_per_demo))
        for t in range(n_steps):
            action = np.asarray(demo_actions[t], dtype=np.float32)
            obs, _, _, _ = env.step(action)
            if t % self.cfg.stride != 0:
                continue
            rgb = np.asarray(obs[f"{self.cfg.camera}_image"], dtype=np.uint8)
            depth = np.asarray(obs[f"{self.cfg.camera}_depth"], dtype=np.float32)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if self.cfg.metric_depth:
                depth = camera_utils.get_real_depth_map(sim, depth.astype(np.float32))
                np.clip(depth, 0.0, self.cfg.depth_clip_m, out=depth)
            proprio: np.ndarray | None = None
            if self.cfg.save_proprio:
                pieces = []
                for k in self.cfg.proprio_keys:
                    v = obs.get(k)
                    if v is None:
                        continue
                    pieces.append(np.asarray(v, dtype=np.float32).reshape(-1))
                if pieces:
                    proprio = np.concatenate(pieces, axis=0)
            yield LiberoFrame(rgb=rgb, depth=depth, action=action, proprio=proprio, step=t)

    def _save_npz(
        self,
        out_path: Path,
        suite: str,
        task,
        task_id: int,
        frames: list[LiberoFrame],
    ) -> None:
        import numpy as np

        rgb = np.stack([f.rgb for f in frames], axis=0).astype(np.uint8)        # (T,H,W,3)
        depth = np.stack([f.depth for f in frames], axis=0).astype(np.float16)  # (T,H,W)
        action = np.stack([f.action for f in frames], axis=0).astype(np.float32)  # (T,A)
        proprio_arr = None
        if self.cfg.save_proprio and frames[0].proprio is not None:
            proprio_arr = np.stack([f.proprio for f in frames], axis=0).astype(np.float32)  # (T,P)
        meta = {
            "suite": suite,
            "task_id": int(task_id),
            "task_name": getattr(task, "name", "") or "",
            "language": getattr(task, "language", None) or "",
            "bddl_file": getattr(task, "bddl_file", "") or "",
            "camera": self.cfg.camera,
            "resolution": int(self.cfg.resolution),
            "n_frames": int(rgb.shape[0]),
            "image_fix_applied": False,        # PSSA's 180-deg rotate is applied at training/eval time
            "depth_units": "meters" if self.cfg.metric_depth else "[0,1] normalised z-buffer (Robosuite native)",
            "depth_clip_m": float(self.cfg.depth_clip_m) if self.cfg.metric_depth else None,
            "proprio_keys": list(self.cfg.proprio_keys) if proprio_arr is not None else [],
            "proprio_dim": int(proprio_arr.shape[-1]) if proprio_arr is not None else 0,
            "extractor_version": 3,            # bumped: per-step proprio added (eef pos+quat+gripper concat)
            **self.cfg.extra_metadata,
        }

        save = np.savez_compressed if self.cfg.compress else np.savez
        save_kwargs = dict(rgb=rgb, depth=depth, action=action, meta=json.dumps(meta))
        if proprio_arr is not None:
            save_kwargs["proprio"] = proprio_arr
        save(out_path, **save_kwargs)
        log.info(
            "wrote %s frames=%d rgb=%.1fMB depth=%.1fMB",
            out_path.name, rgb.shape[0],
            rgb.nbytes / 1e6, depth.nbytes / 1e6,
        )


__all__ = [
    "DEFAULT_LIBERO_ROOT", "DEFAULT_DEMOS_ROOT", "DEFAULT_CAMERA",
    "DEFAULT_RESOLUTION", "DEFAULT_OUT_ROOT",
    "LiberoExtractorConfig", "LiberoFrame", "LiberoDepthExtractor",
]
