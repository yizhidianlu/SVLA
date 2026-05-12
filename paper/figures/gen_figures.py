"""Generate Phase-1 paper figures from training/VQ-VAE metrics.

Outputs (under same dir as this script):
  fig_vqvae_curves.pdf   — VQ-VAE recon RMSE + codes_used over training
  fig_train_curves.pdf   — GeoRel-VLA depth + action loss over training
  fig_codebook_compare.pdf — v1 vs v2 codes-used story (the EMA fix payoff)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# ----- load metrics -----
vq = json.load(open(HERE / "vqvae_metrics.json"))
tr = json.load(open(HERE / "train_metrics.json"))


# ----- VQ-VAE curves (recon RMSE + codes_used) -----

fig, ax1 = plt.subplots(figsize=(5.0, 3.0))
steps = [r["step"] for r in vq]
rmse = [r["rmse_m"] for r in vq]
codes = [r["codes_used"] for r in vq]

ax1.plot(steps, rmse, color="tab:blue", marker="o", markersize=3, lw=1.2,
         label="recon RMSE (m)")
ax1.set_xlabel("training step")
ax1.set_ylabel("recon RMSE (m)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylim(0, max(rmse) * 1.05)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(steps, codes, color="tab:orange", marker="s", markersize=3, lw=1.2,
         label="codes used / 256")
ax2.set_ylabel("codes used", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax2.set_ylim(0, 256)
ax2.axhline(256, ls=":", color="tab:orange", alpha=0.4)

ax1.set_title("VQ-VAE pretrain — depth recon vs codebook utilisation (EMA + dead-code reset)")
fig.tight_layout()
fig.savefig(HERE / "fig_vqvae_curves.pdf")
fig.savefig(HERE / "fig_vqvae_curves.png", dpi=150)
plt.close(fig)
print(f"wrote fig_vqvae_curves.{{pdf,png}} (final: codes={codes[-1]}/256, RMSE={rmse[-1]:.3f}m)")


# ----- Training curves (depth + action loss) -----

fig, ax = plt.subplots(figsize=(5.5, 3.2))
steps = [r["step"] for r in tr]
depth = [r["loss_depth"] for r in tr]
action = [r.get("loss_action", float("nan")) for r in tr]
total = [r["loss_total"] for r in tr]

ax.plot(steps, depth, color="tab:blue", lw=1.0, alpha=0.7, label=r"$L_{\mathrm{depth}}$ (CE over codebook)")
ax.plot(steps, action, color="tab:red", lw=1.0, alpha=0.7, label=r"$L_{\mathrm{action}}$ (CFM)")
ax.plot(steps, total, color="black", lw=1.4, label=r"$L_{\mathrm{total}} = \lambda_t L_d + L_a$")
ax.set_xlabel("training step")
ax.set_ylabel("loss")
ax.set_yscale("log")
ax.set_title("GeoRel-VLA — Phase-1 training (5 epochs, batch=16, A800)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(HERE / "fig_train_curves.pdf")
fig.savefig(HERE / "fig_train_curves.png", dpi=150)
plt.close(fig)
print(f"wrote fig_train_curves.{{pdf,png}} (final: depth={depth[-1]:.3f}, action={action[-1]:.3f})")


# ----- Codebook collapse comparison (v1 vs v2) -----

# v2 codes_used trajectory (real, from metrics)
v2_steps = [r["step"] for r in vq]
v2_codes = [r["codes_used"] for r in vq]

# v1 codes_used trajectory: from the bug-log run, codes were stuck at 5-6 throughout
# (real numbers from /autodl-fs/data/svla/runs/auto_chain.log v1 history)
v1_steps_approx = list(range(50, 1500, 50))
v1_codes_approx = [6] * len(v1_steps_approx)
# actually first ~150 steps showed up to 166 then collapsed; capture more honestly:
v1_known = {
    50: 166, 100: 20, 150: 5, 200: 5, 400: 6, 700: 6, 1000: 6, 1200: 6, 1450: 6,
}
v1_steps = sorted(v1_known.keys())
v1_codes = [v1_known[s] for s in v1_steps]

fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.plot(v1_steps, v1_codes, color="tab:red", marker="x", markersize=6, lw=1.2,
        label="v1: vanilla VQ-VAE (gradient codebook)")
ax.plot(v2_steps, v2_codes, color="tab:green", marker="o", markersize=3, lw=1.2,
        label="v2: + EMA + k-means init + dead-code reset")
ax.axhline(256, ls=":", color="grey", alpha=0.5, label="K = 256 capacity")
ax.set_xlabel("training step")
ax.set_ylabel("active codes (out of 256)")
ax.set_title("VQ-VAE codebook utilisation: 6/256 (collapse) → 250/256 (97.7\\%)")
ax.legend(loc="center right", fontsize=8)
ax.set_ylim(0, 280)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "fig_codebook_compare.pdf")
fig.savefig(HERE / "fig_codebook_compare.png", dpi=150)
plt.close(fig)
print(f"wrote fig_codebook_compare.{{pdf,png}}")
