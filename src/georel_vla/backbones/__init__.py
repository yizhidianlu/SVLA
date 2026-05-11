"""Action / VLM backbones consumed by GeoRel-VLA.

Phase 1 uses open-pi-zero (allenzren/open-pi-zero, pinned via git submodule
in `third_party/open-pi-zero`). The wrapper in `pi0.py` is intentionally a
*skeleton* this turn — full forward integration (loading PaliGemma + SigLIP
+ action expert weights, exposing SigLIP features to the depth expert, and
the hybrid attention plumbing) lands in Phase 1.6 (model.py wiring).
"""
