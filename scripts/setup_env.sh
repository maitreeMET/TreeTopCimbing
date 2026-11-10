#!/bin/bash
# One-shot environment setup for TreeTopClimbing on tiamat.
# Installs Isaac Lab 3.0.0 (+ Isaac Sim 6.2 / PhysX, rsl-rl, skrl, wandb) into a uv env
# on the ROOT disk (/data is 100% full; / has ~140G free as of 2026-11-10).
#
# Usage:  bash scripts/setup_env.sh            (idempotent, safe to re-run)
# Logs:   $ISAAC_ROOT/install.log
set -euo pipefail

export ISAAC_ROOT="${ISAAC_ROOT:-$HOME/isaac}"   # tiamat: /tiamat-data/home/maitree-tiamat/isaac, bahamut: /home/maitree-tiamat/isaac
export ISAACLAB_TAG="${ISAACLAB_TAG:-v3.0.0}"
export PATH=$HOME/.local/bin:/data/maitree-tiamat/miniconda3/bin:$PATH   # uv lives in one of these
export UV_CACHE_DIR="$ISAAC_ROOT/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ISAAC_ROOT/uv-python"
export UV_LINK_MODE=copy

mkdir -p "$ISAAC_ROOT" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
cd "$ISAAC_ROOT"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

if [ ! -d IsaacLab ]; then
  git clone --branch "$ISAACLAB_TAG" --depth 1 https://github.com/isaac-sim/IsaacLab.git
fi
cd IsaacLab

# Core (Newton physics, CUDA-13 torch 2.11) + Isaac Sim 6.2 (PhysX/RTX) + RL libs.
uv sync --extra isaacsim --extra rsl-rl --extra skrl --extra wandb

# Sanity (GPU 1 on tiamat is dead to CUDA; pin a working GPU by UUID)
if [ "$(hostname)" = tiamat ]; then _DEV=GPU-48ae2050-ef0d-f098-3ec1-9aa2f9367d2e; else _DEV=3; fi   # bahamut: plain index, all 4 GPUs healthy
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$_DEV}"
uv run python - <<'PY'
import torch, importlib.metadata as m
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
for p in ("isaaclab", "isaacsim", "newton", "rsl-rl-lib", "skrl"):
    try: print(p, m.version(p))
    except Exception as e: print(p, "MISSING", e)
PY
echo "SETUP_DONE $(date)"
