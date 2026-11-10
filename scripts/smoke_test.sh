#!/bin/bash
# Smoke test (headless is the 3.0 default) on ONE GPU: Cartpole with rsl_rl on (a) Newton and (b) Isaac Sim PhysX.
# Usage: bash scripts/smoke_test.sh [CUDA_VISIBLE_DEVICES value, default = GPU 3 UUID]
set -euo pipefail
source "$(dirname "$0")/isaaclab_env.sh"
export CUDA_VISIBLE_DEVICES="${1:-$CUDA_VISIBLE_DEVICES}"
cd "$UV_PROJECT"
echo "== (a) Newton backend, 64 envs, 20 iters =="
uv run --extra rsl-rl isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct \
  --num_envs 64 --max_iterations 20
echo "== (b) Isaac Sim PhysX backend, 64 envs, 20 iters =="
uv run --extra isaacsim --extra rsl-rl isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct \
  physics=isaacsim_physx --num_envs 64 --max_iterations 20
echo "SMOKE_OK"
