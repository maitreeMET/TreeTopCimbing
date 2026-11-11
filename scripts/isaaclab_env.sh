# source this to get `uv run` pointed at the Isaac Lab project env
export ISAAC_ROOT="${ISAAC_ROOT:-$HOME/isaac}"
export PATH=$HOME/.local/bin:/data/maitree-tiamat/miniconda3/bin:$PATH
export UV_CACHE_DIR="$ISAAC_ROOT/uv-cache" UV_PYTHON_INSTALL_DIR="$ISAAC_ROOT/uv-python" UV_LINK_MODE=copy
export UV_PROJECT="$ISAAC_ROOT/IsaacLab"
# NVIDIA Omniverse EULA accepted by the user on 2026-11-10
export OMNI_KIT_ACCEPT_EULA=YES
# Private temp dir: Isaac Lab caches downloaded assets under $TMPDIR, and /tmp/https is owned by another user on bahamut
export TMPDIR="$ISAAC_ROOT/tmp"; mkdir -p "$TMPDIR"
# Default dev GPU: GPU 3 is free on tiamat (0 and 2 belong to another user as of 2026-11-10)
# GPU 1 (GPU-ff388b9b) is DEAD to CUDA on tiamat as of 2026-11-10 (nvidia-smi shows it, torch cannot init it).
# Bind by UUID so numbering shifts never land on it. GPU 3 = free + working. GPUs 0,2 = other user.
if [ "$(hostname)" = tiamat ]; then _DEV=GPU-48ae2050-ef0d-f098-3ec1-9aa2f9367d2e; else _DEV=3; fi   # bahamut: plain index, all 4 GPUs healthy
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$_DEV}"
# Isaac Sim / Kit caches live under $HOME (root disk), which has space.
alias il='uv run --project "$UV_PROJECT" --extra isaacsim --extra rsl-rl'
# Make the `treetop` package importable (tasks register via --external_callback treetop.register_tasks)
export TREETOP_ROOT="${TREETOP_ROOT:-$HOME/TreeTopClimbing}"
export PYTHONPATH="$TREETOP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
