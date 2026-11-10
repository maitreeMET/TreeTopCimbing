# TreeTopClimbing

Teaching humanoids to climb tree-top rope courses. First task: walk a suspended plank bridge
(swinging wooden slats on cables, hands on an overhead guide cable), as in a kids' adventure park.
Simulated in NVIDIA Isaac Sim via Isaac Lab, trained with RL.

Repo location on bahamut: `~/TreeTopClimbing` (root disk, not `/data`).

## Stack (pinned 2026-11-10)
| Component | Version | Notes |
|---|---|---|
| Isaac Lab | 3.0.0 (2026-10-20) | Python 3.12, uv-managed env |
| Isaac Sim | 6.2.0.0 | PhysX + RTX, installed as the `isaacsim` extra |
| Newton | 1.6.1 | default physics backend in Lab 3.0, kit-less |
| PyTorch | 2.11 cu130 | needs driver >= 580.65 (tiamat: 580.95) |
| RL | rsl-rl, skrl | |

## Machines
* **tiamat** (4x RTX PRO 6000 Blackwell 96 GB): development + baseline on ONE GPU.
  GPUs 0 and 2 are in use by another user; GPU 1 is dead to CUDA (torch cannot init it). Use GPU 3,
  bound by UUID (`scripts/isaaclab_env.sh` does this).
  `/data` is full; the Isaac env lives on the root disk at `/tiamat-data/home/maitree-tiamat/isaac`.
* **bahamut** (4x RTX PRO 6000 Blackwell 96 GB): this checkout. All 4 GPUs healthy and free as of
  2026-11-10; dev runs default to GPU 3. Isaac env at `~/isaac` (root disk; `/data` is 98% full).
* Scale to 4 GPUs only after the env and a baseline are solid.

## Setup
```bash
bash scripts/setup_env.sh          # ~15-20 GB download, idempotent
source scripts/isaaclab_env.sh     # sets UV_PROJECT, caches, CUDA_VISIBLE_DEVICES=<GPU 3 UUID>
bash scripts/smoke_test.sh         # headless Cartpole on Newton and PhysX
```

## Layout
```
treetop/tasks/    task configs (plank bridge env, rewards, curriculum)
treetop/assets/   USD / meshes for the bridge, cables, slats
scripts/          setup, smoke test, train/eval launchers
docs/PLAN.md      phased plan
logs/             local run logs (git-ignored)
```
