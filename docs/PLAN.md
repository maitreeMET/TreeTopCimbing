# Plan

## Phase 0: environment (tiamat, 1 GPU)
- [ ] Isaac Lab 3.0 + Isaac Sim 6.2 installed (`scripts/setup_env.sh`)
- [ ] Headless Cartpole trains on GPU 3 on both Newton and PhysX backends
- [ ] Humanoid baseline: `Isaac-Velocity-Flat-G1` (or H1) runs headless, video capture works

## Phase 1: plank-bridge task (easy version of the photo)
- Build the bridge: N wooden slats hung on two parallel cables between two platforms,
  plus one overhead guide cable at hand height. Start rigid/static slats, then make them swing.
- Observations: proprioception + slat/cable poses relative to the pelvis (privileged first).
- Rewards: forward progress along the bridge, upright pelvis, foot placement on slats,
  penalties for falling / touching cables with feet.
- Curriculum: static slats -> swinging slats -> wider gaps -> wind perturbations.

## Phase 2: scale
- 4-GPU training (torchrun / `--distributed`) once the single-GPU recipe is stable.
- Domain randomization for sim-to-real later.

## Known issues (bahamut, 2026-11-10)
- **Newton backend: works.** Cartpole trains 20 iters on GPU 3 (`logs/smoke_test.log`).
- **Isaac Sim PhysX backend: GPU physics not working yet.** PhysX foundation finds the GPU
  ("CUDA device 0 found... GPU available") but at scene creation logs
  "No CUDA context manager available; forcing CPU simulation", then Isaac Lab crashes with a
  cuda:0 vs cpu tensor mismatch. Same on GPU 0, GPU 3, and with all GPUs visible; enabling
  `omni.physx.gpu` via `--kit_args` did not help. Verbose log: `logs/smoke_physx_verbose.log`.
- Fixed locally: Isaac Lab 3.0.0 PhysX `joint_acc` passed a ProxyArray to a warp kernel
  (`patches/isaaclab-physx-joint-acc.patch`, apply in `~/isaac/IsaacLab` with `git apply`).
- `/tmp/https` is owned by another user, so `isaaclab_env.sh` sets `TMPDIR=~/isaac/tmp`.
- **Root cause of the PhysX + video failures: NVIDIA Vulkan is broken on bahamut.** The Vulkan
  loader finds `/usr/share/vulkan/icd.d/nvidia_icd.json`, but `libGLX_nvidia.so.0` (595.71.05,
  matches the kernel module) returns NULL for `vkCreateInstance`, so only llvmpipe (CPU) shows up.
  Same result with the system loader 1.3.275, Isaac Sim's bundled loader 1.4.313, system/conda/uv
  Python, and DISPLAY unset. Isaac Sim logs "Failed to create any GPU devices". Needs an admin to
  look at the driver install. Newton training is unaffected (no Vulkan needed).
