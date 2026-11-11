"""Task-specific MDP terms for the plank bridge."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bridge_progress(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Distance walked along the bridge (+x) since the spawn point, in metres."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w.torch[:, 0] - env.scene.env_origins[:, 0]


def bridge_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    cross_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Move to a harder bridge after crossing it, to an easier one after failing to get halfway.

    Args:
        cross_distance: Progress (m) from the spawn point that counts as having crossed the bridge.
    """
    terrain = env.scene.terrain
    progress = bridge_progress(env, asset_cfg)[env_ids]
    move_up = progress > cross_distance
    move_down = (progress < 0.5 * cross_distance) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
