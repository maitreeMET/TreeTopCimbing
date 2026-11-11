"""Plank-bridge terrain: wooden slats hung between two platforms over a drop.

Tile layout (x is the walking direction, the robot starts on the start platform):

    x = 0 ......... start_len ............................ start_len + bridge_len ........ size[0]
        [ start platform ][ slat ][gap][ slat ][gap] ... [ slat ][      end platform      ]
                                     deep floor far below (falling = episode over)

Difficulty (0 -> 1) makes the bridge harder: wider gaps, narrower slats and a deeper sag in the middle.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def _lerp(r: tuple[float, float], t: float) -> float:
    return r[0] + t * (r[1] - r[0])


def _box(size_xyz, center_xyz) -> trimesh.Trimesh:
    return trimesh.creation.box(size_xyz, trimesh.transformations.translation_matrix(center_xyz))


def plank_bridge_terrain(difficulty: float, cfg: "PlankBridgeTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    gap = _lerp(cfg.gap_range, difficulty)
    slat_depth = _lerp(cfg.slat_depth_range, difficulty)
    sag = _lerp(cfg.sag_range, difficulty)

    size_x, size_y = cfg.size
    cy = 0.5 * size_y
    t = cfg.platform_thickness
    meshes: list[trimesh.Trimesh] = []

    # platforms (top surface at z = 0)
    end_start = cfg.start_len + cfg.bridge_len
    meshes.append(_box((cfg.start_len, cfg.platform_width, t), (0.5 * cfg.start_len, cy, -0.5 * t)))
    end_len = size_x - end_start
    meshes.append(_box((end_len, cfg.platform_width, t), (end_start + 0.5 * end_len, cy, -0.5 * t)))

    # slats: evenly spaced, the bridge sags as a parabola (0 at both ends, -sag in the middle)
    pitch = slat_depth + gap
    n_slats = max(1, int((cfg.bridge_len - gap) // pitch))
    margin = 0.5 * (cfg.bridge_len - (n_slats * slat_depth + (n_slats - 1) * gap))
    for i in range(n_slats):
        x = cfg.start_len + margin + i * pitch + 0.5 * slat_depth
        u = (x - cfg.start_len) / cfg.bridge_len  # 0..1 along the bridge
        z_top = -sag * 4.0 * u * (1.0 - u)
        meshes.append(_box((slat_depth, cfg.slat_width, cfg.slat_thickness), (x, cy, z_top - 0.5 * cfg.slat_thickness)))

    # deep floor so a falling robot lands somewhere (the height scan sees it as "nothing there")
    meshes.append(_box((size_x, size_y, 0.2), (0.5 * size_x, cy, -cfg.drop_depth - 0.1)))

    origin = np.array([cfg.spawn_x, cy, 0.0])
    return meshes, origin


@configclass
class PlankBridgeTerrainCfg(SubTerrainBaseCfg):
    """Configuration for :func:`plank_bridge_terrain`. Lengths in metres."""

    function = plank_bridge_terrain

    start_len: float = 3.0
    """Length of the start platform along x."""
    bridge_len: float = 8.0
    """Length of the slatted section along x."""
    platform_width: float = 2.0
    platform_thickness: float = 0.3
    spawn_x: float = 1.5
    """Where the robot spawns along x (on the start platform)."""

    slat_width: float = 0.8
    """Width of each slat across the bridge (y)."""
    slat_thickness: float = 0.05
    slat_depth_range: tuple[float, float] = MISSING
    """Slat length along x at difficulty 0 and 1."""
    gap_range: tuple[float, float] = MISSING
    """Gap between slats at difficulty 0 and 1."""
    sag_range: tuple[float, float] = (0.0, 0.0)
    """How far the middle of the bridge hangs below the platforms, at difficulty 0 and 1."""
    drop_depth: float = 3.0
    """Depth of the floor below the bridge."""
