"""Unitree G1 walking across a plank bridge (v0: static slats, no hand cable yet).

Built on Isaac Lab's G1 rough-terrain velocity task, which already gives the policy a height scan
of the ground in front of it. We swap the terrain for generated plank bridges, always command
"walk forward along the bridge", end the episode when the robot falls, and raise the bridge
difficulty (gap width, slat depth, sag) as the robot learns to cross.
"""

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

from treetop.terrains.plank_bridge import PlankBridgeTerrainCfg

from . import mdp

START_LEN = 3.0
BRIDGE_LEN = 8.0
SPAWN_X = 1.5

PLANK_BRIDGE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(20.0, 4.0),
    border_width=5.0,
    num_rows=10,  # difficulty levels
    num_cols=20,  # bridges per level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "plank_bridge": PlankBridgeTerrainCfg(
            proportion=1.0,
            start_len=START_LEN,
            bridge_len=BRIDGE_LEN,
            spawn_x=SPAWN_X,
            slat_depth_range=(0.25, 0.15),
            gap_range=(0.03, 0.30),
            sag_range=(0.0, 0.30),
        ),
    },
)


@configclass
class G1PlankBridgeEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # terrain: plank bridges, easiest levels first
        self.scene.terrain.terrain_generator = PLANK_BRIDGE_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 2
        # finer height scan so gaps between slats show up
        self.scene.height_scanner.pattern_cfg.resolution = 0.05

        # commands: always walk forward along the bridge (+x), facing +x
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (0.4, 0.7)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.heading = (0.0, 0.0)
        cmd.heading_command = True
        cmd.rel_heading_envs = 1.0
        cmd.rel_standing_envs = 0.0

        # resets: spawn on the start platform, roughly facing the bridge
        self.events.reset_base.params["pose_range"] = {"x": (-0.3, 0.3), "y": (-0.2, 0.2), "yaw": (-0.3, 0.3)}
        # gentler pushes than open ground, there is nowhere to step sideways
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.2, 0.2)}

        # terminations: falling off the bridge
        self.terminations.fell_off = DoneTerm(func=base_mdp.root_height_below_minimum, params={"minimum_height": -0.5})

        # curriculum: harder bridge after reaching the far platform
        cross_distance = START_LEN + BRIDGE_LEN - SPAWN_X + 0.5
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.bridge_terrain_levels, params={"cross_distance": cross_distance}
        )


@configclass
class G1PlankBridgeEnvCfg_PLAY(G1PlankBridgeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 6
        self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
