from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1RoughPPORunnerCfg


@configclass
class G1PlankBridgePPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.experiment_name = "treetop_plank_bridge_g1"
