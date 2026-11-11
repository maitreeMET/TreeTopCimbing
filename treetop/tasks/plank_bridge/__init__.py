"""Plank bridge task: Unitree G1 walks across a suspended slatted bridge."""

import gymnasium as gym

from . import agents

gym.register(
    id="TreeTop-PlankBridge-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1PlankBridgeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PlankBridgePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="TreeTop-PlankBridge-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1PlankBridgeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PlankBridgePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
