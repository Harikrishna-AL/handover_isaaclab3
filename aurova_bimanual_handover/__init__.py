"""Bimanual UR5e + Kinova GEN3 handover environment (Isaac Lab 3.0, PhysX backend, kit-less capable)."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Bimanual-Direct-reach-v0",
    entry_point=f"{__name__}.bimanual_direct_env:BimanualDirect",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bimanual_direct_env_cfg:BimanualDirectCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
