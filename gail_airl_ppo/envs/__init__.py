import logging

from gym.envs import register

_REGISTERED = False
def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    register(id='DisabledAnt-v0', entry_point='GAIL-IRL-PPO.gail_irl_ppo.envs.disabled_ant_env:CustomAntEnv',
             kwargs={'gear': 30})
