from .ppo import PPO, PPOExpert
from .sac import SAC, SACExpert
from .gail import GAIL
from .airl import AIRL

ALGOS = {
    'gail': GAIL,
    'airl': AIRL
}
