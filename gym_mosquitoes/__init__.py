import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Mosquitoes-v0',
    entry_point='gym_mosquitoes.envs:MosquitoesEnv'
)

register(
    id='RandMosquitoes-v0',
    entry_point='gym_mosquitoes.envs:RandMosquitoesEnv'
)
