from .agent import Agent

from .base_env import BaseEnv
from .base_single_env import BaseSingleEnv
from .base_zs_env import BaseZeroSumEnv

from .race_car.track import Track
from .race_car.race_car_single import RaceCarSingle5DEnv
from .race_car.race_car_dstb import RaceCarDstb5DEnv
from .race_car.uber_no_belief import UberNoBeliefEnv
from .race_car.belief_game_env import BeliefGameEnv, BeliefGameWithIntentsEnv

from .race_car.cost_bicycle5D import (
    Bicycle5DCost, Bicycle5DConstraint, Bicycle5DReachabilityCost, Bicycle5DRefTrajCost
)

from .cost.quadratic_cost import QuadraticCost
from .cost.half_space_cost import (
    UpperHalfCost, LowerHalfCost, UpperHalfBoxFootprintCost, LowerHalfBoxFootprintCost
)
from .cost.base_cost import BarrierCost, BaseCost
from .cost.box_cost import BoxObsCost, BoxObsBoxFootprintCost

from .dynamics.bicycle5D import Bicycle5D

from .policy.base_policy import BasePolicy
from .policy.nn_policy import NeuralNetworkControlSystem
from .policy.ilqr_policy import ILQR
from .policy.ilqr_spline_policy import ILQRSpline
from .policy.linear_policy import LinearPolicy

from .vec_env.subproc_vec_env import SubprocVecEnv
from .vec_env.vec_env import VecEnvBase

from .utils import (save_obj, load_obj, PrintLogger, parallel_apply, parallel_iapply)

import gym

gym.envs.register(  # no time limit imposed
    id='RaceCarSingle5DEnv-v1',
    entry_point=RaceCarSingle5DEnv,
)

gym.envs.register(  # no time limit imposed
    id='RaceCarDstb5DEnv-v1',
    entry_point=RaceCarDstb5DEnv,
)

# Prevents from opening a pybullet simulator when we don't need to.
# from .spirit_rl.spirit_pybullet_env import SpiritPybulletEnv
# from .spirit_rl.spirit_pybullet_single import SpiritPybulletSingleEnv
# from .spirit_rl.spirit_pybullet_zs import SpiritPybulletZeroSumEnv
