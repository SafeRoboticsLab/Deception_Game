# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for vectorized environments especially for RL training.

Modified from stable-baseline3 and https://github.com/allenzren/alano.
"""

from typing import Type, Any, Dict, Optional
import torch as th
import gym
from .subproc_vec_env import SubprocVecEnv


def make_vec_envs(
    env_type: Any, num_processes: int, seed: int = 0, cpu_offset: int = 0,
    vec_env_type: Type[SubprocVecEnv] = SubprocVecEnv,
    venv_kwargs: Optional[Dict] = None, env_kwargs: Optional[Dict] = None
) -> SubprocVecEnv:

  if env_kwargs is None:
    env_kwargs = {}
  if venv_kwargs is None:
    venv_kwargs = {}
  if isinstance(env_type, str):
    envs = [gym.make(env_type, **env_kwargs) for i in range(num_processes)]
  else:
    envs = [env_type(**env_kwargs) for _ in range(num_processes)]
  for rank, env in enumerate(envs):
    env.seed(seed + rank)
  envs = vec_env_type(envs, cpu_offset, **venv_kwargs)
  return envs


class VecEnvBase(SubprocVecEnv):
  """
  Mostly for torch
  """

  def __init__(
      self, venv, cpu_offset=0, device: str = th.device("cpu"),
      pickle_option='cloudpickle', start_method=None
  ):
    super(VecEnvBase, self).__init__(
        venv, cpu_offset, pickle_option=pickle_option,
        start_method=start_method
    )
    self.device = device

  def reset(self, **kwargs):
    obs = super().reset(**kwargs)
    return th.FloatTensor(obs).to(self.device)

  def reset_one(self, index, **kwargs):
    obs = self.env_method('reset', indices=[index], **kwargs)[0]
    return th.FloatTensor(obs).to(self.device)

  # Overrides
  def step_async(self, actions):
    if isinstance(actions, th.Tensor):
      actions = actions.cpu().numpy()
    super().step_async(actions)

  # Overrides
  def step_wait(self):
    obs, reward, done, info = super().step_wait()
    obs = th.FloatTensor(obs).to(self.device)
    reward = th.FloatTensor(reward).unsqueeze(dim=1).float()
    return obs, reward, done, info

  def get_obs(self, states):
    method_args_list = [(state,) for state in states]
    obs = th.FloatTensor(
        self.env_method_arg(
            '_get_obs', method_args_list=method_args_list,
            indices=range(self.n_envs)
        )
    )
    return obs.to(self.device)
