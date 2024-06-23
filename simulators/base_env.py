# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union
import random
import numpy as np
import gym
import torch

from .utils import GenericAction, GenericState
from .agent import Agent


class BaseEnv(gym.Env, ABC):
  agent: Agent

  def __init__(self, cfg_env) -> None:
    gym.Env.__init__(self)
    self.cnt = 0
    self.timeout = cfg_env.timeout
    self.end_criterion = cfg_env.end_criterion

  @abstractmethod
  def step(self, action: GenericAction) -> Tuple[GenericState, float, bool, Dict]:
    raise NotImplementedError

  @abstractmethod
  def get_obs(self, state: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  def reset(
      self, state: Optional[GenericState] = None, cast_torch: bool = False, **kwargs
  ) -> GenericState:
    """
    Resets the environment and returns the new state.

    Args:
        state (Optional[GenericState], optional): reset to this state if
            provided. Defaults to None.
        cast_torch (bool): cast state to torch if True.

    Returns:
        np.ndarray: the new state.
    """
    self.cnt = 0

  @abstractmethod
  def render(self):
    raise NotImplementedError

  def seed(self, seed: int = 0) -> None:
    self.seed_val = seed
    self.rng = np.random.default_rng(seed)
    random.seed(self.seed_val)
    torch.manual_seed(self.seed_val)
    torch.cuda.manual_seed(self.seed_val)
    torch.cuda.manual_seed_all(self.seed_val)  # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    self.action_space.seed(seed)
    self.observation_space.seed(seed)

  @abstractmethod
  def report(self):
    raise NotImplementedError

  @abstractmethod
  def get_constraints_all(self, states: np.ndarray, actions: Union[np.ndarray,
                                                                   dict]) -> Dict[str, np.ndarray]:
    """
    Gets the values of all constaint functions given current state, current
    action, and next state.

    Args:
        state (np.ndarray)
        controls (np.ndarray)

    Returns:
        Dict: each (key, value) pair is the name and values of a constraint
            evaluated at the states and controls input.
    """
    raise NotImplementedError
