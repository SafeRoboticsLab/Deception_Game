# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A parent class for (control) policy.

This file implements a parent class for (control) policy. A child class should
implement `get_action()`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Optional
import numpy as np
import torch


class BasePolicy(ABC):
  obs_other_list: Optional[List]

  def __init__(self, id: str, cfg) -> None:
    super().__init__()
    self.id = id
    self.cfg = cfg
    self.device = torch.device(cfg.device)
    self.obs_other_list = None

  @abstractmethod
  def get_action(
      self, obs: np.ndarray,
      agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        obs (np.ndarray): current observation.
        agents_action (Optional[Dict]): other agents' actions that are
            observable to the ego agent.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    raise NotImplementedError

  def report(self):
    print(self.id)
    if self.obs_other_list is not None:
      print("  - The policy can observe:", end=' ')
      for i, k in enumerate(self.obs_other_list):
        print(k, end='')
        if i == len(self.obs_other_list) - 1:
          print('.')
        else:
          print(', ', end='')
    else:
      print("  - The policy can only access observation.")

  def to(self, device):
    self.device = device
