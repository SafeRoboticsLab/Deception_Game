# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Tuple, Optional, Any, Dict
import time
import numpy as np

from .base_policy import BasePolicy


class LinearPolicy(BasePolicy):

  def __init__(
      self, id: str, nominal_states: np.ndarray, nominal_controls: np.ndarray,
      K_closed_loop: np.ndarray, k_open_loop: np.ndarray, cfg: Any, **kwargs
  ):
    super().__init__(id, cfg)
    self.policy_type = "Linear"

    # Params
    assert nominal_states.shape[-1] == nominal_controls.shape[
        -1] == K_closed_loop.shape[-1] == k_open_loop.shape[-1]
    self.nominal_states = nominal_states.copy()
    self.nominal_controls = nominal_controls.copy()
    self.K_closed_loop = K_closed_loop.copy()
    self.k_open_loop = k_open_loop.copy()

  def get_action(
      self, obs: np.ndarray,
      agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        state (np.ndarray): current state.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    time_idx = kwargs.get("time_idx")
    state = kwargs.get('state')

    time0 = time.time()
    action = (
        self.nominal_controls[:, time_idx] + self.K_closed_loop[..., time_idx]
        @ (state - self.nominal_states[:, time_idx])
        + self.k_open_loop[..., time_idx]
    )
    t_process = time.time() - time0
    status = 1
    return action, dict(t_process=t_process, status=status)
