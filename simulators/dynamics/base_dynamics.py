# ------------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: haiminh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# Modified from: https://github.com/SafeRoboticsLab/simulators
# ------------------------------------------------------------

"""A parent class for dynamics with control as the only input.

This file implements a parent class for dynamics with control as the only
input. A child class should implement `integrate_forward_jax()` and
`_integrate_forward()` the parent class takes care of the numpy version and
derivatives functions.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np

from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp


class BaseDynamics(ABC):
  dim_x: int

  def __init__(self, cfg: Any, action_space: np.ndarray) -> None:
    """
    Args:
        cfg (Any): an object specifies cfguration.
        action_space (np.ndarray): the action space of the dynamics. The first
          column is the loweer bound and the second column is the upper bound.
    """
    self.dt: float = cfg.dt  # time step for each planning step
    self.ctrl_space = action_space.copy()
    self.dim_u: int = self.ctrl_space.shape[0]

  def integrate_forward(self, state: np.ndarray, control: np.ndarray,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the next state of the vehicle given the current state and
    control input.

    Args:
        state (np.ndarray).
        control (np.ndarray).

    Returns:
        np.ndarray: next state.
        np.ndarray: clipped control.
    """
    state_nxt, ctrl_clip = self.integrate_forward_jax(jnp.array(state), jnp.array(control))
    return np.array(state_nxt), np.array(ctrl_clip)

  @abstractmethod
  def integrate_forward_jax(self, state: DeviceArray,
                            control: DeviceArray) -> Tuple[DeviceArray, DeviceArray]:
    """
    Computes one-step time evolution of the system: x+ = f(x, u) with
    additional treatment on state and/or control constraints.

    Args:
        state (DeviceArray)
        control (DeviceArray)

    Returns:
        DeviceArray: next state.
    """
    raise NotImplementedError

  @abstractmethod
  def _integrate_forward(self, state: DeviceArray, control: DeviceArray) -> DeviceArray:
    """Computes one-step time evolution of the system: x+ = f(x, u).

    Args:
        state (DeviceArray)
        control (DeviceArray)

    Returns:
        DeviceArray: next state.
    """
    raise NotImplementedError

  @partial(jax.jit, static_argnames='self')
  def get_jacobian(self, nominal_states: DeviceArray,
                   nominal_controls: DeviceArray) -> Tuple[DeviceArray, DeviceArray]:
    """
    Returns the linearized 'A' and 'B' matrix of the ego vehicle around
    nominal states and controls.

    Args:
        nominal_states (DeviceArray): states along the nominal trajectory.
        nominal_controls (DeviceArray): controls along the trajectory.

    Returns:
        DeviceArray: the Jacobian of the dynamics w.r.t. the state.
        DeviceArray: the Jacobian of the dynamics w.r.t. the control.
    """
    _jac = jax.jacfwd(self._integrate_forward, argnums=[0, 1])
    jac = jax.jit(jax.vmap(_jac, in_axes=(1, 1), out_axes=(2, 2)))
    return jac(nominal_states, nominal_controls)
