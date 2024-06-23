# ------------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: haiminh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# Modified from: https://github.com/SafeRoboticsLab/simulators
# ------------------------------------------------------------

"""A parent class for dynamics with both control and disturbance.

This file implements a parent class for dynamics with both control and
disturbance. A child class should implement `integrate_forward_jax()` and
`_integrate_forward()` the parent class takes care of the numpy version and
derivatives functions.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Dict
import numpy as np

from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp


class BaseDstbDynamics(ABC):
  dim_x: int

  def __init__(self, cfg: Any, action_space: Dict[str, np.ndarray]) -> None:
    """
    Args:
        cfg (Any): an object specifies configuration.
        action_space (dict): the action space of the dynamics with 'ctrl' for
        control space and 'dstb' for disturbance space The first column is the
        loweer bound and the second column is the upper bound.
    """
    self.dt: float = cfg.dt  # time step for each planning step
    self.ctrl_space = action_space['ctrl'].copy()
    self.dstb_space = action_space['dstb'].copy()
    self.dim_u: int = self.ctrl_space.shape[0]
    self.dim_d: int = self.dstb_space.shape[0]

  def integrate_forward(
      self, state: np.ndarray, control: np.ndarray, noise: Optional[np.ndarray] = None,
      noise_type: Optional[str] = 'unif', adversary: Optional[np.ndarray] = None, **kwargs
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the next state of the vehicle given the current state, control input,
    and disturbance input.

    Args:
        state (np.ndarray).
        control (np.ndarray).
        noise (np.ndarray, optional): the ball radius or standard
            deviation of the Gaussian noise. The magnitude should be in the
            sense of self.dt. Defaults to None.
        noise_type(str, optional): Uniform or Gaussian. Defaults to 'unif'.
        adversary (np.ndarray, optional): adversarial control (disturbance).
            Defaults to None.

    Returns:
        np.ndarray: next state.
        np.ndarray: clipped control.
        np.ndarray: clipped disturbance.
    """
    if adversary is not None:
      # assert adversary.shape[0] == self.dim_x, ("Adversary dim. is incorrect!") #NOTE: dimension
      # of adversary (disturbance) need not equal state dimension!
      disturbance = adversary
    elif noise is not None:
      assert noise.shape[0] == self.dim_x, ("Noise dim. is incorrect!")
      cos = np.cos(state[3])
      sin = np.sin(state[3])
      transform_mtx = np.array([[cos, sin, 0, 0, 0], [-sin, cos, 0, 0, 0], [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
      if noise_type == 'unif':
        rv = (np.random.rand(self.dim_x) - 0.5) * 2  # Maps to [-1, 1]
      else:
        rv = np.random.normal(size=(self.dim_x))
      disturbance = (transform_mtx@noise) * rv
    else:
      disturbance = np.zeros(self.dim_x)
    state_nxt, ctrl_clip, dstb_clip = self.integrate_forward_jax(
        jnp.array(state), jnp.array(control), jnp.array(disturbance)
    )
    return np.array(state_nxt), np.array(ctrl_clip), np.array(dstb_clip)

  @abstractmethod
  def integrate_forward_jax(
      self, state: DeviceArray, control: DeviceArray, disturbance: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray, DeviceArray]:
    """
    Computes one-step time evolution of the system: x+ = f(x, u, d) with
    additional treatment on state and/or control constraints.

    Args:
        state (DeviceArray)
        control (DeviceArray)

    Returns:
        DeviceArray: next state.
    """
    raise NotImplementedError

  @abstractmethod
  def _integrate_forward(
      self, state: DeviceArray, control: DeviceArray, disturbance: DeviceArray
  ) -> DeviceArray:
    """Computes one-step time evolution of the system: x+ = f(x, u, d).

    Args:
        state (DeviceArray)
        control (DeviceArray)
        disturbance (DeviceArray)

    Returns:
        DeviceArray: next state.
    """
    raise NotImplementedError

  @partial(jax.jit, static_argnames='self')
  def get_jacobian(
      self, nominal_states: DeviceArray, nominal_controls: DeviceArray,
      nominal_disturbances: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray, DeviceArray]:
    """
    Returns the linearized 'A' and 'B' matrix of the ego vehicle around
    nominal states and controls.

    Args:
        nominal_states (DeviceArray): states along the nominal trajectory.
        nominal_controls (DeviceArray): controls along the trajectory.
        nominal_disturbances (DeviceArray): disturbances along the trajectory.

    Returns:
        DeviceArray: the Jacobian of the dynamics w.r.t. the state.
        DeviceArray: the Jacobian of the dynamics w.r.t. the control.
        DeviceArray: the Jacobian of the dynamics w.r.t. the disturbance.
    """
    _jac = jax.jacfwd(self._integrate_forward, argnums=[0, 1, 2])
    jac = jax.jit(jax.vmap(_jac, in_axes=(1, 1, 1), out_axes=(2, 2, 2)))
    return jac(nominal_states, nominal_controls, nominal_disturbances)
