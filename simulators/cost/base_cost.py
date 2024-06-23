# ------------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: haiminh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# Modified from: https://github.com/SafeRoboticsLab/simulators
# ------------------------------------------------------------

"""A parent class for costs.

This file implements a parent class for all costs and a barrier wrapper for
cost. A child class should implement `get_stage_cost()` and the parent class
takes care of the vectorizing map and derivatives functions.
"""

from abc import ABC, abstractmethod
from typing import Optional
from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp


class BaseCost(ABC):

  def __init__(self):
    super().__init__()

  @abstractmethod
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    raise NotImplementedError

  @partial(jax.jit, static_argnames='self')
  def get_cost(
      self, state: DeviceArray, ctrl: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    return jax.vmap(self.get_stage_cost, in_axes=(1, 1, 1))(state, ctrl, time_indices)

  @partial(jax.jit, static_argnames='self')
  def get_traj_cost(
      self, state: DeviceArray, ctrl: DeviceArray, time_indices: DeviceArray
  ) -> float:
    costs = jax.vmap(self.get_stage_cost, in_axes=(1, 1, 1))(state, ctrl, time_indices)
    return jnp.sum(costs).astype(float)

  @partial(jax.jit, static_argnames='self')
  def get_cx(self, state: DeviceArray, ctrl: DeviceArray, time_indices: DeviceArray) -> DeviceArray:
    _cx = jax.jacfwd(self.get_stage_cost, argnums=0)
    return jax.vmap(_cx, in_axes=(1, 1, 1), out_axes=1)(state, ctrl, time_indices)

  @partial(jax.jit, static_argnames='self')
  def get_cu(self, state: DeviceArray, ctrl: DeviceArray, time_indices: DeviceArray) -> DeviceArray:
    _cu = jax.jacfwd(self.get_stage_cost, argnums=1)
    return jax.vmap(_cu, in_axes=(1, 1, 1), out_axes=1)(state, ctrl, time_indices)

  @partial(jax.jit, static_argnames='self')
  def get_cxx(
      self, state: DeviceArray, ctrl: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    _cxx = jax.jacfwd(jax.jacrev(self.get_stage_cost, argnums=0), argnums=0)
    return jax.vmap(_cxx, in_axes=(1, 1, 1), out_axes=2)(state, ctrl, time_indices)

  @partial(jax.jit, static_argnames='self')
  def get_cuu(
      self, state: DeviceArray, ctrl: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    _cuu = jax.jacfwd(jax.jacrev(self.get_stage_cost, argnums=1), argnums=1)
    return jax.vmap(_cuu, in_axes=(1, 1, 1), out_axes=2)(state, ctrl, time_indices)

  @partial(jax.jit, static_argnames='self')
  def get_cux(
      self, state: DeviceArray, ctrl: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    _cux = jax.jacfwd(jax.jacrev(self.get_stage_cost, argnums=1), argnums=0)
    return jax.vmap(_cux, in_axes=(1, 1, 1), out_axes=2)(state, ctrl, time_indices)

  @partial(jax.jit, static_argnames='self')
  def get_cxu(
      self, state: DeviceArray, ctrl: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    return self.get_cux(state, ctrl, time_indices).T

  @partial(jax.jit, static_argnames='self')
  def get_derivatives(
      self, state: DeviceArray, ctrl: DeviceArray, time_indices: DeviceArray
  ) -> DeviceArray:
    return (
        self.get_cx(state, ctrl, time_indices),
        self.get_cu(state, ctrl, time_indices),
        self.get_cxx(state, ctrl, time_indices),
        self.get_cuu(state, ctrl, time_indices),
        self.get_cux(state, ctrl, time_indices),
    )


class ProximityCost(BaseCost):

  def __init__(self, max_distance, position_indices):
    """
    Proximity cost defined in the joint state space. Penalizes
      ```min(distance(pos_1, pos_2) - max_distance, 0)^2 ```
    For jit compatibility, number of players is hardcoded to 2 to avoid loops.
    """
    super().__init__()
    self._max_distance = max_distance
    self._position_indices = position_indices

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    """
    Evaluates this cost function on the given input state and/or control.

    Args:
        x (DeviceArray): concatenated state vector of all subsystems (nx,)
        ui (DeviceArray, optional): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        DeviceArray: scalar value of cost (scalar)
    """
    # Players' positional state indices.
    x1_idx = self._position_indices.x1
    y1_idx = self._position_indices.y1
    x2_idx = self._position_indices.x2
    y2_idx = self._position_indices.y2

    # Relative distance between Player 1 and Player 2.
    rel_dist = jnp.sqrt((state[x1_idx] - state[x2_idx])**2 + (state[y1_idx] - state[y2_idx])**2)
    cost = self._max_distance - rel_dist
    # cost = -jnp.minimum(rel_dist - self._max_distance, 0.)
    # cost = jnp.minimum(rel_dist - self._max_distance, 0.)**2
    return cost


class BarrierCost(BaseCost):

  def __init__(
      self, clip_min: Optional[float], clip_max: Optional[float], q1: float, q2: float,
      cost: BaseCost
  ):
    super().__init__()
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.q1 = q1
    self.q2 = q2
    self.cost = cost

  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    """
    Gets the stage cost.
    """
    _cost = self.cost.get_stage_cost(state, ctrl, time_idx)
    return self.q1 * jnp.exp(self.q2 * jnp.clip(a=_cost, a_min=self.clip_min, a_max=self.clip_max))
