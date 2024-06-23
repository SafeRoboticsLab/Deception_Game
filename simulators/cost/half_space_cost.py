# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Classes for costs with respect to half spaces.

This file implements costs with repspect to half spaces. We consider the point
and box footprint.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jaxlib.xla_extension import DeviceArray
from functools import partial

from .base_cost import BaseCost


class UpperHalfCost(BaseCost):
  """
  c = `state`[dim] - `value`
  """

  def __init__(self, value: float, dim: int):
    super().__init__()
    self.dim = dim
    self.value = value

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    c = state[self.dim] - self.value
    return c


class LowerHalfCost(BaseCost):
  """
  c = `value` - `state`[dim]
  """

  def __init__(self, value: float, dim: int):
    super().__init__()
    self.dim = dim
    self.value = value

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    c = self.value - state[self.dim]
    return c


class UpperHalfBoxFootprintCost(BaseCost):

  def __init__(
      self, dim: str, value: float, state_box_limit: np.ndarray, x_dim: int = 0, y_dim: int = 1,
      yaw_dim: int = 3
  ):
    """
    Args:
        dim (str): 'x' ('y') for half space constraint on x (y) axis.
        value (float): the threshold.
        state_box_limit (np.ndarray): [`x_min`, `x_max`, `y_min`, `y_max`],
          vertices of the box footprint.
        box_spec (np.ndarray): [x, y, heading, half_length, half_width], spec
          of the box obstacles.
        x_dim (int): the index of x dimension. Defaults to 0.
        y_dim (int): the index of y dimension. Defaults to 1.
        yaw_dim (int): the index of yaw (heading) dimension. Defaults to 3.
    """
    super().__init__()
    if dim == 'x':
      self.dim = 0
      self.state_ret_dim = x_dim
    else:
      self.dim = 1
      self.state_ret_dim = y_dim

    self.value = value
    self.offset = jnp.array([[state_box_limit[0], state_box_limit[2]],
                             [state_box_limit[0], state_box_limit[3]],
                             [state_box_limit[1], state_box_limit[2]],
                             [state_box_limit[1], state_box_limit[3]]])
    self.yaw_dim = yaw_dim

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    yaw = state[self.yaw_dim]
    rot_mat = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw)], [jnp.sin(yaw), jnp.cos(yaw)]])
    rot_offset = jnp.einsum("ik,jk->ji", rot_mat, self.offset)  # (4, 2)
    pos = rot_offset[:, self.dim] + state[self.state_ret_dim]  # (4, )
    c = jnp.max(pos - self.value)
    return c


class LowerHalfBoxFootprintCost(BaseCost):

  def __init__(
      self, dim: str, value: float, state_box_limit: np.ndarray, x_dim: int = 0, y_dim: int = 1,
      yaw_dim: int = 3
  ):
    """
    Args:
        dim (str): 'x' ('y') for half space constraint on x (y) axis.
        value (float): the threshold.
        state_box_limit (np.ndarray): [`x_min`, `x_max`, `y_min`, `y_max`],
          vertices of the box footprint.
        box_spec (np.ndarray): [x, y, heading, half_length, half_width], spec
          of the box obstacles.
        x_dim (int): the index of x dimension. Defaults to 0.
        y_dim (int): the index of y dimension. Defaults to 1.
        yaw_dim (int): the index of yaw (heading) dimension. Defaults to 3.
    """
    super().__init__()
    if dim == 'x':
      self.dim = 0
      self.state_ret_dim = x_dim
    else:
      self.dim = 1
      self.state_ret_dim = y_dim
    self.value = value
    self.offset = jnp.array([[state_box_limit[0], state_box_limit[2]],
                             [state_box_limit[0], state_box_limit[3]],
                             [state_box_limit[1], state_box_limit[2]],
                             [state_box_limit[1], state_box_limit[3]]])
    self.yaw_dim = yaw_dim

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    yaw = state[self.yaw_dim]
    rot_mat = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw)], [jnp.sin(yaw), jnp.cos(yaw)]])
    rot_offset = jnp.einsum("ik,jk->ji", rot_mat, self.offset)  # (4, 2)
    pos = rot_offset[:, self.dim] + state[self.state_ret_dim]  # (4, )
    c = jnp.max(self.value - pos)
    return c
