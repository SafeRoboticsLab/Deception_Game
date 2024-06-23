# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Classes for quadratic costs.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jaxlib.xla_extension import DeviceArray
from functools import partial

from .base_cost import BaseCost


class QuadraticCost(BaseCost):

  def __init__(self, Q: np.ndarray, R: np.ndarray, S: np.ndarray, q: np.ndarray, r: np.ndarray):
    super().__init__()
    self.Q = jnp.array(Q)  # (n, n)
    self.R = jnp.array(R)  # (m, m)
    self.S = jnp.array(S)  # (m, n)
    self.q = jnp.array(q)  # (n,)
    self.r = jnp.array(r)  # (m,)

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    Qx = jnp.einsum("i,ni->n", state, self.Q)
    xtQx = jnp.einsum("n,n", state, Qx)
    Sx = jnp.einsum("n,mn->m", state, self.S)
    utSx = jnp.einsum("m,m", ctrl, Sx)
    Ru = jnp.einsum("i,mi->m", ctrl, self.R)
    utRu = jnp.einsum("m,m", ctrl, Ru)
    qtx = jnp.einsum("n,n", state, self.q)
    rtu = jnp.einsum("m,m", ctrl, self.r)
    return 0.5 * (xtQx+utRu) + utSx + qtx + rtu


class QuadraticControlCost(BaseCost):

  def __init__(self, R: np.ndarray, r: np.ndarray):
    super().__init__()
    self.R = jnp.array(R)  # (m, m)
    self.r = jnp.array(r)  # (m,)

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    Ru = jnp.einsum("i,mi->m", ctrl, self.R)
    utRu = jnp.einsum("m,m", ctrl, Ru)
    rtu = jnp.einsum("m,m", ctrl, self.r)
    return 0.5*utRu + rtu
