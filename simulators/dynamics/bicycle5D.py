# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for kinematic bicycle with additive disturbance.

This file implements a class for kinematic bicycle. The state is represented by
[`x`, `y`, `v`, `psi`, `delta`], where (`x`, `y`) is the position, `v` is the
forward speed, `psi` is the heading angle, and `delta` is the steering angle.
The control is [`accel`, `omega`], where `accel` is the linear acceleration and
`omega` is the steering angular velocity.
"""

from typing import Tuple, Any
import numpy as np
from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp

from .base_dynamics import BaseDynamics


class Bicycle5D(BaseDynamics):

  def __init__(self, cfg: Any, action_space: np.ndarray) -> None:
    """
    Implements the bicycle dynamics (for Princeton race car). The state is the
    center of the rear axis.

    Args:
        cfg (Any): an object specifies configuration.
        action_space (np.ndarray): action space.
    """
    super().__init__(cfg, action_space)
    self.dim_x = 5  # [x, y, v, psi, delta].

    # load parameters
    self.wheelbase: float = cfg.wheelbase  # vehicle chassis length
    self.delta_min = cfg.delta_min
    self.delta_max = cfg.delta_max
    self.v_min = cfg.v_min
    self.v_max = cfg.v_max

  @partial(jax.jit, static_argnames='self')
  def integrate_forward_jax(
      self, state: DeviceArray, control: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:
    """Clips the control and computes one-step time evolution of the system.

    Args:
        state (DeviceArray): [x, y, v, psi, delta].
        control (DeviceArray): [accel, omega].

    Returns:
        DeviceArray: next state.
        DeviceArray: clipped control.
    """
    # Clips the controller values between min and max accel and steer values.
    ctrl_clip = jnp.clip(control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])

    @jax.jit
    def crit_delta(args):
      c_delta, c_vel, c_flag_vel = args

      def crit_delta_vel(args):
        condx = c_vel < c_delta

        def vel_then_delta(args):
          state_tmp1 = self._integrate_forward_dt(state, ctrl_clip, c_vel)
          state_tmp2 = self._integrate_forward_dt(
              state_tmp1, jnp.array([0., ctrl_clip[1]]), c_delta - c_vel
          )
          return self._integrate_forward_dt(
              state_tmp2, jnp.zeros(2), self.dt - c_delta
          )

        def delta_then_vel(args):
          state_tmp1 = self._integrate_forward_dt(state, ctrl_clip, c_delta)
          state_tmp2 = self._integrate_forward_dt(
              state_tmp1, jnp.array([ctrl_clip[0], 0.]), c_vel - c_delta
          )
          return self._integrate_forward_dt(
              state_tmp2, jnp.zeros(2), self.dt - c_vel
          )

        return jax.lax.cond(
            condx, vel_then_delta, delta_then_vel, (c_delta, c_vel)
        )

      def crit_delta_only(args):
        state_tmp = self._integrate_forward_dt(state, ctrl_clip, c_delta)
        return self._integrate_forward_dt(
            state_tmp, jnp.array([ctrl_clip[0], 0.]), self.dt - c_delta
        )

      return jax.lax.cond(
          c_flag_vel, crit_delta_vel, crit_delta_only, (c_delta, c_vel)
      )

    @jax.jit
    def non_crit_delta(args):
      _, c_vel, c_flag_vel = args

      def crit_vel_only(args):
        state_tmp = self._integrate_forward_dt(state, ctrl_clip, c_vel)
        return self._integrate_forward_dt(
            state_tmp, jnp.array([0., ctrl_clip[1]]), self.dt - c_vel
        )

      def non_crit(args):
        return self._integrate_forward_dt(state, ctrl_clip, self.dt)

      return jax.lax.cond(c_flag_vel, crit_vel_only, non_crit, (c_vel))

    c_vel, c_flag_vel = self.get_crit(
        state[2], self.v_min, self.v_max, ctrl_clip[0], self.dt
    )
    c_delta, c_flag_delta = self.get_crit(
        state[4], self.delta_min, self.delta_max, ctrl_clip[1], self.dt
    )
    state_nxt = jax.lax.cond(
        c_flag_delta, crit_delta, non_crit_delta, (c_delta, c_vel, c_flag_vel)
    )
    state_nxt = state_nxt.at[3].set(
        jnp.mod(state_nxt[3] + jnp.pi, 2 * jnp.pi) - jnp.pi
    )
    # ! hacky
    state_nxt = state_nxt.at[2].set(
        jnp.clip(state_nxt[2], self.v_min, self.v_max)
    )
    state_nxt = state_nxt.at[4].set(
        jnp.clip(state_nxt[4], self.delta_min, self.delta_max)
    )
    return state_nxt, ctrl_clip

  @partial(jax.jit, static_argnames='self')
  def get_crit(self, state_var, value_min, value_max, ctrl,
               dt) -> Tuple[float, bool]:
    crit1 = (value_max-state_var) / (ctrl+1e-8)
    crit2 = (value_min-state_var) / (ctrl+1e-8)
    crit_flag1 = jnp.logical_and(crit1 < dt, crit1 > 0.)
    crit_flag2 = jnp.logical_and(crit2 < dt, crit2 > 0.)
    crit_flag = jnp.logical_or(crit_flag1, crit_flag2)

    def true_func(args):
      crit1, crit2 = args
      return crit1

    def false_func(args):
      crit1, crit2 = args
      return crit2

    # crit_time should be ignored when crit_flag is False.
    crit_time = jax.lax.cond(crit_flag1, true_func, false_func, (crit1, crit2))
    return crit_time, crit_flag

  @partial(jax.jit, static_argnames='self')
  def disc_deriv(
      self, state: DeviceArray, control: DeviceArray
  ) -> DeviceArray:
    deriv = jnp.zeros((self.dim_x,))
    deriv = deriv.at[0].set(state[2] * jnp.cos(state[3]))
    deriv = deriv.at[1].set(state[2] * jnp.sin(state[3]))
    deriv = deriv.at[2].set(control[0])
    deriv = deriv.at[3].set(state[2] * jnp.tan(state[4]) / self.wheelbase)
    deriv = deriv.at[4].set(control[1])
    return deriv

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward(
      self, state: DeviceArray, control: DeviceArray
  ) -> DeviceArray:
    """ Computes one-step time evolution of the system: x_+ = f(x, u).
    The discrete-time dynamics is as below:
        x_k+1 = x_k + v_k cos(psi_k) dt
        y_k+1 = y_k + v_k sin(psi_k) dt
        v_k+1 = v_k + u0_k dt
        psi_k+1 = psi_k + v_k tan(delta_k) / L dt
        delta_k+1 = delta_k + u1_k dt

    Args:
        state (DeviceArray): [x, y, v, psi, delta].
        control (DeviceArray): [accel, omega].

    Returns:
        DeviceArray: next state.
    """
    # @jax.jit
    # def _fwd_step(i, args):  # Euler method.
    #   _state, _ctrl = args
    #   return self._integrate_forward_dt(_state, _ctrl, self.int_dt), _ctrl

    # state_nxt = jax.lax.fori_loop(
    #     0, self.num_segment, _fwd_step, (state, control)
    # )[0]
    # return state_nxt
    return self._integrate_forward_dt(state, control, self.dt)

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward_dt(
      self, state: DeviceArray, control: DeviceArray, dt: float
  ) -> DeviceArray:
    k1 = self.disc_deriv(state, control)
    k2 = self.disc_deriv(state + k1*dt/2, control)
    k3 = self.disc_deriv(state + k2*dt/2, control)
    k4 = self.disc_deriv(state + k3*dt, control)
    return state + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
