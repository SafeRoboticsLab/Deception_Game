# ------------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: haiminh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# Modified from: https://github.com/SafeRoboticsLab/simulators
# ------------------------------------------------------------

"""A class for the human-AV interaction dynamics of the Uber example.

This file implements a class for the human-AV interaction dynamics of the Uber example.
The state is represented by
    [`x`, `y`, `v`, `psi`, `delta`, `xH`, `yH`],
where (`x`, `y`) is the AV's position, `v` is the AV's forward speed, `psi` is the AV's heading
angle, `delta` is the AV's steering angle, and (`xH`, `yH`) is the human's postion.

The control is the AV's input including
    [`accel`, `omega`],
where `accel` is the linear acceleration and `omega` is the steering angular velocity.

The disturbance is the human's input including
    [`vxH`, `vyH`],
where `vxH` is the x-speed and `vyH` is the y-speed.
"""

from typing import Tuple, Any, Dict
import numpy as np
from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp

from .base_dstb_dynamics import BaseDstbDynamics


class UberNoBelief(BaseDstbDynamics):

  def __init__(self, cfg: Any, action_space: Dict[str, np.ndarray]) -> None:
    """
    Implements the bicycle dynamics (for Princeton race car). The state is the
    center of the rear axis.

    Args:
        cfg (Any): an object specifies cfguration.
        action_space (np.ndarray): action space.
    """
    super().__init__(cfg, action_space)
    self.dim_x = 7  # [x, y, v, psi, delta, xH, yH].

    # load parameters
    self.wheelbase: float = cfg.wheelbase  # vehicle chassis length
    self.delta_min = cfg.delta_min
    self.delta_max = cfg.delta_max
    self.v_min = cfg.v_min
    self.v_max = cfg.v_max

  @partial(jax.jit, static_argnames='self')
  def integrate_forward_jax(
      self, state: DeviceArray, control: DeviceArray, disturbance: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:
    """Clips the control and disturbance and computes one-step time evolution
    of the system.

    Args:
        state (DeviceArray): [`x`, `y`, `v`, `psi`, `delta`, `xH`, `yH`].
        control (DeviceArray): [accel, omega].
        disturbance (DeviceArray): [`vxH`, `vyH`].

    Returns:
        DeviceArray: next state.
        DeviceArray: clipped control.
        DeviceArray: clipped disturbance.
    """
    # Clips the controller values between min and max accel and steer values.
    ctrl_clip = jnp.clip(control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])
    dstb_clip = jnp.clip(disturbance, self.dstb_space[:, 0], self.dstb_space[:, 1])

    @jax.jit
    def crit_delta(args):
      c_delta, c_vel, c_flag_vel = args

      def crit_delta_vel(args):
        condx = c_vel < c_delta

        def vel_then_delta(args):
          state_tmp1 = self._integrate_forward_dt(state, ctrl_clip, dstb_clip, c_vel)
          state_tmp2 = self._integrate_forward_dt(
              state_tmp1, jnp.array([0., ctrl_clip[1]]), dstb_clip, c_delta - c_vel
          )
          return self._integrate_forward_dt(state_tmp2, jnp.zeros(2), dstb_clip, self.dt - c_delta)

        def delta_then_vel(args):
          state_tmp1 = self._integrate_forward_dt(state, ctrl_clip, dstb_clip, c_delta)
          state_tmp2 = self._integrate_forward_dt(
              state_tmp1, jnp.array([ctrl_clip[0], 0.]), dstb_clip, c_vel - c_delta
          )
          return self._integrate_forward_dt(state_tmp2, jnp.zeros(2), dstb_clip, self.dt - c_vel)

        return jax.lax.cond(condx, vel_then_delta, delta_then_vel, (c_delta, c_vel))

      def crit_delta_only(args):
        state_tmp = self._integrate_forward_dt(state, ctrl_clip, dstb_clip, c_delta)
        return self._integrate_forward_dt(
            state_tmp, jnp.array([ctrl_clip[0], 0.]), dstb_clip, self.dt - c_delta
        )

      return jax.lax.cond(c_flag_vel, crit_delta_vel, crit_delta_only, (c_delta, c_vel))

    @jax.jit
    def non_crit_delta(args):
      _, c_vel, c_flag_vel = args

      def crit_vel_only(args):
        state_tmp = self._integrate_forward_dt(state, ctrl_clip, dstb_clip, c_vel)
        return self._integrate_forward_dt(
            state_tmp, jnp.array([0., ctrl_clip[1]]), dstb_clip, self.dt - c_vel
        )

      def non_crit(args):
        return self._integrate_forward_dt(state, ctrl_clip, dstb_clip, self.dt)

      return jax.lax.cond(c_flag_vel, crit_vel_only, non_crit, (c_vel))

    c_vel, c_flag_vel = self.get_crit(state[2], self.v_min, self.v_max, ctrl_clip[0], self.dt)
    c_delta, c_flag_delta = self.get_crit(
        state[4], self.delta_min, self.delta_max, ctrl_clip[1], self.dt
    )
    state_nxt = jax.lax.cond(c_flag_delta, crit_delta, non_crit_delta, (c_delta, c_vel, c_flag_vel))
    state_nxt = state_nxt.at[3].set(jnp.mod(state_nxt[3] + jnp.pi, 2 * jnp.pi) - jnp.pi)
    # ! hacky
    state_nxt = state_nxt.at[2].set(jnp.clip(state_nxt[2], self.v_min, self.v_max))
    state_nxt = state_nxt.at[4].set(jnp.clip(state_nxt[4], self.delta_min, self.delta_max))
    return state_nxt, ctrl_clip, dstb_clip

  @partial(jax.jit, static_argnames='self')
  def get_crit(self, state_var, value_min, value_max, ctrl, dt) -> Tuple[float, bool]:
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
      self, state: DeviceArray, control: DeviceArray, disturbance: DeviceArray
  ) -> DeviceArray:
    """
    Right-hand side of the continuous-time dynamics.
    The continuous-time dynamics are as below
      x_dot = v cos(psi)
      y_dot = v sin(psi)
      v_dot = u0
      psi_dot = v tan(delta) / L
      delta_dot = u1
      xH_dot = d0
      yH_dot = d1

    Args:
                            0  1  2  3    4      5   6
      state (DeviceArray): [x, y, v, psi, delta, xH, yH].
      control (DeviceArray): [accel, omega].
      disturbance (DeviceArray): [vxH, vyH].

    Returns:
      DeviceArray: time deriv. of the state.
    """
    deriv = jnp.zeros((self.dim_x,))

    deriv = deriv.at[0].set(state[2] * jnp.cos(state[3]))
    deriv = deriv.at[1].set(state[2] * jnp.sin(state[3]))
    deriv = deriv.at[2].set(control[0])
    deriv = deriv.at[3].set(state[2] * jnp.tan(state[4]) / self.wheelbase)
    deriv = deriv.at[4].set(control[1])
    deriv = deriv.at[5].set(disturbance[0])
    deriv = deriv.at[6].set(disturbance[1])
    return deriv

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward(
      self, state: DeviceArray, control: DeviceArray, disturbance: DeviceArray
  ) -> DeviceArray:
    """
    Computes one-step time evolution of the system: x_+ = f(x, u, d).

    Args:
        state (DeviceArray): [x, y, v, psi, delta, xH, yH].
        control (DeviceArray): [accel, omega].
        disturbance (DeviceArray): [vxH, vyH].

    Returns:
        DeviceArray: next state.
    """
    return self._integrate_forward_dt(state, control, disturbance, self.dt)

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward_dt(
      self, state: DeviceArray, control: DeviceArray, disturbance: DeviceArray, dt: float
  ) -> DeviceArray:
    """
    RK4 integrator.
    """
    k1 = self.disc_deriv(state, control, disturbance)
    k2 = self.disc_deriv(state + k1*dt/2, control, disturbance)
    k3 = self.disc_deriv(state + k2*dt/2, control, disturbance)
    k4 = self.disc_deriv(state + k3*dt, control, disturbance)
    return state + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
