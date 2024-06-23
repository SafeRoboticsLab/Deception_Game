# ------------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: haiminh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# Modified from: https://github.com/SafeRoboticsLab/simulators
# ------------------------------------------------------------

"""A class for the human-AV interaction dynamics of the Uber example with Bayesian inference as the
learning dynamics.

This file implements a class for the human-AV interaction dynamics of the Uber example.
The state is represented by
    [`x`, `y`, `v`, `psi`, `delta`, `xH`, `yH`, `bP`, `bC`],
where (`x`, `y`) is the AV's position, `v` is the AV's forward speed, `psi` is the AV's heading
angle, `delta` is the AV's steering angle, (`xH`, `yH`) is the human's postion, `bP` is the
probability that the opponent is a pedestrian, and `bC` is the probability that the opponent is a
cyclist

The control is the AV's input including
    [`accel`, `omega`],
where `accel` is the linear acceleration and `omega` is the steering angular velocity.

The disturbance is the human's input including
    [`vxH`, `vyH`],
where `vxH` is the x-speed and `vyH` is the y-speed.
"""

import numpy as np
from scipy.special import gamma
from typing import Tuple, Any, Dict

from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp

from .base_dstb_dynamics import BaseDstbDynamics


class BeliefGameDynamics(BaseDstbDynamics):

  def __init__(self, cfg: Any, action_space: Dict[str, np.ndarray]) -> None:
    """
    Implements the bicycle dynamics (for Princeton race car). The state is the
    center of the rear axis. Bayesian inference is the learning dynamics.

    Args:
        cfg (Any): an object specifies cfguration.
        action_space (np.ndarray): action space.
    """
    super().__init__(cfg, action_space)
    self.dim_x = 9  # [x, y, v, psi, delta, xH, yH, bP, bC].

    self.clip_dstb_based_on_belief = True

    # load parameters
    self.wheelbase: float = cfg.wheelbase  # vehicle chassis length
    self.delta_min = cfg.delta_min
    self.delta_max = cfg.delta_max
    self.v_min = cfg.v_min
    self.v_max = cfg.v_max
    self.bel_thres = cfg.bel_thres  # type belief threshold, below which the type is discarded
    self._EPS = 1e-6  # small probability to avoid dividing by 0 in Bayes update
    self.dstb_scenario = jnp.asarray(cfg.dstb_scenario)

    # Likelihood models.
    self.p_ped = GMM(
        mix1=GenNorm(mu=cfg.mu_ped[0], alpha=cfg.alpha_ped, beta=cfg.beta_ped),
        mix2=GenNorm(mu=cfg.mu_ped[1], alpha=cfg.alpha_ped, beta=cfg.beta_ped), coeff=cfg.GMM_coeff
    )
    self.p_cyc = GMM(
        mix1=GenNorm(mu=cfg.mu_cyc[0], alpha=cfg.alpha_cyc, beta=cfg.beta_cyc),
        mix2=GenNorm(mu=cfg.mu_cyc[1], alpha=cfg.alpha_cyc, beta=cfg.beta_cyc), coeff=cfg.GMM_coeff
    )

  @partial(jax.jit, static_argnames='self')
  def integrate_forward_jax(
      self, state: DeviceArray, control: DeviceArray, disturbance: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:
    """Clips the control and disturbance and computes one-step time evolution
    of the system.

    Args:
                              0    1    2    3      4        5     6     7     8
        state (DeviceArray): [`x`, `y`, `v`, `psi`, `delta`, `xH`, `yH`, `bP`, `bC`].
        control (DeviceArray): [accel, omega]. Shape is (nu,)
        disturbance (DeviceArray): [`vxH`, `vyH`].

    Returns:
        DeviceArray: next state.
        DeviceArray: clipped control.
        DeviceArray: clipped disturbance.
    """

    @jax.jit
    def clip_dstb(disturbance, bound):
      return jnp.clip(disturbance, bound[:, 0], bound[:, 1])

    # Clips the controller values.
    ctrl_clip = jnp.clip(control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])

    if self.clip_dstb_based_on_belief:
      # Clips the disturbance values based on the belief.
      _belief = state[7:9]

      #   - 1. Obtains clipped dstb for all scenarios.
      _dstb_clip = jax.vmap(clip_dstb, in_axes=(None, 0))(disturbance, self.dstb_scenario)

      #   - 2. Computes differences b/w the clipped dstb and the original dstb.
      _norm_diff = jnp.linalg.norm(_dstb_clip - disturbance, axis=1)

      #   - 3. Set difference values to inf for low-probability scenarios.
      _norm_diff += jnp.inf * (_belief < self.bel_thres)
      # _norm_diff = _norm_diff.at[jnp.where(_belief < self.bel_thres)].set(jnp.inf)  # Cannot jit

      #   - 4. Pick the clipped dstb that is the closest to the original dstb.
      dstb_clip = _dstb_clip[jnp.argmin(_norm_diff), :]
    else:
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
    NOTE: This function only updates the physical states.
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
                            0  1  2  3    4      5   6   7   8
      state (DeviceArray): [x, y, v, psi, delta, xH, yH, bP, bC].
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
        state (DeviceArray): [x, y, v, psi, delta, xH, yH, bP, bC].
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
    RK4 integrator for the physical dynamics + Bayesian update.

    Args:
                              0  1  2  3    4      5   6   7   8
        state (DeviceArray): [x, y, v, psi, delta, xH, yH, bP, bC].
        control (DeviceArray): [accel, omega].
        disturbance (DeviceArray): [vxH, vyH].

    Returns:
        DeviceArray: next state.
    """
    # Integrates the physical states.
    k1 = self.disc_deriv(state, control, disturbance)
    k2 = self.disc_deriv(state + k1*dt/2, control, disturbance)
    k3 = self.disc_deriv(state + k2*dt/2, control, disturbance)
    k4 = self.disc_deriv(state + k3*dt, control, disturbance)
    state = state + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

    # Updates the belief.
    numer = self.p_ped.pdf_jax(disturbance[1]) * state[7] + self._EPS
    denom = (
        self.p_ped.pdf_jax(disturbance[1]) * state[7]
        + self.p_cyc.pdf_jax(disturbance[1]) * state[8] + self._EPS
    )

    bP_post = numer / denom
    bP_post = jnp.maximum(jnp.minimum(bP_post, 1. - self._EPS), self._EPS)

    state = state.at[7].set(bP_post)
    state = state.at[8].set(1. - bP_post)

    return state


class BeliefGameDynamicsWithIntents(BaseDstbDynamics):

  def __init__(self, cfg: Any, action_space: Dict[str, np.ndarray]) -> None:
    """
    Implements the bicycle dynamics (for Princeton race car). The state is the
    center of the rear axis. Bayesian inference is the learning dynamics.

    Args:
        cfg (Any): an object specifies cfguration.
        action_space (np.ndarray): action space.
    """
    super().__init__(cfg, action_space)

    # Intent hypotheses set.
    self.intent_set = np.array(cfg.intent_set)

    # System dimensions.
    self.dim_x = 9 + self.intent_set.shape[0]  # [x, y, v, psi, delta, xH, yH, bP, bC, [b_intent] ].
    self.num_intent = self.intent_set.shape[0]

    self.clip_dstb_based_on_belief = True

    # Loads parameters.
    self.wheelbase: float = cfg.wheelbase  # vehicle chassis length
    self.delta_min = cfg.delta_min
    self.delta_max = cfg.delta_max
    self.v_min = cfg.v_min
    self.v_max = cfg.v_max
    self.bel_thres = cfg.bel_thres  # type belief threshold, below which the type is discarded
    self.bel_thres_intent = cfg.bel_thres_intent  # intent belief threshold
    self._EPS = 1e-6  # small probability to avoid dividing by 0 in Bayes update
    self.dstb_scenario = jnp.asarray(cfg.dstb_scenario)
    self.K_vx = cfg.K_vx
    self.K_vy = cfg.K_vy
    self.eps_vx = cfg.eps_vx
    self.eps_vy = cfg.eps_vy

    # Likelihood models - type hypotheses.
    self.p_ped = GMM(
        mix1=GenNorm(mu=cfg.mu_ped[0], alpha=cfg.alpha_ped, beta=cfg.beta_ped),
        mix2=GenNorm(mu=cfg.mu_ped[1], alpha=cfg.alpha_ped, beta=cfg.beta_ped), coeff=cfg.GMM_coeff
    )
    self.p_cyc = GMM(
        mix1=GenNorm(mu=cfg.mu_cyc[0], alpha=cfg.alpha_cyc, beta=cfg.beta_cyc),
        mix2=GenNorm(mu=cfg.mu_cyc[1], alpha=cfg.alpha_cyc, beta=cfg.beta_cyc), coeff=cfg.GMM_coeff
    )

    # Likelihood models - intent hypotheses.
    self.p_intent = GenNorm(mu=0., alpha=cfg.alpha_intent, beta=cfg.beta_intent)

  @partial(jax.jit, static_argnames='self')
  def integrate_forward_jax(
      self, state: DeviceArray, control: DeviceArray, disturbance: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:
    """Clips the control and disturbance and computes one-step time evolution
    of the system.

    Args:
                              0    1    2    3      4        5     6     7     8     9, 10,...
        state (DeviceArray): [`x`, `y`, `v`, `psi`, `delta`, `xH`, `yH`, `bP`, `bC`, `b_intent`].
        control (DeviceArray): [accel, omega]. Shape is (nu,)
        disturbance (DeviceArray): [`vxH`, `vyH`].

    Returns:
        DeviceArray: next state.
        DeviceArray: clipped control.
        DeviceArray: clipped disturbance.
    """

    @jax.jit
    def clip_dstb_type(disturbance, bound):
      return jnp.clip(disturbance, bound[:, 0], bound[:, 1])

    @jax.jit
    def clip_dstb_intent(disturbance, pos_human, goal):
      _v_min = self.dstb_space[:, 0]
      _v_max = self.dstb_space[:, 1]
      _eps_v = jnp.array([self.eps_vx, self.eps_vy])
      nominal_dstb = jnp.array([
          self.K_vx * (pos_human[0] - goal[0]), self.K_vy * (pos_human[1] - goal[1])
      ])
      clip_min = jnp.maximum(_v_min, nominal_dstb - _eps_v)
      clip_max = jnp.minimum(_v_max, nominal_dstb + _eps_v)
      return jnp.clip(disturbance, clip_min, clip_max)

    # Clips the controller values.
    ctrl_clip = jnp.clip(control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])

    if self.clip_dstb_based_on_belief:
      # Clips the disturbance values based on the agent TYPE belief.
      _belief_type = state[7:9]

      #   - 1. Obtains clipped dstb for all possible types.
      _dstb_clip_type = jax.vmap(clip_dstb_type, in_axes=(None, 0))(disturbance, self.dstb_scenario)

      #   - 2. Computes differences b/w the clipped dstb and the original dstb.
      _norm_diff_type = jnp.linalg.norm(_dstb_clip_type - disturbance, axis=1)

      #   - 3. Sets difference values to inf for low-probability type hypotheses.
      _norm_diff_type += jnp.inf * (_belief_type < self.bel_thres)

      #   - 4. Picks the clipped dstb that is the closest to the original dstb.
      dstb_clip = _dstb_clip_type[jnp.argmin(_norm_diff_type), :]

      # Clips the disturbance values based on the agent INTENT belief.
      _belief_intent = state[9:]

      #   - 1. Obtains clipped dstb for all possible intents (goals).
      _v_clip = dstb_clip
      _v_clip = (
          jax.vmap(clip_dstb_intent, in_axes=(None, None, 0))(_v_clip, state[5:7], self.intent_set)
      )

      #   - 2. Computes differences b/w the clipped dstb and the original dstb.
      _norm_diff_intent = jnp.linalg.norm(_v_clip - dstb_clip, axis=1)

      #   - 3. Sets difference values to inf for low-probability intent hypotheses.
      _norm_diff_intent += jnp.inf * (_belief_intent < self.bel_thres_intent)

      #   - 4. Pickss the clipped dstb that is the closest to the original dstb.
      dstb_clip = _v_clip[jnp.argmin(_norm_diff_intent)]
    else:
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
    NOTE: This function only updates the physical states.
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
                            0    1    2    3      4        5     6     7     8     9, 10,...
      state (DeviceArray): [`x`, `y`, `v`, `psi`, `delta`, `xH`, `yH`, `bP`, `bC`, `b_intent`].
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
        state (DeviceArray): [x, y, v, psi, delta, xH, yH, bP, bC].
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
    RK4 integrator for the physical dynamics + Bayesian update.

    Args:
                              0    1    2    3      4        5     6     7     8     9, 10,...
        state (DeviceArray): [`x`, `y`, `v`, `psi`, `delta`, `xH`, `yH`, `bP`, `bC`, `b_intent`].
        control (DeviceArray): [accel, omega].
        disturbance (DeviceArray): [vxH, vyH].

    Returns:
        DeviceArray: next state.
    """

    @jax.jit
    def compute_likelihood(vxvy, pos_human, goal):
      nominal_vx = self.K_vx * (pos_human[0] - goal[0])
      nominal_vy = self.K_vy * (pos_human[1] - goal[1])
      p_goal_x = self.p_intent.pdf_jax(vxvy[0] - nominal_vx)
      p_goal_y = self.p_intent.pdf_jax(vxvy[1] - nominal_vy)
      return p_goal_x * p_goal_y

    # Integrates the physical states.
    k1 = self.disc_deriv(state, control, disturbance)
    k2 = self.disc_deriv(state + k1*dt/2, control, disturbance)
    k3 = self.disc_deriv(state + k2*dt/2, control, disturbance)
    k4 = self.disc_deriv(state + k3*dt, control, disturbance)
    state = state + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

    # Updates the type belief.
    numer = self.p_ped.pdf_jax(disturbance[1]) * state[7] + self._EPS
    denom = (
        self.p_ped.pdf_jax(disturbance[1]) * state[7]
        + self.p_cyc.pdf_jax(disturbance[1]) * state[8] + self._EPS
    )

    bP_post = numer / denom
    bP_post = jnp.maximum(jnp.minimum(bP_post, 1. - self._EPS), self._EPS)

    state = state.at[7].set(bP_post)
    state = state.at[8].set(1. - bP_post)

    # Updates the intent belief.
    likelihood = jax.vmap(compute_likelihood, in_axes=(None, None, 0))(
        disturbance,
        state[5:7],  # xH, yH
        self.intent_set,
    )
    bIntent_post = state[9:] * likelihood
    bIntent_post /= jnp.sum(bIntent_post)
    bIntent_post = jnp.clip(bIntent_post, jnp.zeros_like(bIntent_post), jnp.ones_like(bIntent_post))

    state = state.at[9:].set(bIntent_post)

    return state


class GenNorm():
  """
  Gaussian model.
  """

  def __init__(self, mu=0.0, alpha=1.0, beta=2.0) -> None:
    self.mu = mu
    self.alpha = alpha
    self.beta = beta
    self.coeff = self.beta / (2. * self.alpha * gamma(1. / self.beta))

  def pdf(self, x):
    exp_term = np.exp(-(np.abs(x - self.mu) / self.alpha)**self.beta)
    return self.coeff * exp_term

  @partial(jax.jit, static_argnames='self')
  def pdf_jax(self, x: DeviceArray) -> DeviceArray:
    exp_term = jnp.exp(-(jnp.abs(x - self.mu) / self.alpha)**self.beta)
    return self.coeff * exp_term


class GMM():
  """
  Gaussian mixture model.
  TODO: Generalize to more than 2 mixture components.
  """

  def __init__(self, mix1: GenNorm, mix2: GenNorm, coeff) -> None:
    self.mix1 = mix1
    self.mix2 = mix2
    self.coeff = coeff

  def pdf(self, x):
    return self.coeff[0] * self.mix1.pdf(x) + self.coeff[1] * self.mix2.pdf(x)

  @partial(jax.jit, static_argnames='self')
  def pdf_jax(self, x: DeviceArray) -> DeviceArray:
    return self.coeff[0] * self.mix1.pdf_jax(x) + self.coeff[1] * self.mix2.pdf_jax(x)


class IntentLikelihood():
  """
  Intent likelihood model. Modified from the Gaussian model.
  """

  def __init__(self, goal: float, K: float, alpha=1.0, beta=2.0) -> None:
    self.goal = goal
    self.K = K
    self.alpha = alpha
    self.beta = beta
    self.coeff = self.beta / (2. * self.alpha * gamma(1. / self.beta))

  def pdf(self, state, control):
    mu = self.K * (state - self.goal)
    exp_term = np.exp(-(np.abs(control - mu) / self.alpha)**self.beta)
    return self.coeff * exp_term
