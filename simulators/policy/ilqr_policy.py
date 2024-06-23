# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Tuple, Optional, Dict
import time
import copy
import numpy as np
import jax
from jax import numpy as jnp
from jaxlib.xla_extension import DeviceArray
from functools import partial

from .base_policy import BasePolicy
from ..dynamics.base_dynamics import BaseDynamics
from ..cost.base_cost import BaseCost


class ILQR(BasePolicy):

  def __init__(
      self, id: str, cfg, dyn: BaseDynamics, cost: BaseCost, **kwargs
  ) -> None:
    super().__init__(id, cfg)
    self.policy_type = "ILQR"
    self.dyn = copy.deepcopy(dyn)
    self.cost = copy.deepcopy(cost)

    # ILQR parameters
    self.dim_x = dyn.dim_x
    self.dim_u = dyn.dim_u
    self.plan_horizon = int(cfg.plan_horizon)
    self.max_iter = int(cfg.max_iter)
    self.tol = float(cfg.tol)  # ILQR update tolerance.

    # regularization parameters
    self.reg_min = float(cfg.reg_min)
    self.reg_max = float(cfg.reg_max)
    self.reg_init = float(cfg.reg_init)
    self.reg_scale_up = float(cfg.reg_scale_up)
    self.reg_scale_down = float(cfg.reg_scale_down)
    self.max_attempt = int(cfg.max_attempt)

    # TODO: Other line search methods
    self.alphas = 0.5**(np.arange(25))
    self.horizon_indices = jnp.arange(self.plan_horizon).reshape(1, -1)

  def get_action(
      self, obs: np.ndarray, controls: Optional[np.ndarray] = None,
      agents_action: Optional[Dict] = None, **kwargs
  ) -> np.ndarray:
    status = 0

    # `controls` include control input at timestep N-1, which is a dummy
    # control of zeros.
    if controls is None:
      controls = jnp.zeros((self.dim_u, self.plan_horizon))
    else:
      assert controls.shape[1] == self.plan_horizon
      controls = jnp.array(controls)

    # Rolls out the nominal trajectory and gets the initial cost.
    states, controls = self.rollout_nominal(
        jnp.array(kwargs.get('state')), controls
    )
    J = self.cost.get_traj_cost(
        states, controls, time_indices=self.horizon_indices
    )
    reg = self.reg_init
    fail_attempts = 0

    converged = False
    time0 = time.time()
    for i in range(self.max_iter):
      # We need cost derivatives from 0 to N-1, but we only need dynamics
      # jacobian from 0 to N-2.
      c_x, c_u, c_xx, c_uu, c_ux = self.cost.get_derivatives(
          states, controls, time_indices=self.horizon_indices
      )
      fx, fu = self.dyn.get_jacobian(states[:, :-1], controls[:, :-1])
      K_closed_loop, k_open_loop, reg = self.backward_pass(
          c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu,
          reg=reg
      )
      updated = False
      for alpha in self.alphas:
        X_new, U_new, J_new = self.forward_pass(
            states, controls, K_closed_loop, k_open_loop, alpha
        )

        if J_new <= J:  # Improved!
          if np.abs((J-J_new) / J) < self.tol:  # Small improvement.
            converged = True

          # Updates nominal trajectory and best cost.
          J = J_new
          states = X_new
          controls = U_new
          updated = True
          reg = max(self.reg_min, reg / self.reg_scale_down)
          break

      # Terminates early if the line search fails and reg >= reg_max.
      if not updated:
        reg = reg * self.reg_scale_up
        fail_attempts += 1
        if fail_attempts > self.max_attempt or reg > self.reg_max:
          status = 2
          break

      # Terminates early if the objective improvement is negligible.
      if converged:
        status = 1
        break
    t_process = time.time() - time0

    states = np.asarray(states)
    controls = np.asarray(controls)
    K_closed_loop = np.asarray(K_closed_loop)
    k_open_loop = np.asarray(k_open_loop)
    solver_info = dict(
        states=states, controls=controls, K_closed_loop=K_closed_loop,
        k_open_loop=k_open_loop, t_process=t_process, status=status, J=J
    )
    return controls[:, 0], solver_info

  @partial(jax.jit, static_argnames='self')
  def forward_pass(
      self, nominal_states: DeviceArray, nominal_controls: DeviceArray,
      K_closed_loop: DeviceArray, k_open_loop: DeviceArray, alpha: float
  ) -> Tuple[DeviceArray, DeviceArray, float]:
    # We seperate the rollout and cost explicitly since get_cost might rely on
    # other information, such as env parameters (track), and is difficult for
    # jax to differentiate.
    X, U = self.rollout(
        nominal_states, nominal_controls, K_closed_loop, k_open_loop, alpha
    )
    J = self.cost.get_traj_cost(X, U, time_indices=self.horizon_indices)
    return X, U, J

  @partial(jax.jit, static_argnames='self')
  def rollout(
      self, nominal_states: DeviceArray, nominal_controls: DeviceArray,
      K_closed_loop: DeviceArray, k_open_loop: DeviceArray, alpha: float
  ) -> Tuple[DeviceArray, DeviceArray]:

    @jax.jit
    def _rollout_step(i, args):
      X, U = args
      u_fb = jnp.einsum(
          "ik,k->i", K_closed_loop[:, :, i], (X[:, i] - nominal_states[:, i])
      )
      u = nominal_controls[:, i] + alpha * k_open_loop[:, i] + u_fb
      x_nxt, u_clip = self.dyn.integrate_forward_jax(X[:, i], u)
      X = X.at[:, i + 1].set(x_nxt)
      U = U.at[:, i].set(u_clip)
      return X, U

    X = jnp.zeros((self.dim_x, self.plan_horizon))
    U = jnp.zeros((self.dim_u, self.plan_horizon)
                 )  # Assumes the last ctrl are zeros.
    X = X.at[:, 0].set(nominal_states[:, 0])

    X, U = jax.lax.fori_loop(0, self.plan_horizon - 1, _rollout_step, (X, U))
    return X, U

  @partial(jax.jit, static_argnames='self')
  def rollout_nominal(
      self, initial_state: DeviceArray, controls: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:

    @jax.jit
    def _rollout_nominal_step(i, args):
      X, U = args
      x_nxt, u_clip = self.dyn.integrate_forward_jax(X[:, i], U[:, i])
      X = X.at[:, i + 1].set(x_nxt)
      U = U.at[:, i].set(u_clip)
      return X, U

    X = jnp.zeros((self.dim_x, self.plan_horizon))
    X = X.at[:, 0].set(initial_state)
    X, U = jax.lax.fori_loop(
        0, self.plan_horizon - 1, _rollout_nominal_step, (X, controls)
    )
    return X, U

  @partial(jax.jit, static_argnames='self')
  def backward_pass(
      self, c_x: DeviceArray, c_u: DeviceArray, c_xx: DeviceArray,
      c_uu: DeviceArray, c_ux: DeviceArray, fx: DeviceArray, fu: DeviceArray,
      reg: float
  ) -> Tuple[DeviceArray, DeviceArray, float]:
    """
    Jitted backward pass looped computation.

    Args:
        c_x (DeviceArray): (dim_x, N)
        c_u (DeviceArray): (dim_u, N)
        c_xx (DeviceArray): (dim_x, dim_x, N)
        c_uu (DeviceArray): (dim_u, dim_u, N)
        c_ux (DeviceArray): (dim_u, dim_x, N)
        fx (DeviceArray): (dim_x, dim_x, N-1)
        fu (DeviceArray): (dim_x, dim_u, N-1)

    Returns:
        Ks (DeviceArray): gain matrices (dim_u, dim_x, N - 1)
        ks (DeviceArray): gain vectors (dim_u, N - 1)
    """

    def init():
      Ks = jnp.zeros((self.dim_u, self.dim_x, self.plan_horizon - 1))
      ks = jnp.zeros((self.dim_u, self.plan_horizon - 1))
      V_x = c_x[:, -1]
      V_xx = c_xx[:, :, -1]
      return V_x, V_xx, ks, Ks

    @jax.jit
    def backward_pass_looper(val):
      V_x, V_xx, ks, Ks, t, reg = val
      Q_x = c_x[:, t] + fx[:, :, t].T @ V_x
      Q_u = c_u[:, t] + fu[:, :, t].T @ V_x
      Q_xx = c_xx[:, :, t] + fx[:, :, t].T @ V_xx @ fx[:, :, t]
      Q_ux = c_ux[:, :, t] + fu[:, :, t].T @ V_xx @ fx[:, :, t]
      Q_uu = c_uu[:, :, t] + fu[:, :, t].T @ V_xx @ fu[:, :, t]

      # The regularization is added to Vxx for robustness.
      # Ref: http://roboticexplorationlab.org/papers/ILQR_Tutorial.pdf
      reg_mat = reg * jnp.eye(self.dim_x)
      V_xx_reg = V_xx + reg_mat
      Q_ux_reg = c_ux[:, :, t] + fu[:, :, t].T @ V_xx_reg @ fx[:, :, t]
      Q_uu_reg = c_uu[:, :, t] + fu[:, :, t].T @ V_xx_reg @ fu[:, :, t]

      @jax.jit
      def isposdef(fx, reg):
        # If the regularization is too large, but the matrix is still not
        # positive definite, we will let the backward pass continue to avoid
        # infinite loop.
        return (jnp.all(jnp.linalg.eigvalsh(fx) > 0) | (reg >= self.reg_max))

      @jax.jit
      def false_func(val):
        V_x, V_xx, ks, Ks = init()
        updated_reg = self.reg_scale_up * reg
        updated_reg = jax.lax.cond(
            updated_reg <= self.reg_max, lambda x: x, lambda x: self.reg_max,
            updated_reg
        )
        return V_x, V_xx, ks, Ks, self.plan_horizon - 2, updated_reg

      @jax.jit
      def true_func(val):
        Ks, ks = val
        Q_uu_reg_inv = jnp.linalg.inv(Q_uu_reg)

        Ks = Ks.at[:, :, t].set(-Q_uu_reg_inv @ Q_ux_reg)
        ks = ks.at[:, t].set(-Q_uu_reg_inv @ Q_u)

        V_x = (
            Q_x + Ks[:, :, t].T @ Q_u + Q_ux.T @ ks[:, t]
            + Ks[:, :, t].T @ Q_uu @ ks[:, t]
        )
        V_xx = (
            Q_xx + Ks[:, :, t].T @ Q_ux + Q_ux.T @ Ks[:, :, t]
            + Ks[:, :, t].T @ Q_uu @ Ks[:, :, t]
        )
        return V_x, V_xx, ks, Ks, t - 1, reg

      return jax.lax.cond(
          isposdef(Q_uu_reg, reg), true_func, false_func, (Ks, ks)
      )

    @jax.jit
    def cond_fun(val):
      _, _, _, _, t, _ = val
      return t >= 0

    V_x, V_xx, ks, Ks = init()  # Initializes.
    V_x, V_xx, ks, Ks, t, reg = jax.lax.while_loop(
        cond_fun, backward_pass_looper,
        (V_x, V_xx, ks, Ks, self.plan_horizon - 2, reg)
    )
    return Ks, ks, reg
