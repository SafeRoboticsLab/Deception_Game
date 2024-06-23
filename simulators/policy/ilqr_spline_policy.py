# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Tuple, Optional, Dict
import time
import copy
import numpy as np
from jax import numpy as jnp
from jaxlib.xla_extension import DeviceArray

from .ilqr_policy import ILQR
from ..dynamics.base_dynamics import BaseDynamics
from ..cost.base_cost import BaseCost
from ..race_car.track import Track


class ILQRSpline(ILQR):

  def __init__(
      self, id: str, cfg, dyn: BaseDynamics, cost: BaseCost, track: Track,
      ref_traj: Optional[np.ndarray] = None, **kwargs
  ) -> None:
    super().__init__(id, cfg, dyn, cost)
    self.track = copy.deepcopy(track)
    self.update_ref_traj = getattr(cfg, "update_ref_traj", False)
    if self.update_ref_traj:
      assert ref_traj.ndim == 2 and ref_traj.shape[1] >= self.plan_horizon
      assert hasattr(cost, "ref_traj")
      self.update_ref_traj = True
      self.ref_traj = ref_traj.copy()

  def get_action(
      self, obs: np.ndarray, controls: Optional[np.ndarray] = None,
      agents_action: Optional[Dict] = None, **kwargs
  ) -> np.ndarray:
    status = 0

    if self.update_ref_traj:
      tmp = (self.horizon_indices + kwargs.get("time_idx")).reshape(-1)
      self.cost.ref_traj = jnp.asarray(self.ref_traj[:, tmp])

    # `controls` include control input at timestep N-1, which is a dummy
    # control of zeros.
    if controls is None:
      controls = jnp.zeros((self.dim_u, self.plan_horizon))
    else:
      assert controls.shape[1] == self.plan_horizon
      controls = jnp.array(controls)

    # Rolls out the nominal trajectory and gets the initial cost.
    # * This is differnet from the naive ILQR as it relies on the information
    # * from the pyspline.
    states, controls = self.rollout_nominal(
        jnp.array(kwargs.get('state')), controls
    )
    closest_pt, slope, theta = self.track.get_closest_pts(
        np.asarray(states[:2, :])
    )
    closest_pt = jnp.array(closest_pt)
    slope = jnp.array(slope)
    theta = jnp.array(theta)
    J = self.cost.get_traj_cost(
        states, controls, closest_pt, slope, theta,
        time_indices=self.horizon_indices
    )
    reg = self.reg_init
    fail_attempts = 0

    converged = False
    time0 = time.time()
    for i in range(self.max_iter):
      # We need cost derivatives from 0 to N-1, but we only need dynamics
      # jacobian from 0 to N-2.
      c_x, c_u, c_xx, c_uu, c_ux = self.cost.get_derivatives(
          states, controls, closest_pt, slope, theta,
          time_indices=self.horizon_indices
      )
      fx, fu = self.dyn.get_jacobian(states[:, :-1], controls[:, :-1])
      K_closed_loop, k_open_loop, reg = self.backward_pass(
          c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu,
          reg=reg
      )
      updated = False
      for alpha in self.alphas:
        X_new, U_new, J_new, closest_pt_new, slope_new, theta_new = (
            self.forward_pass(
                states, controls, K_closed_loop, k_open_loop, alpha
            )
        )

        if J_new < J:  # Improved!
          if np.abs((J-J_new) / J) < self.tol:  # Small improvement.
            converged = True

          # Updates nominal trajectory and best cost.
          J = J_new
          states = X_new
          controls = U_new
          closest_pt = closest_pt_new
          slope = slope_new
          theta = theta_new
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

    solver_info = dict(
        states=np.asarray(states), controls=np.asarray(controls),
        K_closed_loop=np.asarray(K_closed_loop),
        k_open_loop=np.asarray(k_open_loop), t_process=t_process,
        status=status, J=J, c_x=np.asarray(c_x), c_u=np.asarray(c_u),
        c_xx=np.asarray(c_xx), c_uu=np.asarray(c_uu), c_ux=np.asarray(c_ux),
        fx=np.asarray(fx), fu=np.asarray(fu)
    )
    return np.asarray(controls[:, 0]), solver_info

  def forward_pass(
      self, nominal_states: DeviceArray, nominal_controls: DeviceArray,
      K_closed_loop: DeviceArray, k_open_loop: DeviceArray, alpha: float
  ) -> Tuple[DeviceArray, DeviceArray, float, DeviceArray, DeviceArray,
             DeviceArray]:
    X, U = self.rollout(
        nominal_states, nominal_controls, K_closed_loop, k_open_loop, alpha
    )
    # * This is differnet from the naive ILQR as it relies on the information
    # * from the pyspline.
    closest_pt, slope, theta = self.track.get_closest_pts(np.asarray(X[:2, :]))
    closest_pt = jnp.array(closest_pt)
    slope = jnp.array(slope)
    theta = jnp.array(theta)
    J = self.cost.get_traj_cost(
        X, U, closest_pt, slope, theta, time_indices=self.horizon_indices
    )
    return X, U, J, closest_pt, slope, theta
