# ------------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: haiminh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# Modified from: https://github.com/SafeRoboticsLab/simulators
# ------------------------------------------------------------

"""Classes for costs and constraints used in the race car simulator.
"""

from typing import Dict, List
from functools import partial
from jaxlib.xla_extension import DeviceArray
import numpy as np
import jax.numpy as jnp
import jax

from ..cost.base_cost import BaseCost, BarrierCost, ProximityCost
from ..cost.box_cost import BoxObsBoxFootprintCost
from ..cost.half_space_cost import LowerHalfCost, UpperHalfCost
from ..cost.spline_cost import (
    BaseSplineCost, SplineBarrierCost, SplineRoadBoundaryCost, SplineWideningRoadBoundaryCost,
    SplineYawCost
)


class UberExampleCost(BaseSplineCost):

  def __init__(self, cfg):
    super().__init__(cfg)

    # Performance cost parameters.
    self.v_ref: float = cfg.v_ref  # reference velocity.
    self.w_vel: float = cfg.w_vel
    self.w_contour: float = cfg.w_contour
    self.w_theta: float = cfg.w_theta
    self.w_accel: float = cfg.w_accel
    self.w_omega: float = cfg.w_omega
    self.wheelbase: float = cfg.wheelbase
    self.track_offset: float = cfg.track_offset

    # Soft constraint parameters.
    self.q1_yaw: float = cfg.q1_yaw
    self.q2_yaw: float = cfg.q2_yaw
    self.q1_road: float = cfg.q1_road
    self.q2_road: float = cfg.q2_road
    self.q1_obs: float = cfg.q1_obs
    self.q2_obs: float = cfg.q2_obs
    self.yaw_min: float = cfg.yaw_min
    self.yaw_max: float = cfg.yaw_max

    self.obs_spec = [np.asarray(x) for x in cfg.obs_spec]
    self.obs_precision = list(getattr(cfg, "obs_precision", [31, 11]))
    self.barrier_clip_min: float = cfg.barrier_clip_min
    self.barrier_clip_max: float = cfg.barrier_clip_max
    self.buffer: float = getattr(cfg, "buffer", 0.)
    self.yaw_barrier_cost = SplineBarrierCost(
        self.barrier_clip_min, self.barrier_clip_max, self.q1_yaw, self.q2_yaw, SplineYawCost(cfg)
    )
    self.road_barrier_cost = SplineBarrierCost(
        self.barrier_clip_min, self.barrier_clip_max, self.q1_road, self.q2_road,
        SplineWideningRoadBoundaryCost(cfg=cfg, buffer=self.buffer)
    )
    self.collision_barrier_cost = BarrierCost(
        self.barrier_clip_min, self.barrier_clip_max, self.q1_obs, self.q2_obs,
        ProximityCost(cfg.collision_dist, cfg.position_indices)
    )

    self.has_obs_constr: bool = cfg.has_obs_constr
    if self.has_obs_constr:
      self.obs_barrier_cost: List[BarrierCost] = []
      for box_spec in self.obs_spec:
        self.obs_barrier_cost.append(
            BarrierCost(
                self.barrier_clip_min, self.barrier_clip_max, self.q1_obs, self.q2_obs,
                BoxObsBoxFootprintCost(
                    state_box_limit=self.state_box_limit, box_spec=box_spec,
                    precision=self.obs_precision, buffer=self.buffer
                )
            )
        )

    self.has_vel_constr: bool = cfg.has_vel_constr
    if self.has_vel_constr:
      self.q1_v: float = cfg.q1_v
      self.q2_v: float = cfg.q2_v
      self.v_min: float = cfg.v_min
      self.v_max: float = cfg.v_max
      self.vel_max_barrier_cost = BarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
          UpperHalfCost(value=self.v_max, dim=2)
      )
      self.vel_min_barrier_cost = BarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
          LowerHalfCost(value=self.v_min, dim=2)
      )

    self.has_delta_constr: bool = cfg.has_delta_constr
    if self.has_delta_constr:
      self.q1_delta: float = cfg.q1_delta
      self.q2_delta: float = cfg.q2_delta
      self.delta_min: float = cfg.delta_min
      self.delta_max: float = cfg.delta_max
      self.delta_max_barrier_cost = BarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_delta, self.q2_delta,
          UpperHalfCost(value=self.delta_max, dim=4)
      )
      self.delta_min_barrier_cost = BarrierCost(
          self.barrier_clip_min, self.barrier_clip_max, self.q1_delta, self.q2_delta,
          LowerHalfCost(value=self.delta_min, dim=4)
      )

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray, slope: DeviceArray,
      theta: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)
        closest_pt (DeviceArray, vector shape)
        slope (DeviceArray, vector shape)
        theta (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    # control cost
    cost = self.w_accel * ctrl[0]**2 + self.w_omega * ctrl[1]**2

    # state cost
    sr = jnp.sin(slope[0])
    cr = jnp.cos(slope[0])
    offset = (sr * (state[0] - closest_pt[0]) - cr * (state[1] - closest_pt[1]) - self.track_offset)
    cost += self.w_contour * offset**2
    cost += self.w_vel * (state[2] - self.v_ref)**2
    cost -= self.w_theta * theta[0]

    # soft constraint cost
    cost += self.yaw_barrier_cost.get_stage_cost(state, ctrl, closest_pt, slope, theta, time_idx)
    cost += self.road_barrier_cost.get_stage_cost(state, ctrl, closest_pt, slope, theta, time_idx)
    cost += self.collision_barrier_cost.get_stage_cost(state, ctrl, time_idx)

    if self.has_obs_constr:
      for _obs_barrier_cost in self.obs_barrier_cost:
        _obs_barrier_cost: BarrierCost
        cost += _obs_barrier_cost.get_stage_cost(state, ctrl, time_idx)
    if self.has_vel_constr:
      cost += self.vel_max_barrier_cost.get_stage_cost(state, ctrl, time_idx)
      cost += self.vel_min_barrier_cost.get_stage_cost(state, ctrl, time_idx)
    if self.has_delta_constr:
      cost += self.delta_max_barrier_cost.get_stage_cost(state, ctrl, time_idx)
      cost += self.delta_min_barrier_cost.get_stage_cost(state, ctrl, time_idx)
    return cost


class UberExampleConstraint(BaseSplineCost):

  def __init__(self, cfg):
    super().__init__(cfg)

    self.yaw_min = cfg.yaw_min
    self.yaw_max = cfg.yaw_max

    if hasattr(cfg, "buffer"):
      self.buffer: float = cfg.buffer
      print(f"Adds buffer ({self.buffer}) around obstacles and road boundaries.")
    else:
      self.buffer = 0.

    self.yaw_constraint = SplineYawCost(cfg)
    self.road_constraint = SplineWideningRoadBoundaryCost(cfg=cfg, buffer=self.buffer)
    self.collision_constraint = ProximityCost(cfg.collision_dist, cfg.position_indices)

    self.obs_spec = cfg.obs_spec
    self.has_obs_constr = cfg.has_obs_constr
    if self.has_obs_constr:
      self.obs_precision = getattr(cfg, "obs_precision", [31, 11])
      self.obs_constraint = []
      for box_spec in self.obs_spec:
        self.obs_constraint.append(
            BoxObsBoxFootprintCost(
                state_box_limit=self.state_box_limit, box_spec=box_spec,
                precision=self.obs_precision, buffer=self.buffer
            )
        )

    self.has_vel_constr = cfg.has_vel_constr
    if self.has_vel_constr:
      self.v_min = cfg.v_min
      self.v_max = cfg.v_max
      self.vel_max_constraint = UpperHalfCost(value=self.v_max, dim=2)
      self.vel_min_constraint = LowerHalfCost(value=self.v_min, dim=2)

    self.has_delta_constr = cfg.has_delta_constr
    if self.has_delta_constr:
      self.delta_min = cfg.delta_min
      self.delta_max = cfg.delta_max
      self.delta_max_constraint = UpperHalfCost(value=self.delta_max, dim=4)
      self.delta_min_constraint = LowerHalfCost(value=self.delta_min, dim=4)

  @partial(jax.jit, static_argnames='self')
  def get_stage_cost(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray, slope: DeviceArray,
      theta: DeviceArray, time_idx: DeviceArray
  ) -> DeviceArray:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    cost = self.yaw_constraint.get_stage_cost(state, ctrl, closest_pt, slope, theta, time_idx)

    cost = jnp.maximum(
        cost, self.road_constraint.get_stage_cost(state, ctrl, closest_pt, slope, theta, time_idx)
    )

    cost = jnp.maximum(cost, self.collision_constraint.get_stage_cost(state, ctrl, time_idx))

    if self.has_vel_constr:
      cost = jnp.maximum(cost, self.vel_max_constraint.get_stage_cost(state, ctrl, time_idx))
      cost = jnp.maximum(cost, self.vel_min_constraint.get_stage_cost(state, ctrl, time_idx))

    if self.has_delta_constr:
      cost = jnp.maximum(cost, self.delta_max_constraint.get_stage_cost(state, ctrl, time_idx))
      cost = jnp.maximum(cost, self.delta_min_constraint.get_stage_cost(state, ctrl, time_idx))

    if self.has_obs_constr:
      for _obs_constraint in self.obs_constraint:
        _obs_constraint: BaseCost
        cost = jnp.maximum(cost, _obs_constraint.get_stage_cost(state, ctrl, time_idx))
    return cost

  @partial(jax.jit, static_argnames='self')
  def get_cost_dict(
      self, state: DeviceArray, ctrl: DeviceArray, closest_pt: DeviceArray, slope: DeviceArray,
      theta: DeviceArray, time_indices: DeviceArray
  ) -> Dict:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """

    yaw_cons = self.yaw_constraint.get_cost(state, ctrl, closest_pt, slope, theta, time_indices)
    road_cons = self.road_constraint.get_cost(state, ctrl, closest_pt, slope, theta, time_indices)
    collision_cons = self.collision_constraint.get_cost(state, ctrl, time_indices)

    info = dict(yaw_cons=yaw_cons, road_cons=road_cons, collision_cons=collision_cons)
    # info = dict(yaw_cons=yaw_cons, road_cons=road_cons)

    if self.has_vel_constr:
      info['vel_max_cons'] = self.vel_max_constraint.get_cost(state, ctrl, time_indices)
      info['vel_min_cons'] = self.vel_min_constraint.get_cost(state, ctrl, time_indices)
    if self.has_delta_constr:
      info['delta_max_cons'] = self.delta_max_constraint.get_cost(state, ctrl, time_indices)
      info['delta_min_cons'] = self.delta_min_constraint.get_cost(state, ctrl, time_indices)
    if self.has_obs_constr:
      obs_cons = -jnp.inf
      for _obs_constraint in self.obs_constraint:
        _obs_constraint: BaseCost
        obs_cons = jnp.maximum(obs_cons, _obs_constraint.get_cost(state, ctrl, time_indices))
      info['obs_cons'] = obs_cons

    return info
