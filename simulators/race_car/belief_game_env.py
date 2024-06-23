# ------------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: haiminh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# Modified from: https://github.com/SafeRoboticsLab/simulators
# ------------------------------------------------------------

from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib
from gym import spaces
from jax import numpy as jnp
from scipy.special import softmax

from .track import Track
from .utils import get_centerline_from_traj
from .cost_uber import UberExampleCost, UberExampleConstraint
from ..base_zs_env import BaseZeroSumEnv
from ..utils import ActionZS


class BeliefGameEnv(BaseZeroSumEnv):
  """
  This simulator is basically the same as the single-agent one, with the
  difference of agent's dynamics, action space, and step function.
  """

  def __init__(self, cfg_env, cfg_agent, cfg_cost) -> None:
    super().__init__(cfg_env, cfg_agent)

    # Track width.
    self.track_width_right = cfg_env.track_width_right
    self.track_width_left = cfg_env.track_width_left

    # Track length and centerline.
    if hasattr(cfg_env, "track_file"):
      self.track_len = None
      center_line = get_centerline_from_traj(cfg_env.track_file)
    else:
      self.track_len = cfg_env.track_len
      _center_line_x = (
          np.linspace(start=0., stop=self.track_len, num=1000, endpoint=True).reshape(1, -1)
      )
      center_line = np.concatenate(
          (_center_line_x, cfg_env.track_center_py * np.ones_like(_center_line_x)), axis=0
      )

    # Constructs the track.
    self.track = Track(
        center_line=center_line, width_left=self.track_width_left,
        width_right=self.track_width_right, loop=getattr(cfg_env, 'loop', True)
    )

    # Observations.
    self.failure_thr = getattr(cfg_env, "failure_thr", 0.)
    self.reset_thr = getattr(cfg_env, "reset_thr", 0.)
    self.obs_type = getattr(cfg_env, "obs_type", "perfect")
    self.step_keep_constraints = True
    self.step_keep_targets = False

    # Constructs the cost and constraint. Assume the same constraint.
    cfg_cost.state_box_limit = cfg_agent.state_box_limit
    cfg_cost.wheelbase = cfg_agent.wheelbase
    cfg_cost.track_width_left = cfg_env.track_width_left
    cfg_cost.track_width_right = cfg_env.track_width_right
    cfg_cost.track_width_left_narrow = cfg_env.track_width_left_narrow
    cfg_cost.track_two_lane_end = cfg_env.track_two_lane_end
    cfg_cost.track_four_lane_start = cfg_env.track_four_lane_start
    cfg_cost.obs_spec = cfg_env.obs_spec
    self.cost_type = getattr(cfg_cost, "cost_type", "Lagrange")
    if self.cost_type == "Lagrange":
      self.cost = UberExampleCost(cfg_cost)
    else:
      raise ValueError("Cost type not supported!")
    self.constraint = UberExampleConstraint(cfg_cost)
    self.g_x_fail = float(cfg_env.g_x_fail)

    # region: Visualization
    track_cat = (
        np.concatenate((self.track.track_bound[:2, :], self.track.track_bound[2:, :]), axis=-1)
    )
    x_min, y_min = np.min(track_cat, axis=1)
    x_max, y_max = np.max(track_cat, axis=1)
    self.visual_bounds = np.array([[x_min, x_max], [y_min, y_max]])
    x_eps = (x_max-x_min) * 0.005
    y_eps = (y_max-y_min) * 0.005
    self.visual_extent = np.array([
        self.visual_bounds[0, 0] - x_eps, self.visual_bounds[0, 1] + x_eps,
        self.visual_bounds[1, 0] - y_eps, self.visual_bounds[1, 1] + y_eps
    ])
    self.obs_vertices_list = []
    for box_spec in self.constraint.obs_spec:
      box_center = np.array([box_spec[0], box_spec[1]])
      box_yaw = box_spec[2]
      # rotate clockwise (to move the world frame to obstacle frame)
      rot_mat = np.array([
          [np.cos(box_yaw), -np.sin(box_yaw)],
          [np.sin(box_yaw), np.cos(box_yaw)],
      ])
      box_halflength = box_spec[3]
      box_halfwidth = box_spec[4]
      offset = np.array([
          [-box_halflength, -box_halfwidth],
          [-box_halflength, box_halfwidth],
          [box_halflength, box_halfwidth],
          [box_halflength, -box_halfwidth],
      ])
      rot_offset = np.einsum("ik,jk->ji", rot_mat, offset)
      self.obs_vertices_list.append(rot_offset + box_center)
    # endregion

    # Initializes.
    self.reset_rej_sampling = getattr(cfg_env, "reset_rej_sampling", True)
    self.build_obs_rst_space(cfg_env, cfg_agent, cfg_cost)
    self.seed(cfg_env.seed)
    self.reset()

  # region: abstract functions in BaseZeroSumEnv
  def get_cost(
      self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray,
      constraints: Optional[dict] = None
  ) -> float:
    states_all, controls_all = self._reshape(state, action, state_nxt)
    closest_pt, slope, theta = self.track.get_closest_pts(states_all[:2, :])
    states_all = jnp.array(states_all)
    controls_all = jnp.array(controls_all)
    closest_pt = jnp.array(closest_pt)
    slope = jnp.array(slope)
    theta = jnp.array(theta)
    dummy_time_indices = jnp.zeros((1, states_all.shape[1]), dtype=int)
    cost = float(
        jnp.sum(
            self.cost.get_cost(
                states_all, controls_all, closest_pt, slope, theta, time_indices=dummy_time_indices
            )
        )
    )
    return cost

  def get_constraints(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray) -> Dict:
    states_all, controls_all = self._reshape(state, action, state_nxt)
    closest_pt, slope, theta = self.track.get_closest_pts(states_all[:2, :])
    states_all = jnp.array(states_all)
    controls_all = jnp.array(controls_all)
    closest_pt = jnp.array(closest_pt)
    slope = jnp.array(slope)
    theta = jnp.array(theta)
    dummy_time_indices = jnp.zeros((1, states_all.shape[1]), dtype=int)
    cons_dict: Dict = self.constraint.get_cost_dict(
        states_all, controls_all, closest_pt, slope, theta, time_indices=dummy_time_indices
    )
    for k, v in cons_dict.items():
      cons_dict[k] = np.asarray(v).reshape(-1, 2)
    return cons_dict

  def get_target_margin(self, state: np.ndarray, action: ActionZS,
                        state_nxt: np.ndarray) -> Optional[Dict]:
    return None

  def get_done_and_info(
      self, state: np.ndarray, constraints: Dict, targets: Optional[Dict] = None,
      final_only: bool = True, end_criterion: Optional[str] = None
  ) -> Tuple[bool, Dict]:
    """
    Gets the done flag and a dictionary to provide additional information of
    the step function given current state, current action, next state,
    constraints, and targets.

    Args:
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.
        targets (Dict): each (key, value) pair is the name and value of a
            target margin function.

    Returns:
        bool: True if the episode ends.
        Dict: additional information of the step, such as target margin and
            safety margin used in reachability analysis.
    """
    if end_criterion is None:
      end_criterion = self.end_criterion

    done = False
    done_type = "not_raised"
    if self.cnt >= self.timeout:
      done = True
      done_type = "timeout"
    if self.track_len is not None:
      if state[0] > self.track_len:
        done = True
        done_type = "leave_track"

    # Retrieves constraints / traget values.
    constraint_values = None
    for key, value in constraints.items():
      if constraint_values is None:
        num_pts = value.shape[1]
        constraint_values = value
      else:
        assert num_pts == value.shape[1], ("The length of constraint ({}) do not match".format(key))
        constraint_values = np.concatenate((constraint_values, value), axis=0)
    g_x_list = np.max(constraint_values, axis=0)

    if targets is not None:
      target_values = np.empty((0, constraint_values.shape[1]))
      for key, value in targets.items():
        assert num_pts == value.shape[1], ("The length of target ({}) do not match".format(key))
        target_values = np.concatenate((target_values, value), axis=0)
      l_x_list = np.max(target_values, axis=0)
    else:
      l_x_list = np.full((num_pts,), fill_value=np.inf)

    # Gets info.
    if final_only:
      g_x = float(g_x_list[-1])
      l_x = float(l_x_list[-1])
      binary_cost = 1. if g_x > self.failure_thr else 0.
    else:
      g_x = g_x_list
      l_x = l_x_list
      binary_cost = 1. if np.any(g_x > self.failure_thr) else 0.

    # Gets done flag
    if end_criterion == 'failure':
      if final_only:
        failure = np.any(constraint_values[:, -1] > self.failure_thr)
      else:
        failure = np.any(constraint_values > self.failure_thr)
      if failure:
        done = True
        done_type = "failure"
        g_x = self.g_x_fail
    elif end_criterion == 'reach-avoid':
      if final_only:
        failure = g_x > self.failure_thr
        success = not failure and l_x <= 0.
      else:
        v_x_list = np.empty(shape=(num_pts,))
        v_x_list[num_pts - 1] = max(l_x_list[num_pts - 1], g_x_list[num_pts - 1])
        for i in range(num_pts - 2, -1, -1):
          v_x_list[i] = max(g_x_list[i], min(l_x_list[i], v_x_list[i + 1]))
        inst = np.argmin(v_x_list)
        failure = np.any(constraint_values[:, :inst + 1] > self.failure_thr)
        success = not failure and (v_x_list[inst] <= 0)
      if success:
        done = True
        done_type = "success"
      elif failure:
        done = True
        done_type = "failure"
        g_x = self.g_x_fail
    elif end_criterion == 'timeout':
      pass
    else:
      raise ValueError("End criterion not supported!")

    # Gets info
    info = {"done_type": done_type, "g_x": g_x, "l_x": l_x, "binary_cost": binary_cost}
    if self.step_keep_constraints:
      info['constraints'] = constraints
    if self.step_keep_targets:
      info['targets'] = targets
    return done, info

  # endregion

  # region: gym
  def build_obs_rst_space(self, cfg_env, cfg_agent, cfg_cost):
    # Reset Sample Space.
    # NOTE:
    # - The first two dimension is in the local frame and it needs to call track.local2global() to
    # get the (x, y) position in the global frame.
    # - Dimension 7 and onwards are pre-softmax'd probability values
    reset_space = np.array(cfg_env.reset_space, dtype=np.float32)
    self.reset_sample_sapce = spaces.Box(low=reset_space[:, 0], high=reset_space[:, 1])

    # Observation space.
    track_cat = (
        np.concatenate((self.track.track_bound[:2, :], self.track.track_bound[2:, :]), axis=-1)
    )
    x_min, y_min = np.min(track_cat, axis=1)
    x_max, y_max = np.max(track_cat, axis=1)
    if self.obs_type == "perfect":
      low = np.zeros((self.state_dim,))
      low[0] = x_min
      low[1] = y_min
      low[2] = cfg_agent.v_min
      low[3] = -np.pi
      low[4] = cfg_agent.delta_min
      low[5] = x_min
      low[6] = y_min
      low[7:] = 0.
      high = np.zeros((self.state_dim,))
      high[0] = x_max
      high[1] = y_max
      high[2] = cfg_agent.v_max
      high[3] = np.pi
      high[4] = cfg_agent.delta_max
      high[5] = x_max
      high[6] = y_max
      high[7:] = 1.
    else:
      raise ValueError("Observation type {} is not supported!")
    self.observation_space = spaces.Box(low=np.float32(low), high=np.float32(high))
    self.obs_dim = self.observation_space.low.shape[0]

  def seed(self, seed: int = 0):
    super().seed(seed)
    self.reset_sample_sapce.seed(seed)

  def reset(self, state: Optional[np.ndarray] = None, cast_torch: bool = False,
            **kwargs) -> Union[np.ndarray, torch.FloatTensor]:
    """
    Resets the environment and returns the new state.

    Args:
        state (np.ndarray, optional): reset to this state if provided. Defaults
            to None.
        cast_torch (bool, optional): cast state to torch if True. Defaults to
            False.

    Returns:
        np.ndarray: the new state of the shape (dim_x, ).
    """
    super().reset()
    if state is None:
      reset_flag = True
      while reset_flag:
        # Samples a state.
        state = self.reset_sample_sapce.sample()

        # Converts local positions (the first two states) to global positions.
        state[:2], slope = self.track.local2global(state[:2], return_slope=True)

        # Obtains a probability vector for belief states using softmax.
        state[7:9] = softmax(state[7:9])

        if self.reset_rej_sampling:
          ctrl = jnp.zeros((2, 1))
          closest_pt, slope, theta = self.track.get_closest_pts(state[:2, np.newaxis])
          state_jnp = jnp.array(state[:, np.newaxis])
          closest_pt = jnp.array(closest_pt)
          slope = jnp.array(slope)
          theta = jnp.array(theta)
          dummy_time_indices = jnp.zeros((1, state_jnp.shape[1]), dtype=int)
          cons = self.constraint.get_cost(
              state_jnp, ctrl, closest_pt, slope, theta, time_indices=dummy_time_indices
          )[0]
          reset_flag = cons > self.reset_thr
        else:
          reset_flag = False
      state[3] = np.mod(slope + state[3] + np.pi, 2 * np.pi) - np.pi
    self.state = state.copy()

    obs = self.get_obs(state)
    if cast_torch:
      obs = torch.FloatTensor(obs)
    return obs

  def get_obs(self, state: np.ndarray) -> np.ndarray:
    """Gets the observation given the state.

    Args:
        state (np.ndarray): state of the shape (dim_x, ) or  (dim_x, N).

    Returns:
        np.ndarray: observation. It can be the state or uses cos theta and
            sin theta to represent yaw.
    """
    assert state.shape[0] == self.state_dim, ("State shape is incorrect!")
    if self.obs_type == 'perfect':
      obs = state.copy()
      # obs = ((obs - self.observation_space.low) /
      #        (self.observation_space.high - self.observation_space.low) * 2 - 1)
    else:
      raise NotImplementedError
    return obs

  def get_samples(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gets state samples for value function plotting.

    Args:
        nx (int): the number of points along x-axis.
        ny (int): the number of points along y-axis.

    Returns:
        np.ndarray: a list of x-position.
        np.ndarray: a list of y-position.
    """
    xs = np.linspace(self.visual_bounds[0, 0], self.visual_bounds[0, 1], nx)
    ys = np.linspace(self.visual_bounds[1, 0], self.visual_bounds[1, 1], ny)
    return xs, ys

  # endregion

  # region: utils
  def get_agent_action(self, obs, init_control, state, **kwargs):
    return self.agent.get_action(obs=obs, controls=init_control, state=state, **kwargs)

  def update_obstacle(self, obs_spec: np.ndarray):  # TODO
    pass

  def check_on_track(self, states: np.ndarray, thr: float = 0.) -> np.ndarray:
    """Checks if the state is on the track (considering footprint).

    Args:
        states (np.ndarray).
        thr(float, optional): threshold to boundary. Defaults to 0.

    Returns:
        np.ndarray: a bool array of shape (N, ). True if the agent is on the
            track.
    """
    if states.ndim == 1:
      states = states[:, np.newaxis]
    ctrls = np.zeros((self.agent.dyn.dim_u, states.shape[1]))
    closest_pt, slope, theta = self.track.get_closest_pts(states[:2, :])
    states = jnp.array(states)
    ctrls = jnp.array(ctrls)
    closest_pt = jnp.array(closest_pt)
    slope = jnp.array(slope)
    theta = jnp.array(theta)
    dummy_time_indices = jnp.zeros((1, states.shape[1]), dtype=int)
    road_cons = self.constraint.road_constraint.get_cost(
        states, ctrls, closest_pt, slope, theta, time_indices=dummy_time_indices
    )
    flags = road_cons <= thr
    return flags.reshape(-1)

  def render(
      self, ax: Optional[matplotlib.axes.Axes] = None, c_track: str = 'k', c_obs: str = 'r',
      c_ego: str = 'b', s: float = 12
  ):
    """Visualizes the current environment.

    Args:
        ax (Optional[matplotlib.axes.Axes], optional): the axes of matplotlib
            to plot if provided. Otherwise, we use the current axes. Defaults
            to None.
        c_track (str, optional): the color of the track. Defaults to 'k'.
        c_obs (str, optional): the color of the obstacles. Defaults to 'r'.
        c_ego (str, optional): the color of the ego agent. Defaults to 'b'.
        s (float, optional): the size of the ego agent point. Defaults to 12.
    """
    if ax is None:
      ax = plt.gca()
    self.track.plot_track(ax, c=c_track)
    self.render_footprint(ax=ax, state=self.state, c=c_ego)
    self.render_obs(ax=ax, c=c_obs)
    ax.axis(self.visual_extent)
    ax.set_aspect('equal')

  def render_footprint(
      self, ax, state: np.ndarray, c: str = 'b', s: float = 12, lw: float = 1.5, alpha: float = 1.,
      plot_center: bool = True
  ):
    if plot_center:
      ax.scatter(state[0], state[1], c=c, s=s)
    ego = self.agent.footprint.move2state(state[[0, 1, 3]])
    ego.plot(ax, color=c, lw=lw, alpha=alpha)

  def render_obs(self, ax, c: str = 'r'):
    for vertices in self.obs_vertices_list:
      for i in range(4):
        if i == 3:
          ax.plot(vertices[[3, 0], 0], vertices[[3, 0], 1], c=c, zorder=0)
        else:
          ax.plot(vertices[i:i + 2, 0], vertices[i:i + 2, 1], c=c, zorder=0)

  def _reshape(self, state: np.ndarray, action: ActionZS,
               state_nxt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenates states with state_nxt and appends dummy control after action.

    Args:
        state (np.ndarray): current states of the shape (dim_x, N-1).
        action (np.ndarray): current actions of the shape (dim_u, N-1).
        state_nxt (np.ndarray): next state or final state of the shape
            (dim_x, ).

    Returns:
        np.ndarray: states with the final state.
        np.ndarray: action with the dummy control.
    """
    if state.ndim == 1:
      state = state[:, np.newaxis]
    if state_nxt.ndim == 1:
      state_nxt = state_nxt[:, np.newaxis]
    ctrl = action['ctrl']
    if ctrl.ndim == 1:
      ctrl = ctrl[:, np.newaxis]
    assert state.shape[1] == ctrl.shape[1], (
        "The length of states ({}) and ".format(state.shape[1]),
        "the length of controls ({}) don't match!".format(ctrl.shape[1])
    )
    assert state_nxt.shape[1] == 1, "state_nxt should consist only 1 state!"

    states_with_final = np.concatenate((state, state_nxt), axis=1)
    controls_with_final = np.concatenate((ctrl, np.zeros((ctrl.shape[0], 1))), axis=1)
    return states_with_final, controls_with_final

  def report(self):
    print("This is a zero-sum environment.")
    if self.track_len is not None:
      print("Straight road, box footprint, box obstacles!")
    else:
      print("road from file, box footprint, box obstacles!")

  def get_constraints_all(
      self, states: np.ndarray, controls: np.ndarray, time_indices: np.ndarray
  ) -> Dict:
    closest_pt, slope, theta = self.track.get_closest_pts(states[:2, :])
    states = jnp.array(states)
    controls = jnp.array(controls)
    closest_pt = jnp.array(closest_pt)
    slope = jnp.array(slope)
    theta = jnp.array(theta)
    time_indices = jnp.array(time_indices)
    cons_dict: Dict = self.constraint.get_cost_dict(
        states, controls, closest_pt, slope, theta, time_indices=time_indices
    )
    for k, v in cons_dict.items():
      cons_dict[k] = np.asarray(v).reshape(-1, states.shape[1])
    return cons_dict

  # endregion


class BeliefGameWithIntentsEnv(BeliefGameEnv):
  """
  Belief Game environment that additionally considers intent hypotheses.
  """

  def __init__(self, cfg_env, cfg_agent, cfg_cost) -> None:
    super().__init__(cfg_env, cfg_agent, cfg_cost)

  # region: gym
  def reset(self, state: Optional[np.ndarray] = None, cast_torch: bool = False,
            **kwargs) -> Union[np.ndarray, torch.FloatTensor]:
    """
    Resets the environment and returns the new state.

    Args:
        state (np.ndarray, optional): reset to this state if provided. Defaults
            to None.
        cast_torch (bool, optional): cast state to torch if True. Defaults to
            False.

    Returns:
        np.ndarray: the new state of the shape (dim_x, ).
    """
    super().reset()
    if state is None:
      reset_flag = True
      while reset_flag:
        # Samples a state.
        state = self.reset_sample_sapce.sample()

        # Converts local positions (the first two states) to global positions.
        state[:2], slope = self.track.local2global(state[:2], return_slope=True)

        # Obtains a probability vector for belief states using softmax.
        state[7:9] = softmax(state[7:9])
        state[9:] = softmax(state[9:])

        if self.reset_rej_sampling:
          ctrl = jnp.zeros((2, 1))
          closest_pt, slope, theta = self.track.get_closest_pts(state[:2, np.newaxis])
          state_jnp = jnp.array(state[:, np.newaxis])
          closest_pt = jnp.array(closest_pt)
          slope = jnp.array(slope)
          theta = jnp.array(theta)
          dummy_time_indices = jnp.zeros((1, state_jnp.shape[1]), dtype=int)
          cons = self.constraint.get_cost(
              state_jnp, ctrl, closest_pt, slope, theta, time_indices=dummy_time_indices
          )[0]
          reset_flag = cons > self.reset_thr
        else:
          reset_flag = False
      state[3] = np.mod(slope + state[3] + np.pi, 2 * np.pi) - np.pi
    self.state = state.copy()

    obs = self.get_obs(state)
    if cast_torch:
      obs = torch.FloatTensor(obs)
    return obs

  # endregion
