# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import time
import random
from abc import abstractmethod
from typing import Any, Tuple, Optional, Callable, List, Dict, Union
import numpy as np
import torch
from gym import spaces
from tqdm import tqdm
from scipy.special import softmax

from .agent import Agent
from .base_env import BaseEnv
from .utils import ActionZS

import copy


class BaseZeroSumEnv(BaseEnv):
  """
  Implements an environment of a two-player discrete-time zero-sum dynamic
  game. The agent minimizing the cost is called 'ctrl', while the agent
  maximizing the cost is called 'dstb'.
  """

  def __init__(self, cfg_env: Any, cfg_agent: Any) -> None:
    # assert cfg_env.num_agents == 2, (
    #     "Zero-Sum Game currently only supports two agents!"
    # )
    super().__init__(cfg_env)
    self.env_type = "zero-sum"

    # Action Space.
    ctrl_space = np.array(cfg_agent.action_range.ctrl, dtype=np.float32)
    self.action_space_ctrl = spaces.Box(low=ctrl_space[:, 0], high=ctrl_space[:, 1])
    dstb_space = np.array(cfg_agent.action_range.dstb, dtype=np.float32)
    self.action_space_dstb = spaces.Box(low=dstb_space[:, 0], high=dstb_space[:, 1])
    self.action_space = spaces.Dict(dict(ctrl=self.action_space_ctrl, dstb=self.action_space_dstb))
    self.action_dim_ctrl = ctrl_space.shape[0]
    self.action_dim_dstb = dstb_space.shape[0]
    tmp_action_space = {'ctrl': ctrl_space, 'dstb': dstb_space}
    self.agent = Agent(cfg_agent, tmp_action_space)
    self.state_dim = self.agent.dyn.dim_x

    self.integrate_kwargs = getattr(cfg_env, "integrate_kwargs", {})
    if "noise" in self.integrate_kwargs:
      if self.integrate_kwargs['noise'] is not None:
        self.integrate_kwargs['noise'] = np.array(self.integrate_kwargs['noise'])

  def step(
      self,
      action: ActionZS,
      cast_torch: bool = False,
  ) -> Tuple[Union[np.ndarray, torch.Tensor], float, bool, Dict]:
    """Implements the step function in the environment.

    Args:
        action (ActionZS): a dictionary consists of ctrl and dstb, which are
            accessed by action['ctrl'] and action['dstb'].
        cast_torch (bool): cast state to torch if True.

    Returns:
        np.ndarray: next state.
        float: the reward that ctrl wants to maximize and dstb wants to
            minimize.
        bool: True if the episode ends.
        Dict[str, Any]]: additional information of the step, such as target
            margin and safety margin used in reachability analysis.
    """

    self.cnt += 1
    state_nxt, ctrl_clip, dstb_clip = self.agent.integrate_forward(
        state=self.state, control=action['ctrl'], adversary=action['dstb'], **self.integrate_kwargs
    )
    state_cur = self.state.copy()
    self.state = state_nxt.copy()

    constraints = self.get_constraints(state_cur, action, state_nxt)
    cost = self.get_cost(state_cur, action, state_nxt, constraints)
    targets = self.get_target_margin(state_cur, action, state_nxt)
    done, info = self.get_done_and_info(state_nxt, constraints, targets)

    obs = self.get_obs(state_nxt)
    if cast_torch:
      obs = torch.FloatTensor(obs)

    info['ctrl_clip'] = ctrl_clip
    info['dstb_clip'] = dstb_clip

    return obs, -cost, done, info

  @abstractmethod
  def get_cost(
      self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray,
      constraints: Optional[Dict] = None
  ) -> float:
    """
    Gets the cost given current state, current action, and next state.

    Args:
        state (np.ndarray): current state.
        action (ActionZS): a dictionary consists of ctrl and dstb, which are
            accessed by action['ctrl'] and action['dstb'].
        state_nxt (np.ndarray): next state.
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.

    Returns:
        float: the cost that ctrl wants to minimize and dstb wants to maximize.
    """
    raise NotImplementedError

  @abstractmethod
  def get_constraints(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray) -> Dict:
    """
    Gets the values of all constaint functions given current state, current
    action, and next state.

    Args:
        state (np.ndarray): current state.
        action (ActionZS): a dictionary consists of ctrl and 'dstb', which are
            accessed by action['ctrl'] and action['dstb'].
        state_nxt (np.ndarray): next state.

    Returns:
        Dict: each (key, value) pair is the name and value of a constraint
            function.
    """
    raise NotImplementedError

  @abstractmethod
  def get_target_margin(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray) -> Dict:
    """
    Gets the values of all target margin functions given current state, current
    action, and next state.

    Args:
        state (np.ndarray): current states.
        action (ActionZS): a dictionary consists of ctrl and dstb, which are
            accessed by action['ctrl'] and action['dstb'].
        state_nxt (np.ndarray): next state.

    Returns:
        Dict: each (key, value) pair is the name and value of a target margin
            function.
    """
    raise NotImplementedError

  @abstractmethod
  def get_done_and_info(
      self, state: np.ndarray, constraints: Dict, targets: Dict, final_only: bool = True,
      end_criterion: Optional[str] = None
  ) -> Tuple[bool, Dict]:
    """
    Gets the done flag and a dictionary to provide additional information of
    the step function given current state, current action, next state,
    constraints, and targets.

    Args:
        state (np.ndarray): current state.
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.
        targets (Dict): each (key, value) pair is the name and value of a
            target margin function.

    Returns:
        bool: True if the episode ends.
        Dict: additional information of the step, such as target margin and
            safety margin used in reachability analysis.
    """
    raise NotImplementedError

  def get_shielded_trajectory_and_cap_zone(
      self,
      state_init: np.ndarray,
      py_track: float,
      v_track: float,
      T_rollout: int,
      adversary: Callable[[np.ndarray, np.ndarray], np.ndarray],
      map_plan=False,
      rbs_plan=False,
      con_plan=False,
  ) -> Tuple[np.ndarray, int, Dict]:
    """
    Modified from simulate_one_trajectory
    TODO:
      - ego's nominal control
      - cap zones
      - add a baseline strategy

    Args:
        state_init (np.ndarray): initial state
        py_track (float): py position to track
        v_track (float): velocity to track
        T_rollout (int): rollout horizon
        adversary (callable): a mapping from current state and ctrl to adversarial ctrl (dstb).

    Returns:
        np.ndarray: state trajectory.
        int: result (0: unfinished, 1: success, -1: failure).
        Dict: auxiliarry information -
            "action_hist": action sequence.
            "plan_hist": planning info for every step.
            "reward_hist": rewards for every step.
            "step_hist": information for every step.
    """
    # Stores the environment attributes and sets to rollout settings.
    state_hist = []
    obs_hist = []
    action_hist = {'ctrl': [], 'dstb': []}
    reward_hist = []
    plan_hist = []
    step_hist = []

    # Initializes robot.
    init_control = None
    result = 0
    obs = state_init
    self.state = state_init
    state_hist.append(self.state)

    # tt = time.time()
    for _ in range(T_rollout):
      obs_ego = copy.deepcopy(obs)

      # Gets the ego action.
      if map_plan:
        # region: computes ego MAP policy.
        # type uncertainty
        obs_ego[7:9] = 0.
        b_type_map = np.zeros(2,)
        b_type_map[np.argmax(obs[7:9])] = 1.
        obs_ego[7:9] = b_type_map
        # intent uncertainty
        _N_intent_map = 1
        b_intent_map = np.zeros(self.agent.dyn.num_intent,)
        _b_intent = obs[9:]
        _idx_tmp = _b_intent.argsort()[-_N_intent_map:]
        b_intent_map[_idx_tmp] = _b_intent[_idx_tmp] / sum(_b_intent[_idx_tmp])
        # b_intent_map[_idx_tmp] = 1. / _N_intent_map
        obs_ego[9:] = b_intent_map
        # print(obs)
        # print(obs_ego)
        with torch.no_grad():
          ctrl, solver_info = self.agent.get_action(obs=obs_ego, controls=init_control)
        # endregion
      elif rbs_plan:
        with torch.no_grad():
          ctrl, solver_info = self.agent_robust.get_action(obs=obs_ego[:7], controls=init_control)
      elif con_plan:
        # region: computes ego Contingency policy.
        obs_con = copy.deepcopy(obs_ego)
        # shrinks type uncetainty
        _flag = False
        if obs_ego[7] <= self.agent.dyn.bel_thres:
          _flag = True
          obs_con[7] = 0.
          obs_con[8] = 1.
        elif obs_ego[8] <= self.agent.dyn.bel_thres:
          _flag = True
          obs_con[7] = 1.
          obs_con[8] = 0.
        # shrinks the intent uncertainty
        _b_intent_tmp = np.asarray(obs_con[9:])
        _idx_tmp = np.where(_b_intent_tmp <= self.agent.dyn.bel_thres_intent)
        if len(_idx_tmp[0]) > 4:
          _flag = True
          _b_intent_tmp[_idx_tmp] = 0.
          _b_intent_tmp = _b_intent_tmp / sum(_b_intent_tmp)
          obs_con[9:] = _b_intent_tmp.tolist()
        if _flag:  # Branching out: Use Belief policy
          with torch.no_grad():
            ctrl, solver_info = self.agent.get_action(obs=obs_con, controls=init_control)
        else:  # Safeguard against all
          with torch.no_grad():
            ctrl, solver_info = self.agent_robust.get_action(obs=obs_ego[:7], controls=init_control)
        # endregion
      else:
        # Belief Game.
        with torch.no_grad():
          ctrl, solver_info = self.agent.get_action(obs=obs, controls=init_control)

      # Applies action: `done` and `info` are evaluated at the next state.
      dstb = adversary(obs, ctrl)
      action = {'ctrl': ctrl, 'dstb': dstb}
      obs, reward, done, step_info = self.step(action)

      # Executes step callback and stores history.
      state_hist.append(self.state)
      obs_hist.append(obs)
      action_hist['ctrl'].append(step_info['ctrl_clip'])
      action_hist['dstb'].append(step_info['dstb_clip'])
      plan_hist.append(solver_info)
      reward_hist.append(reward)
      step_hist.append(step_info)

      # print(dstb, step_info['dstb_clip'])

      # Checks termination criterion.
      if done:
        if step_info["done_type"] == "success":
          result = 1
        elif step_info["done_type"] == "failure":
          result = -1
        self.reset()
        break

    # print(time.time()-tt)

    for k, v in action_hist.items():
      action_hist[k] = np.array(v)
    info = dict(
        obs_hist=np.array(obs_hist), action_hist=action_hist, plan_hist=plan_hist,
        reward_hist=np.array(reward_hist), step_hist=step_hist
    )
    self.reset()
    return np.array(state_hist), result, info

  def get_trajectory_eval(
      self, state_init: np.ndarray, py_track: float, T_rollout: int, adversary, human_mode,
      map_plan=False, rbs_plan=False, con_plan=False
  ) -> Tuple[np.ndarray, int, Dict]:

    # Initializes the system.
    init_control = None
    obs = state_init
    self.state = state_init

    # Initializes metrics.
    result = 0
    py_err = 0.
    acc_old = 0.
    jerk = []
    steer = []

    # Use noisily-rational oppo policy if not adversary
    if human_mode == 'rat':
      type_idx = random.randrange(2)
      oppo_range = self.agent.dyn.dstb_scenario[type_idx, :, :]
      goal_idx = random.randrange(self.agent.dyn.num_intent)
      oppo_goal = self.agent.dyn.intent_set[goal_idx, :]
      K_vx = self.agent.dyn.K_vx
      K_vy = self.agent.dyn.K_vy

    # Main simulation loop.
    for sim_step in range(T_rollout):
      obs_ego = copy.deepcopy(obs)

      # Gets the ego action.
      if map_plan:
        # region: computes ego MAP policy.
        # type uncertainty
        obs_ego[7:9] = 0.
        b_type_map = np.zeros(2,)
        b_type_map[np.argmax(obs[7:9])] = 1.
        obs_ego[7:9] = b_type_map
        # intent uncertainty
        _N_intent_map = 1
        b_intent_map = np.zeros(self.agent.dyn.num_intent,)
        _b_intent = obs[9:]
        _idx_tmp = _b_intent.argsort()[-_N_intent_map:]
        b_intent_map[_idx_tmp] = _b_intent[_idx_tmp] / sum(_b_intent[_idx_tmp])
        # b_intent_map[_idx_tmp] = 1. / _N_intent_map
        obs_ego[9:] = b_intent_map
        # print(obs)
        # print(obs_ego)
        with torch.no_grad():
          ctrl, _ = self.agent.get_action(obs=obs_ego, controls=init_control)
        # endregion
      elif rbs_plan:
        with torch.no_grad():
          ctrl, _ = self.agent_robust.get_action(obs=obs_ego[:7], controls=init_control)
      elif con_plan:
        # region: computes ego Contingency policy.
        obs_con = copy.deepcopy(obs_ego)
        # shrinks type uncetainty
        _flag = False
        if obs_ego[7] <= self.agent.dyn.bel_thres:
          _flag = True
          obs_con[7] = 0.
          obs_con[8] = 1.
        elif obs_ego[8] <= self.agent.dyn.bel_thres:
          _flag = True
          obs_con[7] = 1.
          obs_con[8] = 0.
        # shrinks the intent uncertainty
        _b_intent_tmp = np.asarray(obs_con[9:])
        _idx_tmp = np.where(_b_intent_tmp <= self.agent.dyn.bel_thres_intent)
        if len(_idx_tmp[0]) > 4:
          _flag = True
          _b_intent_tmp[_idx_tmp] = 0.
          _b_intent_tmp = _b_intent_tmp / sum(_b_intent_tmp)
          obs_con[9:] = _b_intent_tmp.tolist()
        if _flag:  # Branching out: Use Belief policy
          with torch.no_grad():
            ctrl, _ = self.agent.get_action(obs=obs_con, controls=init_control)
        else:  # Safeguard against all
          with torch.no_grad():
            ctrl, _ = self.agent_robust.get_action(obs=obs_ego[:7], controls=init_control)
        # endregion
      else:
        with torch.no_grad():
          ctrl, _ = self.agent.get_action(obs=obs_ego, controls=init_control)

      # Applies action: `done` and `info` are evaluated at the next state.
      # Obtains human policy according to the specified mode.
      if human_mode == 'rat':
        # region: noisily-rational human
        # Randomize b/w rat and adv
        mode_idx = random.randrange(100)
        if mode_idx < 50:
          dstb_vx = K_vx * (obs[5] - oppo_goal[0]) + random.uniform(-0.1, 0.1)
          dstb_vy = K_vy * (obs[6] - oppo_goal[1]) + random.uniform(-0.5, 0.5)
          dstb = np.array([dstb_vx, dstb_vy])
          dstb = np.clip(dstb, oppo_range[:, 0], oppo_range[:, 1])
        else:
          dstb = adversary(obs, ctrl)
        # endregion
      # Human policy in robust mode does not take in beliefs.
      elif human_mode == 'adv_rbs' or human_mode == 'umd_rbs':
        dstb = adversary(obs[:7], ctrl)
      else:
        dstb = adversary(obs, ctrl)

      action = {'ctrl': ctrl, 'dstb': dstb}
      obs, reward, done, step_info = self.step(action)

      # Updates metrics
      py_err += np.abs(self.state[1] - py_track)
      ctrl_clip = step_info['ctrl_clip']
      jerk.append(np.abs(ctrl_clip[0] - acc_old))
      acc_old = ctrl_clip[0]
      steer.append(np.abs(ctrl_clip[1]))

      # Checks termination criterion.
      if done:
        if step_info["done_type"] == "success":
          result = 1
        elif step_info["done_type"] == "failure":
          result = -1
        self.reset()
        break

      if self.state[0] >= self.track_len:
        result = 1
        self.reset()
        break

    avg_py_err = py_err / sim_step
    avg_norm_jerk = sum(jerk) / sim_step
    avg_norm_steer = sum(steer) / sim_step

    self.reset()
    return result, sim_step, avg_py_err, avg_norm_jerk + avg_norm_steer

  def simulate_one_trajectory(
      self,
      T_rollout: int,
      end_criterion: str,
      adversary: Callable[[np.ndarray, np.ndarray], np.ndarray],
      reset_kwargs: Optional[Dict] = None,
      action_kwargs: Optional[Dict] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None,
      **kwargs,
  ) -> Tuple[np.ndarray, int, Dict]:
    """
    Rolls out the trajectory given the horizon, termination criterion, reset
    keyword arguments, callback afeter every step, and callout after the
    rollout.

    Args:
        T_rollout (int): rollout horizon.
        end_criterion (str): termination criterion.
        adversary (callable): a mapping from current state and ctrl to
            adversarial ctrl (dstb).
        reset_kwargs (Dict): keyword argument dictionary for reset function.
        action_kwargs (Dict): keyword argument dictionary for get_action
            function.
        rollout_step_callback (Callable): function to call after every step.
        rollout_episode_callback (Callable): function to call after rollout.

    Returns:
        np.ndarray: state trajectory.
        int: result (0: unfinished, 1: success, -1: failure).
        Dict: auxiliarry information -
            "action_hist": action sequence.
            "plan_hist": planning info for every step.
            "reward_hist": rewards for every step.
            "step_hist": information for every step.
    """
    # Stores the environment attributes and sets to rollout settings.
    timeout_backup = self.timeout
    end_criterion_backup = self.end_criterion
    self.timeout = T_rollout
    self.end_criterion = end_criterion
    if reset_kwargs is None:
      reset_kwargs = {}
    if action_kwargs is None:
      action_kwargs = {}
    controller = None
    if "controller" in kwargs.keys():
      controller = copy.deepcopy(kwargs["controller"])

    state_hist = []
    obs_hist = []
    action_hist = {'ctrl': [], 'dstb': []}
    reward_hist = []
    plan_hist = []
    step_hist = []
    shield_ind = []

    # Initializes robot.
    init_control = None
    result = 0
    obs = self.reset(**reset_kwargs)
    state_hist.append(self.state)

    for t in range(T_rollout):
      if controller is None:
        # Gets action.
        action_kwargs['state'] = self.state.copy()
        action_kwargs['time_idx'] = t
        with torch.no_grad():
          ctrl, solver_info = self.agent.get_action(obs=obs, controls=init_control, **action_kwargs)
      else:
        new_joint_pos = controller.get_action()
        ctrl = new_joint_pos - np.array(self.agent.dyn.robot.get_joint_position())
        solver_info = None

      # Applies action: `done` and `info` are evaluated at the next state.
      dstb = adversary(obs, ctrl, **action_kwargs)
      action = {'ctrl': ctrl, 'dstb': dstb}
      obs, reward, done, step_info = self.step(action)

      # Executes step callback and stores history.
      state_hist.append(self.state)
      obs_hist.append(obs)
      action_hist['ctrl'].append(step_info['ctrl_clip'])
      action_hist['dstb'].append(step_info['dstb_clip'])
      plan_hist.append(solver_info)
      reward_hist.append(reward)
      step_hist.append(step_info)
      if rollout_step_callback is not None:
        rollout_step_callback(self, state_hist, action_hist, plan_hist, step_hist, time_idx=t)
      if solver_info is not None:
        if 'shield' in solver_info:
          shield_ind.append(solver_info['shield'])

      # Checks termination criterion.
      if done:
        if step_info["done_type"] == "success":
          result = 1
        elif step_info["done_type"] == "failure":
          result = -1
        break

      # Warms up initial controls with the computation from the previous cycle.
      if 'controls' in solver_info:
        init_control = np.zeros_like(solver_info['controls'], dtype=float)
        init_control[:, :-1] = solver_info['controls'][:, 1:]

    if rollout_episode_callback is not None:
      rollout_episode_callback(self, state_hist, action_hist, plan_hist, step_hist)
    # Reverts to training setting.
    self.timeout = timeout_backup
    self.end_criterion = end_criterion_backup
    for k, v in action_hist.items():
      action_hist[k] = np.array(v)
    info = dict(
        obs_hist=np.array(obs_hist), action_hist=action_hist, plan_hist=plan_hist,
        reward_hist=np.array(reward_hist), step_hist=step_hist, shield_ind=shield_ind
    )
    return np.array(state_hist), result, info

  def simulate_trajectories(
      self, num_trajectories: int, T_rollout: int, end_criterion: str,
      adversary: Callable[[np.ndarray, np.ndarray, Any],
                          np.ndarray], reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None, return_info=False, **kwargs
  ):
    """
    Rolls out multiple trajectories given the horizon, termination criterion,
    reset keyword arguments, callback afeter every step, and callback after the
    rollout. Need to call env.reset() after this function to revert back to the
    training mode.
    """

    if isinstance(reset_kwargs_list, list):
      assert num_trajectories == len(reset_kwargs_list), (
          "The length of reset_kwargs_list does not match with",
          "the number of rollout trajectories"
      )
    if isinstance(action_kwargs_list, list):
      assert num_trajectories == len(action_kwargs_list), (
          "The length of action_kwargs_list does not match with",
          "the number of rollout trajectories"
      )

    results = np.empty(shape=(num_trajectories,), dtype=int)
    length = np.empty(shape=(num_trajectories,), dtype=int)
    trajectories = []
    info_list = []
    use_tqdm = kwargs.get('use_tqdm', False)
    leave = kwargs.get('leave', False)
    if use_tqdm:
      iterable = tqdm(range(num_trajectories), desc='sim trajs', leave=leave)
    else:
      iterable = range(num_trajectories)

    for trial in iterable:
      if isinstance(reset_kwargs_list, list):
        reset_kwargs = reset_kwargs_list[trial]
      else:
        reset_kwargs = reset_kwargs_list
      if isinstance(action_kwargs_list, list):
        action_kwargs = action_kwargs_list[trial]
      else:
        action_kwargs = action_kwargs_list

      state_hist, result, info = self.simulate_one_trajectory(
          T_rollout=T_rollout, end_criterion=end_criterion, adversary=adversary,
          reset_kwargs=reset_kwargs, action_kwargs=action_kwargs,
          rollout_step_callback=rollout_step_callback,
          rollout_episode_callback=rollout_episode_callback, **kwargs
      )
      trajectories.append(state_hist)
      results[trial] = result
      length[trial] = len(state_hist)
      info_list.append(info)
    if return_info:
      return trajectories, results, length, info_list
    else:
      return trajectories, results, length
