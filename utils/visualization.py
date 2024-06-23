# ------------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: haiminh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# Modified from: https://github.com/SafeRoboticsLab/simulators
# ------------------------------------------------------------

from typing import Callable, Optional, Union, List, Any
from matplotlib import cm
import numpy as np
import torch
import jax.numpy as jnp

from matplotlib import pyplot as plt


def get_values(
    env,
    critic: Callable[[np.ndarray, Optional[Union[np.ndarray, torch.Tensor]]], np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    v: float,
    yaw: float,
    batch_size: int,
    delta: float = 0.,
    fail_value: float = 1.,
    xH=0.,
    yH=0.,
    bP=None,
    bC=None,
):
  values = np.full((xs.shape[0], ys.shape[0]), fill_value=fail_value)
  it = np.nditer(values, flags=['multi_index'])
  while not it.finished:
    idx_all = []
    obs_all = np.empty((0, env.obs_dim), dtype=float)
    while len(idx_all) < batch_size:
      idx = it.multi_index
      x = xs[idx[0]]
      y = ys[idx[1]]
      if env.state_dim == 4:
        state = np.array([x, y, v, yaw]).reshape(-1, 1)
      elif env.state_dim == 5:
        state = np.array([x, y, v, yaw, delta]).reshape(-1, 1)
      elif env.state_dim == 7:
        state = np.array([x, y, v, yaw, delta, xH, yH]).reshape(-1, 1)
      elif env.state_dim == 9:
        if bP and bC:
          state = np.array([x, y, v, yaw, delta, xH, yH, bP, bC]).reshape(-1, 1)
        else:
          uni_belief = .5 * np.ones(2)
          physical_state = np.array([x, y, v, yaw, delta, xH, yH])
          state = np.concatenate((physical_state, uni_belief)).reshape(-1, 1)
      elif env.state_dim > 9:
        intent_dim = env.state_dim - 9
        uni_belief = (1./intent_dim) * np.ones(intent_dim)
        physical_state = np.array([x, y, v, yaw, delta, xH, yH])
        state = np.concatenate((physical_state, .5 * np.ones(2)))
        state = np.concatenate((state, uni_belief)).reshape(-1, 1)

      closest_pt, slope, theta = env.track.get_closest_pts(state[:2, :])
      state_jnp = jnp.array(state)
      control_jnp = jnp.zeros((2, 1))
      closest_pt = jnp.array(closest_pt)
      slope = jnp.array(slope)
      theta = jnp.array(theta)
      dummy_time_indices = jnp.zeros((1, state_jnp.shape[1]), dtype=int)
      g_x = env.constraint.get_cost(
          state_jnp, control_jnp, closest_pt, slope, theta, time_indices=dummy_time_indices
      )
      g_x = np.asarray(g_x)[0]
      if g_x < 0.:
        obs = env.get_obs(state.reshape(-1))
        obs_all = np.concatenate((obs_all, obs.reshape(1, -1)))
        idx_all += [idx]
      it.iternext()
      if it.finished:
        break
    v_all = critic(obs_all, append=None)
    for v_s, idx in zip(v_all, idx_all):
      values[idx] = v_s
  return values


def plot_traj(
    ax, trajectory: np.ndarray, result: int, c: str = 'b', lw: float = 2., zorder: int = 1,
    vel_scatter: bool = False, s: int = 40, plot_human=False
):
  traj_x = trajectory[:, 0]
  traj_y = trajectory[:, 1]
  if plot_human:
    traj_xH = trajectory[:, 5]
    traj_yH = trajectory[:, 6]

  if vel_scatter:
    if plot_human:
      raise not NotImplementedError

    vel = trajectory[:, 2]
    ax.scatter(
        traj_x[0], traj_y[0], s=s, c=vel[0], cmap=cm.copper, vmin=0, vmax=2., edgecolor='none',
        marker='s', zorder=zorder
    )
    ax.scatter(
        traj_x[1:-1], traj_y[1:-1], s=s - 12, c=vel[1:-1], cmap=cm.copper, vmin=0, vmax=2.,
        edgecolor='none', marker='o', zorder=zorder
    )
    if result == -1:
      marker_final = 'X'
      edgecolor_final = 'r'
    elif result == 1:
      marker_final = '*'
      edgecolor_final = 'g'
    else:
      marker_final = '^'
      edgecolor_final = 'y'
    ax.scatter(
        traj_x[-1], traj_y[-1], s=s, c=vel[-1], cmap=cm.copper, vmin=0, vmax=2.,
        edgecolor=edgecolor_final, marker=marker_final, zorder=zorder
    )
  else:
    ax.scatter(traj_x[0], traj_y[0], s=s, c=c, zorder=zorder)
    ax.plot(traj_x, traj_y, c=c, ls='-', lw=lw, zorder=zorder)
    if plot_human:
      ax.scatter(traj_xH[0], traj_yH[0], s=s, c='m', zorder=zorder)
      ax.plot(traj_xH, traj_yH, c='m', ls='--', lw=lw, zorder=zorder)

    if result == -1:
      ax.scatter(traj_x[-1], traj_y[-1], s=s, marker='x', c='r', zorder=zorder)
    elif result == 1:
      ax.scatter(traj_x[-1], traj_y[-1], s=s, marker='*', c='g', zorder=zorder)
    else:
      ax.scatter(traj_x[-1], traj_y[-1], s=s, marker='^', c='y', zorder=zorder)


def get_trajectories_zs(
    env, adversary: Callable[[np.ndarray, np.ndarray, Any],
                             np.ndarray], vel_list: List[float], yaw_list: List[float],
    num_pts: int = 5, T_rollout: int = 150, end_criterion: str = "failure", dim_ctrl=2, dim_dstb=5
):
  num_traj = len(vel_list) * len(yaw_list) * num_pts
  reset_kwargs_list = []
  for _ in range(num_pts):
    far_failure = False
    cnt = 0
    a_dummy = {'ctrl': np.zeros((dim_ctrl, 1)), 'dstb': np.zeros((dim_dstb, 1))}
    while (not far_failure) and (cnt <= 10):
      env.reset()
      state = env.state.copy()
      cons_dict = env.get_constraints(state, a_dummy, state)
      constraint_values = None
      for key, value in cons_dict.items():
        if constraint_values is None:
          num_pts = value.shape[1]
          constraint_values = value
        else:
          assert num_pts == value.shape[1], (
              "The length of constraint ({}) do not match".format(key)
          )
          constraint_values = np.concatenate((constraint_values, value), axis=0)
      g_x = np.max(constraint_values[:, -1], axis=0)
      far_failure = g_x <= -0.1
      cnt += 1
    for vel in vel_list:
      for yaw in yaw_list:
        state[2] = vel
        state[3] = yaw
        reset_kwargs_list.append(dict(state=state.copy()))

  trajectories, results, _ = env.simulate_trajectories(
      num_traj, T_rollout=T_rollout, end_criterion=end_criterion,
      reset_kwargs_list=reset_kwargs_list, adversary=adversary
  )
  return trajectories, results
