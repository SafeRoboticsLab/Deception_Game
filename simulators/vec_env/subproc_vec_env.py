# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for vectorized environments.

Modified from stable-baseline3 and https://github.com/allenzren/alano.
"""

from typing import Callable, Optional, Union, List, Dict, Any
import numpy as np
import multiprocessing as mp
import cloudpickle
import dill
import gym
from tqdm import tqdm


def _worker(remote, parent_remote, env_fn_wrapper, cpu_ind=-1):
  parent_remote.close()
  env = env_fn_wrapper.var  # was var.() when using gym
  while True:
    try:
      cmd, data = remote.recv()
      if cmd == "step":
        observation, reward, done, info = env.step(data)
        # if done:
        #   # save final observation where user can get it, then reset
        #   info["terminal_observation"] = observation
        #   observation = env.reset()
        remote.send((observation, reward, done, info))
      elif cmd == "seed":
        remote.send(env.seed(data))
      elif cmd == "reset":
        observation = env.reset(**data)
        remote.send(observation)
      elif cmd == "render":
        remote.send(env.render(data))
      elif cmd == "close":
        env.close()
        remote.close()
        break
      elif cmd == "get_spaces":
        remote.send((env.observation_space, env.action_space))
      elif cmd == "env_method":
        method = getattr(env, data[0])
        remote.send(method(*data[1], **data[2]))
      elif cmd == "get_attr":
        remote.send(getattr(env, data))
      elif cmd == "set_attr":
        remote.send(setattr(env, data[0], data[1]))
      else:
        raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
    except EOFError:
      break


class SubprocVecEnv():
  """
  Creates a multiprocess vectorized wrapper for multiple environments,
  distributing each environment to its own process, allowing significant speed
  up when the environment is computationally complex.

  For performance reasons, if your environment is not IO bound, the number of
  environments should not exceed the number of logical cores on your CPU.

  Warning:
      Only 'forkserver' and 'spawn' start methods are thread-safe,
      which is important when TensorFlow sessions or other non thread-safe
      libraries are used in the parent (see issue #217). However, compared to
      'fork' they incur a small start-up cost and have restrictions on
      global variables. With those methods, users must wrap the code in an
      ``if __name__ == "__main__":`` block.
      For more information, see the multiprocessing documentation.

  Attribute:
      env_fns: Environments to run in subprocesses
      start_method: method used to start the subprocesses. Must be one of the
          methods returned by multiprocessing.get_all_start_methods(). Defaults
          to 'forkserver' on available platforms, and 'spawn' otherwise.
  """

  def __init__(
      self, env_fns, cpu_offset=0, start_method=None,
      pickle_option='cloudpickle'
  ):
    self.waiting = False
    self.closed = False
    self.n_envs = len(env_fns)

    if start_method is None:
      # Fork is not a thread safe method (see issue #217)
      # but is more user friendly (does not require to wrap the code in
      # a `if __name__ == "__main__":`)
      forkserver_available = "forkserver" in mp.get_all_start_methods()
      start_method = "forkserver" if forkserver_available else "spawn"
    ctx = mp.get_context(start_method)

    self.remotes, self.work_remotes = zip(
        *[ctx.Pipe() for _ in range(self.n_envs)]
    )
    self.processes = []
    if pickle_option == 'cloudpickle':
      pickle_wrapper = CloudpickleWrapper
    elif pickle_option == 'dill':
      pickle_wrapper = DillWrapper
    else:
      raise 'Unknown pickle options!'
    for ind, (work_remote, remote, env_fn) in enumerate(
        zip(self.work_remotes, self.remotes, env_fns)
    ):
      args = (work_remote, remote, pickle_wrapper(env_fn), ind + cpu_offset)
      # daemon=True: if the main process crashes, we should not cause things to
      # hang. pytype:disable=attribute-error
      process = ctx.Process(target=_worker, args=args, daemon=True)
      process.start()
      self.processes.append(process)
      work_remote.close()

    self.remotes[0].send(("get_spaces", None))
    observation_space, action_space = self.remotes[0].recv()
    self.observation_space: gym.Space = observation_space
    self.action_space: gym.Space = action_space

  # region: step functions.

  def step(self, actions):
    """ Step the environments with the given action"""
    self.step_async(actions)
    return self.step_wait()

  def step_async(self, actions):
    for remote, action in zip(self.remotes, actions):
      remote.send(("step", action))
    self.waiting = True

  def step_wait(self):
    results = [remote.recv() for remote in self.remotes]
    self.waiting = False
    obs, rews, dones, infos = zip(*results)
    return np.stack(obs), np.stack(rews), np.stack(dones), infos

  # endregion

  # region: reset functions.

  def reset(self, **kwargs):
    for remote in self.remotes:
      remote.send(("reset", kwargs))
    obs = [remote.recv() for remote in self.remotes]
    return np.stack(obs)

  def reset_arg(self, args_list, **kwargs):
    obs = self.env_method_arg("reset", args_list, **kwargs)
    return np.stack(obs)

  # endregion

  # region: utils.

  def seed(self, seed):
    for idx, remote in enumerate(self.remotes):
      remote.send(("seed", seed + idx))
    return [remote.recv() for remote in self.remotes]

  def close(self):
    if self.closed:
      return
    if self.waiting:
      for remote in self.remotes:
        remote.recv()
    for remote in self.remotes:
      remote.send(("close", None))
    for process in self.processes:
      process.join()
    self.closed = True

  # def get_images(self):
  #     for pipe in self.remotes:
  #         # gather images from subprocesses
  #         # `mode` will be taken into account later
  #         pipe.send(("render", "rgb_array"))
  #     imgs = [pipe.recv() for pipe in self.remotes]
  #     return imgs

  def get_attr(self, attr_name, indices=None):
    """Returns attribute from vectorized environment (see base class)."""
    target_remotes = self._get_target_remotes(indices)
    for remote in target_remotes:
      remote.send(("get_attr", attr_name))
    return [remote.recv() for remote in target_remotes]

  def set_attr(
      self, attr_name, value, indices=None, value_batch: bool = False
  ):
    """Sets attribute inside vectorized environments (see base class)."""
    target_remotes = self._get_target_remotes(indices)
    if value_batch:
      for remote, value_item in zip(target_remotes, value):
        remote.send(("set_attr", (attr_name, value_item)))
    else:
      for remote in target_remotes:
        remote.send(("set_attr", (attr_name, value)))
    for remote in target_remotes:
      remote.recv()

  def env_method(
      self, method_name: str, *method_args, indices=None, **method_kwargs
  ):
    """Calls instance methods of vectorized environments."""
    target_remotes = self._get_target_remotes(indices)
    for remote in target_remotes:
      remote.send(("env_method", (method_name, method_args, method_kwargs)))
    return [remote.recv() for remote in target_remotes]

  def env_method_arg(
      self, method_name, method_args_list, indices=None, **method_kwargs
  ):
    """Calls instance methods of vectorized environments with args."""
    target_remotes = self._get_target_remotes(indices)
    for method_args, remote in zip(method_args_list, target_remotes):
      remote.send(("env_method", (method_name, method_args, method_kwargs)))
    return [remote.recv() for remote in target_remotes]

  def _get_target_remotes(self, indices):
    """
    Gets the connection object needed to communicate with the wanted envs that
    are in subprocesses.
    """
    if indices is None:
      indices = range(self.n_envs)
    return [self.remotes[i] for i in indices]

  def simulate_trajectories(
      self, num_trajectories: int, T_rollout: int, end_criterion: str,
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      return_info=False, use_tqdm=False, **kwargs
  ):
    if not isinstance(reset_kwargs_list, list):
      reset_kwargs_list = [reset_kwargs_list for _ in range(num_trajectories)]
    if not isinstance(action_kwargs_list, list):
      action_kwargs_list = [
          action_kwargs_list for _ in range(num_trajectories)
      ]

    results = np.empty(shape=(num_trajectories,), dtype=int)
    length = np.empty(shape=(num_trajectories,), dtype=int)
    trajectories = []
    info_list = []

    n_parallel_runs = max(int(np.ceil(num_trajectories / self.n_envs)), 1)

    if use_tqdm:
      iterable = tqdm(
          range(n_parallel_runs), desc=f'sim trajs with {self.n_envs} envs',
          leave=False
      )
    else:
      iterable = range(n_parallel_runs)
    for i in iterable:
      tmp = i * self.n_envs
      n_envs_need = min(num_trajectories - tmp, self.n_envs)
      method_args_list = [(
          T_rollout, end_criterion, reset_kwargs_list[tmp + j],
          action_kwargs_list[tmp + j]
      ) for j in range(n_envs_need)]
      res_all = self.env_method_arg(
          method_name="simulate_one_trajectory",
          method_args_list=method_args_list, indices=np.arange(n_envs_need),
          **kwargs
      )
      for j, res in enumerate(res_all):
        trajectories.append(res[0])
        results[tmp + j] = res[1]
        length[tmp + j] = len(res[0])
        info_list.append(res[2])

    if return_info:
      return trajectories, results, length, info_list
    else:
      return trajectories, results, length

  def simulate_trajectories_zs(
      self,
      num_trajectories: int,
      T_rollout: int,
      end_criterion: str,
      adversary: Union[Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
                       List],
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      return_info=False,
      use_tqdm=False,
      **kwargs,
  ):
    if not isinstance(reset_kwargs_list, list):
      reset_kwargs_list = [reset_kwargs_list for _ in range(num_trajectories)]
    if not isinstance(action_kwargs_list, list):
      action_kwargs_list = [
          action_kwargs_list for _ in range(num_trajectories)
      ]
    if not isinstance(adversary, list):
      adversary_list = [adversary for _ in range(self.n_envs)]
    else:
      assert len(adversary) == self.n_envs
      adversary_list = adversary

    results = np.empty(shape=(num_trajectories,), dtype=int)
    length = np.empty(shape=(num_trajectories,), dtype=int)
    trajectories = []
    info_list = []

    n_parallel_runs = max(int(np.ceil(num_trajectories / self.n_envs)), 1)

    if use_tqdm:
      iterable = tqdm(
          range(n_parallel_runs), desc=f'sim trajs with {self.n_envs} envs',
          leave=False
      )
    else:
      iterable = range(n_parallel_runs)
    for i in iterable:
      tmp = i * self.n_envs
      n_envs_need = min(num_trajectories - tmp, self.n_envs)
      method_args_list = [(
          T_rollout, end_criterion, adversary_list[j],
          reset_kwargs_list[tmp + j], action_kwargs_list[tmp + j]
      ) for j in range(n_envs_need)]
      res_all = self.env_method_arg(
          method_name="simulate_one_trajectory",
          method_args_list=method_args_list, indices=np.arange(n_envs_need),
          **kwargs
      )
      for j, res in enumerate(res_all):
        trajectories.append(res[0])
        results[tmp + j] = res[1]
        length[tmp + j] = len(res[0])
        info_list.append(res[2])

    if return_info:
      return trajectories, results, length, info_list
    else:
      return trajectories, results, length

  # endregion


class CloudpickleWrapper:
  """
  Uses cloudpickle to serialize contents (otherwise multiprocessing tries to
  use pickle).
  """

  def __init__(self, var):
    self.var = var

  def __getstate__(self):
    return cloudpickle.dumps(self.var)

  def __setstate__(self, var) -> None:
    self.var = cloudpickle.loads(var)


class DillWrapper:
  """
  Uses dill to serialize contents (otherwise multiprocessing tries to use
  pickle).
  """

  def __init__(self, var):
    self.var = var

  def __getstate__(self):
    return dill.dumps(self.var)

  def __setstate__(self, var) -> None:
    self.var = dill.loads(var)
