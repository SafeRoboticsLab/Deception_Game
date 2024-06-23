# ------------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: haiminh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# Modified from: https://github.com/SafeRoboticsLab/simulators
# ------------------------------------------------------------

from typing import Optional, Tuple, Union, Dict, List
import copy
import numpy as np
import torch

# Dynamics.
from .dynamics.bicycle5D import Bicycle5D
from .dynamics.bicycle5D_dstb import BicycleDstb5D
from .dynamics.uber_no_belief import UberNoBelief
from .dynamics.belief_game_dynamics import BeliefGameDynamics, BeliefGameDynamicsWithIntents

from .cost.base_cost import BaseCost

# Footprint.
from .footprint.box import BoxFootprint

# Policy.
from .policy.base_policy import BasePolicy
from .policy.ilqr_policy import ILQR
from .policy.ilqr_spline_policy import ILQRSpline
from .policy.ilqr_reachability_spline_policy import ILQRReachabilitySpline
from .policy.nn_policy import NeuralNetworkControlSystem


class Agent:
  """A basic unit in our environments.

  Attributes:
      dyn (object): agent's dynamics.
      footprint (object): agent's shape.
      policy (object): agent's policy.
  """
  policy: Optional[BasePolicy]
  ego_observable: Optional[List]
  agents_policy: Dict[str, BasePolicy]
  agents_order: Optional[List]

  def __init__(self, cfg, action_space: np.ndarray, env=None) -> None:
    if cfg.dyn == "Bicycle5D":
      self.dyn = Bicycle5D(cfg, action_space)
    elif cfg.dyn == "BicycleDstb5D":
      self.dyn = BicycleDstb5D(cfg, action_space)
    elif cfg.dyn == "UberNoBelief":
      self.dyn = UberNoBelief(cfg, action_space)
    elif cfg.dyn == "BeliefGameDynamics":
      self.dyn = BeliefGameDynamics(cfg, action_space)
    elif cfg.dyn == "BeliefGameDynamicsWithIntents":
      self.dyn = BeliefGameDynamicsWithIntents(cfg, action_space)
    else:
      raise ValueError("Dynamics type not supported!")

    try:
      self.env = copy.deepcopy(env)  # imaginary environment
    except Exception as e:
      print("WARNING: Cannot copy env - {}".format(e))

    if cfg.footprint == "Box":
      self.footprint = BoxFootprint(box_limit=cfg.state_box_limit)

    # Policy should be initialized by `init_policy()`.
    self.policy = None
    self.id: str = cfg.agent_id
    self.ego_observable = None

  def integrate_forward(
      self,
      state: np.ndarray,
      control: Optional[Union[np.ndarray, torch.Tensor]] = None,
      num_segment: Optional[int] = 1,
      noise: Optional[np.ndarray] = None,
      noise_type: Optional[str] = 'unif',
      adversary: Optional[Union[np.ndarray, torch.Tensor]] = None,
      **kwargs,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the next state of the vehicle given the current state and
    control input.

    Args:
        state (np.ndarray): (dyn.dim_x, ) array.
        control (np.ndarray): (dyn.dim_u, ) array.
        num_segment (int, optional): The number of segements to forward the
            dynamics. Defaults to 1.
        noise (np.ndarray, optional): the ball radius or standard
            deviation of the Gaussian noise. The magnitude should be in the
            sense of self.dt. Defaults to None.
        noise_type(str, optional): Uniform or Gaussian. Defaults to 'unif'.
        adversary (np.ndarray, optional): adversarial control (disturbance).
            Defaults to None.

    Returns:
        np.ndarray: next state.
        np.ndarray: clipped control.
    """
    assert control is not None or self.policy is not None, (
        "You need to either pass in a control or construct a policy!"
    )
    if control is None:
      obs: np.ndarray = kwargs.get('obs')
      kwargs['state'] = state.copy()
      control = self.get_action(obs=obs.copy(), **kwargs)[0]
    elif not isinstance(control, np.ndarray):
      control = control.cpu().numpy()
    if noise is not None:
      assert isinstance(noise, np.ndarray)
    if adversary is not None and not isinstance(adversary, np.ndarray):
      adversary = adversary.cpu().numpy()

    return self.dyn.integrate_forward(
        state=state, control=control, num_segment=num_segment, noise=noise, noise_type=noise_type,
        adversary=adversary, **kwargs
    )

  def get_dyn_jacobian(
      self,
      nominal_states: np.ndarray,
      nominal_controls: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the linearized 'A' and 'B' matrix of the ego vehicle around
    nominal states and controls.

    Args:
        nominal_states (np.ndarray): states along the nominal trajectory.
        nominal_controls (np.ndarray): controls along the trajectory.

    Returns:
        np.ndarray: the Jacobian of next state w.r.t. the current state.
        np.ndarray: the Jacobian of next state w.r.t. the current control.
    """
    A, B = self.dyn.get_jacobian(nominal_states, nominal_controls)
    return np.asarray(A), np.asarray(B)

  def get_action(
      self, obs: np.ndarray, agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        obs (np.ndarray): current observation.
        agents_action (Optional[Dict]): other agents' actions that are
            observable to the ego agent.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    if self.ego_observable is not None:
      for agent_id in self.ego_observable:
        assert agent_id in agents_action

    action, solver_info = self.policy.get_action(  # Proposed action.
        obs=obs, agents_action=agents_action, **kwargs
    )

    return action, solver_info

  def init_policy(self, policy_type: str, cfg, cost: Optional[BaseCost] = None, **kwargs):
    if policy_type == "ILQR":
      self.policy = ILQR(self.id, cfg, self.dyn, cost, **kwargs)
    elif policy_type == "ILQRSpline":
      self.policy = ILQRSpline(self.id, cfg, self.dyn, cost, **kwargs)
    elif policy_type == "ILQRReachabilitySpline":
      self.policy = ILQRReachabilitySpline(self.id, cfg, self.dyn, cost, **kwargs)
    # elif policy_type == "MPC":
    elif policy_type == "NNCS":
      self.policy = NeuralNetworkControlSystem(id=self.id, cfg=cfg, **kwargs)
    else:
      raise ValueError("The policy type ({}) is not supported!".format(policy_type))

  def report(self):
    print(self.id)
    if self.ego_observable is not None:
      print("  - The agent can observe:", end=' ')
      for i, k in enumerate(self.ego_observable):
        print(k, end='')
        if i == len(self.ego_observable) - 1:
          print('.')
        else:
          print(', ', end='')
    else:
      print("  - The agent can only access observation.")

    if self.agents_order is not None:
      print("  - The agent keeps agents order:", end=' ')
      for i, k in enumerate(self.agents_order):
        print(k, end='')
        if i == len(self.agents_order) - 1:
          print('.')
        else:
          print(' -> ', end='')
