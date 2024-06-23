# ------------------------------------------------------------
# Belief Game with Intent Hypotheses | ISAACS
# Copyright (c) 2023 Princeton University
# Email: haiminh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# Modified from: https://github.com/SafeRoboticsLab/simulators
# ------------------------------------------------------------

import os
import sys
from types import SimpleNamespace
import numpy as np
import wandb
import argparse
from functools import partial
from shutil import copyfile
import jax
from omegaconf import OmegaConf

from agent.isaacs import ISAACS
from simulators import BeliefGameWithIntentsEnv, save_obj, PrintLogger

from script.bgame_intent_pretrain_dstb import visualize

jax.config.update('jax_platform_name', 'cpu')


def main(config_file):
  # Loads config.
  cfg = OmegaConf.load(config_file)
  cfg.train.device = cfg.solver.device
  os.makedirs(cfg.solver.out_folder, exist_ok=True)
  copyfile(config_file, os.path.join(cfg.solver.out_folder, 'config.yaml'))
  log_path = os.path.join(cfg.solver.out_folder, 'log.txt')
  if os.path.exists(log_path):
    os.remove(log_path)
  sys.stdout = PrintLogger(log_path)
  sys.stderr = PrintLogger(log_path)

  # Sets up Weights & Biases.
  if cfg.solver.use_wandb:
    wandb.init(entity='haiminh', project=cfg.solver.project_name, name=cfg.solver.name)
    tmp_cfg = {
        'environment': OmegaConf.to_container(cfg.environment),
        'solver': OmegaConf.to_container(cfg.solver),
        'arch': OmegaConf.to_container(cfg.arch),
        'train': OmegaConf.to_container(cfg.train)
    }
    wandb.config.update(tmp_cfg)

  # Sets up the environment.
  if cfg.agent.dyn == "BeliefGameDynamicsWithIntents":
    env_class = BeliefGameWithIntentsEnv
  else:
    raise ValueError("Dynamics type not supported!")

  # Constructs the environment.
  print("\n== Environment information ==")
  env = env_class(cfg.environment, cfg.agent, cfg.cost)
  env.step_keep_constraints = False
  env.report()

  # Constructs the solver.
  print("\n== Solver information ==")
  solver = ISAACS(cfg.solver, cfg.train, cfg.arch, cfg.environment)
  policy = solver.policy
  env.agent.init_policy(
      policy_type="NNCS", cfg=SimpleNamespace(device=policy.device), actor=policy.ctrl.net
  )
  n_params_ctrl = sum(p.numel() for p in policy.ctrl.net.parameters() if p.requires_grad)
  print(f'\nTotal parameters in ctrl: {n_params_ctrl}')
  print(f"We want to use: {cfg.train.device}, and Agent uses: {policy.device}")
  print("Critic is using cuda: ", next(policy.critic.net.parameters()).is_cuda)

  # Training starts.
  print("\n== Learning starts ==")
  if cfg.solver.vis:
    vel_list = [13., 15., 17.]
    yaw_list = [-np.pi / 4, -np.pi / 6, -np.pi / 8, 0., np.pi / 8, np.pi / 6, np.pi / 4]
    visualize_callback = partial(
        visualize, vel_list=vel_list, yaw_list=yaw_list,
        end_criterion=cfg.solver.rollout_end_criterion, T_rollout=cfg.solver.eval_timeout,
        nx=cfg.solver.cmap_res_x, ny=cfg.solver.cmap_res_y, subfigsz_x=cfg.solver.fig_size_x,
        subfigsz_y=cfg.solver.fig_size_y, vmin=-cfg.environment.g_x_fail,
        vmax=cfg.environment.g_x_fail, markersz=40
    )
  else:
    visualize_callback = None
  train_record, train_progress, violation_record, episode_record, pq_top_k = (
      solver.learn(env, visualize_callback=visualize_callback)
  )
  train_dict = {}
  train_dict['train_record'] = train_record
  train_dict['train_progress'] = train_progress
  train_dict['violation_record'] = violation_record
  train_dict['episode_record'] = episode_record
  train_dict['pq_top_k'] = list(pq_top_k.queue)
  save_obj(train_dict, os.path.join(cfg.solver.out_folder, 'train'))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("config", "intent_isaacs.yaml")
  )
  args = parser.parse_args()
  main(args.config_file)
