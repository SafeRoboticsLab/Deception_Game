# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import csv
import numpy as np


def trans2cossin(state: np.ndarray) -> np.ndarray:
  if state.ndim == 1:
    _state = np.zeros(6)
  else:
    _state = np.zeros((6, state.shape[1]))
  _state[:3, ...] = state[:3, ...].copy()  # x, y, v
  _state[3, ...] = np.cos(state[3, ...].copy())
  _state[4, ...] = np.sin(state[3, ...].copy())
  _state[5:, ...] = state[4:, ...].copy()
  return _state


def get_centerline_from_traj(filepath: str) -> np.ndarray:
  """
  Gets the centerline of the track from the trajectory data. We currently only
  support 2D track.

  Args:
      filepath (str): the path to file consisting of the centerline position.

  Returns:
      np.ndarray: centerline, of the shape (2, N).
  """
  x = []
  y = []
  with open(filepath) as f:
    spamreader = csv.reader(f, delimiter=',')
    for i, row in enumerate(spamreader):
      if i > 0:
        x.append(float(row[0]))
        y.append(float(row[1]))

  return np.array([x, y])
