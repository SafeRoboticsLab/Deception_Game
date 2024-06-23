from typing import Optional
import numpy as np


class BoxFootprint:

  def __init__(
      self, center: Optional[np.ndarray] = None,
      box_limit: Optional[np.ndarray] = None,
      offset: Optional[np.ndarray] = None
  ) -> None:
    if center is None:
      self.center = np.zeros(3)
    else:
      self.center = center.copy()

    if offset is None:
      assert box_limit is not None
      self.offset = np.array([[box_limit[0], box_limit[2]],
                              [box_limit[0], box_limit[3]],
                              [box_limit[1], box_limit[3]],
                              [box_limit[1], box_limit[2]]])
    else:
      self.offset = offset.copy()

  def move2state(self, state: np.ndarray):
    """
    Args:
        state (np.ndarray): x, y, yaw

    Returns:
        _type_: _description_
    """
    center = state.copy()
    yaw = state[2]
    rot_mat = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
    rot_offset = np.einsum("ik,jk->ji", rot_mat, self.offset)
    return BoxFootprint(center=center, offset=rot_offset)

  def plot(self, ax, color='r', lw=1.5, alpha=1.):
    vertices = self.center[[0, 1]] + self.offset
    for i in range(-1, 4):
      if i == 3:
        ax.plot(
            vertices[[3, 0], 0], vertices[[3, 0], 1], c=color, lw=lw,
            alpha=alpha, zorder=0
        )
      else:
        ax.plot(
            vertices[i:i + 2, 0], vertices[i:i + 2, 1], c=color, lw=lw,
            alpha=alpha, zorder=0
        )
