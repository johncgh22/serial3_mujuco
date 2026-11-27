"""Base interfaces and reference containers for 3-DOF serial robot controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class TrajectoryReference:
    """Simultaneous joint-space and Cartesian (MuJoCo world frame) reference state."""
    t: float
    q_d: np.ndarray       # (3,) desired joint angles [rad]
    qd_d: np.ndarray      # (3,) desired joint velocities [rad/s]
    qdd_d: np.ndarray     # (3,) desired joint accelerations [rad/s^2]
    x_d: np.ndarray       # (3,) desired end-effector position [m]
    xd_d: np.ndarray      # (3,) desired end-effector velocity [m/s]
    xdd_d: np.ndarray     # (3,) desired end-effector acceleration [m/s^2]


class BaseController(ABC):
    """Common interface for all joint-space and task-space torque controllers."""

    name: str = "base"
    short_name: str = "base"
    control_space: str = "joint"  # "joint" or "cartesian"

    def __init__(self, torque_limit: float | np.ndarray = 1.5):
        if np.ndim(torque_limit) == 0:
            self.torque_limit = np.full(3, float(torque_limit), dtype=np.float64)
        else:
            self.torque_limit = np.asarray(torque_limit, dtype=np.float64)

    def reset(self) -> None:
        """Reset any internal controller state (integrators, parameter estimates)."""

    def clip_torque(self, tau: np.ndarray) -> np.ndarray:
        """Enforce actuator torque saturation limits [N*m]."""
        return np.clip(tau, -self.torque_limit, self.torque_limit)

    @abstractmethod
    def compute(
        self,
        t: float,
        q: np.ndarray,
        qd: np.ndarray,
        ref: TrajectoryReference,
        dt: float,
    ) -> np.ndarray:
        """Compute joint torque command tau [N*m] of shape (3,)."""
