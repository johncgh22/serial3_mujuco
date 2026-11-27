"""Reference trajectory generators for joint-space and Cartesian-space benchmarks."""

from __future__ import annotations

import numpy as np

from .base import TrajectoryReference
from .dynamics import AnalyticalDynamics


class SmoothJointTrajectory:
    """Smooth 3-DOF sinusoidal joint trajectory kept away from the upright singularity.

    Centered at q_center = [0.0, 0.55, 0.70] rad so the arm stays well-conditioned
    (det(J_v) != 0) throughout the motion, enabling direct head-to-head comparison
    between joint-space controllers and Cartesian Operational Space Control (OSC).
    """

    def __init__(
        self,
        dynamics: AnalyticalDynamics,
        q_center: np.ndarray | None = None,
        amp: np.ndarray | None = None,
        omega: np.ndarray | None = None,
        phase: np.ndarray | None = None,
    ):
        self.dyn = dynamics
        self.q_center = np.array([0.0, 0.55, 0.70], dtype=np.float64) if q_center is None else np.asarray(q_center, dtype=np.float64)
        self.amp = np.array([0.75, 0.30, 0.45], dtype=np.float64) if amp is None else np.asarray(amp, dtype=np.float64)
        self.omega = np.array([1.2, 1.5, 1.8], dtype=np.float64) if omega is None else np.asarray(omega, dtype=np.float64)
        self.phase = np.array([0.0, 0.0, 0.5], dtype=np.float64) if phase is None else np.asarray(phase, dtype=np.float64)

    def initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        ref0 = self(0.0)
        return ref0.q_d.copy(), ref0.qd_d.copy()

    def __call__(self, t: float) -> TrajectoryReference:
        s = np.sin(self.omega * t + self.phase)
        c = np.cos(self.omega * t + self.phase)
        q_d = self.q_center + self.amp * s
        qd_d = self.amp * self.omega * c
        qdd_d = -self.amp * (self.omega ** 2) * s

        x_d = self.dyn.fk(q_d)
        J = self.dyn.jacobian_v(q_d)
        Jd = self.dyn.jacobian_v_dot(q_d, qd_d)
        xd_d = J @ qd_d
        xdd_d = J @ qdd_d + Jd @ qd_d
        return TrajectoryReference(t=t, q_d=q_d, qd_d=qd_d, qdd_d=qdd_d, x_d=x_d, xd_d=xd_d, xdd_d=xdd_d)


class CartesianCircleTrajectory:
    """3D circular end-effector trajectory in front of the robot with analytical IK.

    Circle lies in a tilted vertical plane centered at [0.24, 0.0, 0.25] m with
    radius 0.07 m — strictly inside the reachable workspace and well away from
    singularities and joint limits.
    """

    def __init__(
        self,
        dynamics: AnalyticalDynamics,
        center: np.ndarray | None = None,
        radius: float = 0.07,
        omega: float = 1.5,
    ):
        self.dyn = dynamics
        self.center = np.array([0.24, 0.0, 0.25], dtype=np.float64) if center is None else np.asarray(center, dtype=np.float64)
        self.radius = float(radius)
        self.omega = float(omega)

    def solve_ik_exact(self, x: np.ndarray) -> np.ndarray:
        """Closed-form analytical IK for the 3-DOF RRR arm (elbow-up branch, q3 > 0)."""
        p = self.dyn.params
        px, py, pz = float(x[0]), float(x[1]), float(x[2])
        q1 = np.arctan2(py, px)
        r = np.hypot(px, py)
        dz = pz - p.l1
        d2 = r * r + dz * dz
        cos_q3 = np.clip((d2 - p.l2**2 - p.l3**2) / (2.0 * p.l2 * p.l3), -1.0, 1.0)
        q3 = np.arccos(cos_q3)  # positive elbow bend
        # r = l2*sin(q2) + l3*sin(q2+q3), dz = l2*cos(q2) + l3*cos(q2+q3)
        k1 = p.l2 + p.l3 * np.cos(q3)
        k2 = p.l3 * np.sin(q3)
        # [dz; r] = [[cos(q2), -sin(q2)], [sin(q2), cos(q2)]] @ [k1; k2]
        q2 = np.arctan2(r, dz) - np.arctan2(k2, k1)
        return np.array([q1, q2, q3], dtype=np.float64)

    def initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        ref0 = self(0.0)
        return ref0.q_d.copy(), ref0.qd_d.copy()

    def __call__(self, t: float) -> TrajectoryReference:
        c, s = np.cos(self.omega * t), np.sin(self.omega * t)
        w = self.omega
        R = self.radius
        # Circle in Y-Z plane with slight X modulation for full 3-DOF excitation
        x_d = self.center + np.array([0.025 * s, R * c, R * s], dtype=np.float64)
        xd_d = np.array([0.025 * w * c, -R * w * s, R * w * c], dtype=np.float64)
        xdd_d = np.array([-0.025 * w * w * s, -R * w * w * c, -R * w * w * s], dtype=np.float64)

        q_d = self.solve_ik_exact(x_d)
        J = self.dyn.jacobian_v(q_d)
        qd_d = np.linalg.solve(J, xd_d)
        Jd = self.dyn.jacobian_v_dot(q_d, qd_d)
        qdd_d = np.linalg.solve(J, xdd_d - Jd @ qd_d)
        return TrajectoryReference(t=t, q_d=q_d, qd_d=qd_d, qdd_d=qdd_d, x_d=x_d, xd_d=xd_d, xdd_d=xdd_d)
