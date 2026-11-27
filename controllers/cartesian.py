"""Task-space (Cartesian) controllers: Operational Space Control & Impedance."""

from __future__ import annotations

import numpy as np

from .base import BaseController, TrajectoryReference
from .dynamics import AnalyticalDynamics


class OperationalSpaceController(BaseController):
    """Singularity-robust Operational Space Control (Khatib, 1987) for 3D (x, y, z).

    Because the 3-DOF RRR arm has a square 3x3 translational Jacobian J_v(q),
    end-effector Cartesian dynamics can be decoupled directly without IK:

        Lambda(q) = (J_v * M^-1 * J_v^T + damping^2 * I)^-1
        a_cmd     = xdd_d + Kp * (x_d - x) + Kd * (xd_d - xd) - J_v_dot * qd
        tau       = J_v^T * (Lambda * a_cmd) + C(q, qd) * qd + G(q)
    """

    name = "Operational Space (OSC)"
    short_name = "OSC"
    control_space = "cartesian"

    def __init__(
        self,
        dynamics: AnalyticalDynamics,
        kp: np.ndarray | list[float] = (320.0, 320.0, 320.0),
        kd: np.ndarray | list[float] = (36.0, 36.0, 36.0),
        damping: float = 1e-2,
        torque_limit: float = 1.5,
    ):
        super().__init__(torque_limit=torque_limit)
        self.dyn = dynamics
        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.damping = float(damping)

    def compute(
        self,
        t: float,
        q: np.ndarray,
        qd: np.ndarray,
        ref: TrajectoryReference,
        dt: float,
    ) -> np.ndarray:
        x = self.dyn.fk(q)
        J = self.dyn.jacobian_v(q)
        Jd = self.dyn.jacobian_v_dot(q, qd)
        xd = J @ qd

        M = self.dyn.mass_matrix(q)
        C = self.dyn.coriolis_matrix(q, qd)
        G = self.dyn.gravity_vector(q)

        M_inv_JT = np.linalg.solve(M, J.T)
        # Task-space mobility matrix J * M^-1 * J^T with adaptive singularity damping
        A = J @ M_inv_JT + (self.damping ** 2) * np.eye(3)

        a_cmd = ref.xdd_d + self.kp * (ref.x_d - x) + self.kd * (ref.xd_d - xd) - Jd @ qd
        f_task = np.linalg.solve(A, a_cmd)
        tau = J.T @ f_task + C @ qd + G
        return self.clip_torque(tau)


class CartesianImpedanceController(BaseController):
    """Passive Cartesian Spring-Damper Impedance Control + Gravity Compensation.

        tau = J_v(q)^T * (K_x * (x_d - x) + D_x * (xd_d - xd)) + G(q)

    Naturally compliant to external forces and requires no matrix inversion
    (completely safe even right at kinematic singularities).
    """

    name = "Cartesian Impedance"
    short_name = "Impedance"
    control_space = "cartesian"

    def __init__(
        self,
        dynamics: AnalyticalDynamics,
        kx: np.ndarray | list[float] = (45.0, 45.0, 45.0),
        dx: np.ndarray | list[float] = (3.5, 3.5, 3.5),
        torque_limit: float = 1.5,
    ):
        super().__init__(torque_limit=torque_limit)
        self.dyn = dynamics
        self.kx = np.asarray(kx, dtype=np.float64)
        self.dx = np.asarray(dx, dtype=np.float64)

    def compute(
        self,
        t: float,
        q: np.ndarray,
        qd: np.ndarray,
        ref: TrajectoryReference,
        dt: float,
    ) -> np.ndarray:
        x = self.dyn.fk(q)
        J = self.dyn.jacobian_v(q)
        xd = J @ qd
        f_task = self.kx * (ref.x_d - x) + self.dx * (ref.xd_d - xd)
        tau = J.T @ f_task + self.dyn.gravity_vector(q)
        return self.clip_torque(tau)
