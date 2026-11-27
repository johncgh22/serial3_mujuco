"""Classical joint-space controllers: PID (with anti-windup) and PD + Gravity."""

from __future__ import annotations

import numpy as np

from .base import BaseController, TrajectoryReference
from .dynamics import AnalyticalDynamics


class PIDController(BaseController):
    """Independent joint-space PID controller with conditional anti-windup.

    Model-free baseline:
        tau = Kp * e + Ki * integral(e) + Kd * e_dot
    """

    name = "PID (Anti-Windup)"
    short_name = "PID"
    control_space = "joint"

    def __init__(
        self,
        kp: np.ndarray | list[float] = (1.8, 3.5, 1.2),
        ki: np.ndarray | list[float] = (4.0, 12.0, 4.0),
        kd: np.ndarray | list[float] = (0.12, 0.28, 0.08),
        integral_limit: float = 0.8,
        torque_limit: float = 1.5,
    ):
        super().__init__(torque_limit=torque_limit)
        self.kp = np.asarray(kp, dtype=np.float64)
        self.ki = np.asarray(ki, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.integral_limit = float(integral_limit)
        self._e_int = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self._e_int[:] = 0.0

    def compute(
        self,
        t: float,
        q: np.ndarray,
        qd: np.ndarray,
        ref: TrajectoryReference,
        dt: float,
    ) -> np.ndarray:
        e = ref.q_d - q
        ed = ref.qd_d - qd

        # Proposed unsaturated control
        e_int_next = np.clip(self._e_int + e * dt, -self.integral_limit, self.integral_limit)
        tau_unsat = self.kp * e + self.ki * e_int_next + self.kd * ed
        tau = self.clip_torque(tau_unsat)

        # Conditional integration anti-windup: freeze integrator on saturated axes pushing further
        saturated_same_dir = (np.abs(tau_unsat) > self.torque_limit) & (np.sign(tau_unsat) == np.sign(e))
        self._e_int = np.where(saturated_same_dir, self._e_int, e_int_next)
        return tau


class PDGravityController(BaseController):
    """PD + Analytical Gravity Compensation: tau = Kp * e + Kd * e_dot + G(q).

    Eliminates steady-state gravity sag without integral windup; globally
    asymptotically stable for set-point regulation (Takegaki & Arimoto, 1981).
    """

    name = "PD + Gravity"
    short_name = "PD+G"
    control_space = "joint"

    def __init__(
        self,
        dynamics: AnalyticalDynamics,
        kp: np.ndarray | list[float] = (1.8, 3.5, 1.2),
        kd: np.ndarray | list[float] = (0.12, 0.28, 0.08),
        torque_limit: float = 1.5,
    ):
        super().__init__(torque_limit=torque_limit)
        self.dyn = dynamics
        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)

    def compute(
        self,
        t: float,
        q: np.ndarray,
        qd: np.ndarray,
        ref: TrajectoryReference,
        dt: float,
    ) -> np.ndarray:
        e = ref.q_d - q
        ed = ref.qd_d - qd
        tau = self.kp * e + self.kd * ed + self.dyn.gravity_vector(q)
        return self.clip_torque(tau)
