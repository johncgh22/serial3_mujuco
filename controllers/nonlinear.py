"""Model-based nonlinear controllers: Computed-Torque (CTC) & Slotine-Li Adaptive."""

from __future__ import annotations

import numpy as np

from .base import BaseController, TrajectoryReference
from .dynamics import AnalyticalDynamics


class ComputedTorqueController(BaseController):
    """Inverse-dynamics feedback linearization (Computed-Torque Control):

        tau = M(q) * (qdd_d + Kp * e + Kd * e_dot) + C(q, qd) * qd + G(q)

    Uses the nominal analytical model (independent of MuJoCo internals), so
    perturbed-plant benchmarks genuinely test robustness to model mismatch.
    """

    name = "Computed-Torque (CTC)"
    short_name = "CTC"
    control_space = "joint"

    def __init__(
        self,
        dynamics: AnalyticalDynamics,
        kp: np.ndarray | list[float] = (320.0, 320.0, 320.0),
        kd: np.ndarray | list[float] = (36.0, 36.0, 36.0),
        torque_limit: float = 1.5,
        name: str | None = None,
        short_name: str | None = None,
    ):
        super().__init__(torque_limit=torque_limit)
        self.dyn = dynamics
        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        if name is not None:
            self.name = name
        if short_name is not None:
            self.short_name = short_name

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
        v = ref.qdd_d + self.kp * e + self.kd * ed
        tau = self.dyn.inverse_dynamics(q, qd, v)
        return self.clip_torque(tau)


class SlotineLiAdaptiveController(BaseController):
    """Slotine-Li passivity-based adaptive manipulator controller (1987).

    Exploits the exact skew-symmetry of (M_dot - 2*C) and linearity in the
    10 minimal base inertial parameters (+ 6 optional viscous/Coulomb friction
    parameters for AX-12A gearbox adaptation):

        qd_r  = qd_d + Lambda * e
        qdd_r = qdd_d + Lambda * e_dot
        s     = qd - qd_r = -(e_dot + Lambda * e)
        tau   = Y(q, qd, qd_r, qdd_r) * theta_hat - K_D * s
        theta_hat_dot = -Gamma * Y^T * s - sigma * (theta_hat - theta_0)
    """

    name = "Slotine-Li Adaptive"
    short_name = "Adaptive"
    control_space = "joint"

    def __init__(
        self,
        dynamics: AnalyticalDynamics,
        lam: np.ndarray | list[float] = (14.0, 14.0, 14.0),
        kd: np.ndarray | list[float] = (0.12, 0.25, 0.06),
        gamma_inertial: float = 0.08,
        gamma_gravity: float = 0.45,
        gamma_friction: float = 1.80,
        include_friction: bool = True,
        sigma_leak: float = 1e-3,
        torque_limit: float = 1.5,
    ):
        super().__init__(torque_limit=torque_limit)
        self.dyn = dynamics
        self.lam = np.asarray(lam, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.include_friction = bool(include_friction)
        self.sigma_leak = float(sigma_leak)

        n_params = 16 if include_friction else 10
        self.theta_0 = np.zeros(n_params, dtype=np.float64)
        self.theta_0[:10] = dynamics.theta_nominal

        gamma = np.full(n_params, gamma_inertial, dtype=np.float64)
        gamma[8:10] = gamma_gravity
        if include_friction:
            gamma[10:16] = gamma_friction
        self.gamma = gamma

        self.theta_hat = self.theta_0.copy()
        self.theta_history: list[np.ndarray] = []

    def reset(self) -> None:
        self.theta_hat = self.theta_0.copy()
        self.theta_history = []

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

        qd_r = ref.qd_d + self.lam * e
        qdd_r = ref.qdd_d + self.lam * ed
        s = qd - qd_r  # = -(ed + lam * e)

        Y = self.dyn.regressor(q, qd, qd_r, qdd_r, include_friction=self.include_friction)
        tau_unsat = Y @ self.theta_hat - self.kd * s
        tau = self.clip_torque(tau_unsat)

        # Update parameter estimates (pause adaptation during torque saturation)
        if np.all(np.abs(tau_unsat) < self.torque_limit):
            theta_dot = -self.gamma * (Y.T @ s) - self.sigma_leak * (self.theta_hat - self.theta_0)
            self.theta_hat = self.theta_hat + theta_dot * dt
            # Keep physical positivity for pure diagonal inertias & friction terms
            self.theta_hat[0:5] = np.maximum(self.theta_hat[0:5], 0.0)
            self.theta_hat[6:10] = np.maximum(self.theta_hat[6:10], 0.0)
            if self.include_friction:
                self.theta_hat[10:16] = np.clip(self.theta_hat[10:16], 0.0, 2.0)

        self.theta_history.append(self.theta_hat.copy())
        return tau
