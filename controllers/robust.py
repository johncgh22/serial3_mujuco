"""Robust nonlinear controllers: Boundary-Layer SMC and Super-Twisting (2nd-order SMC)."""

from __future__ import annotations

import numpy as np

from .base import BaseController, TrajectoryReference
from .dynamics import AnalyticalDynamics


class SlidingModeController(BaseController):
    """First-order Integral Sliding Mode Control with smooth tanh boundary layer.

    Sliding surface:  s = e_dot + Lambda * e + Lambda_i * integral(e)
    Control law:
        tau = M_hat(q)*(qdd_d + Lambda*e_dot + Lambda_i*e) + C_hat(q,qd)*qd + G_hat(q)
              + K_s * tanh(s / phi) + K_d * s

    The inertia-scaled boundary layer `phi` guarantees chatter-free discrete-time
    stability even on low-inertia bare links (`M33 = 6.4e-5 kg*m^2`), while the
    integral sliding term eliminates steady-state error under unknown payloads
    and AX-12A gearbox friction.
    """

    name = "Sliding Mode (Boundary Layer)"
    short_name = "SMC-BL"
    control_space = "joint"

    def __init__(
        self,
        dynamics: AnalyticalDynamics,
        lam: np.ndarray | list[float] = (14.0, 14.0, 14.0),
        lam_i: np.ndarray | list[float] = (45.0, 45.0, 160.0),
        k_switch: np.ndarray | list[float] = (0.35, 0.55, 0.38),
        k_linear: np.ndarray | list[float] = (0.04, 0.10, 0.012),
        phi: np.ndarray | list[float] = (0.70, 0.35, 6.20),
        integral_limit: float = 0.5,
        torque_limit: float = 1.5,
    ):
        super().__init__(torque_limit=torque_limit)
        self.dyn = dynamics
        self.lam = np.asarray(lam, dtype=np.float64)
        self.lam_i = np.asarray(lam_i, dtype=np.float64)
        self.k_switch = np.asarray(k_switch, dtype=np.float64)
        self.k_linear = np.asarray(k_linear, dtype=np.float64)
        self.phi = np.asarray(phi, dtype=np.float64)
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
        e_int_next = np.clip(self._e_int + e * dt, -self.integral_limit, self.integral_limit)
        s = ed + self.lam * e + self.lam_i * e_int_next

        tau_eq = self.dyn.inverse_dynamics(q, qd, ref.qdd_d + self.lam * ed + self.lam_i * e)
        tau_rob = self.k_switch * np.tanh(s / self.phi) + self.k_linear * s
        tau_unsat = tau_eq + tau_rob
        tau = self.clip_torque(tau_unsat)

        sat_same = (np.abs(tau_unsat) > self.torque_limit) & (np.sign(tau_unsat) == np.sign(e))
        self._e_int = np.where(sat_same, self._e_int, e_int_next)
        return tau


class SuperTwistingSMC(BaseController):
    """Second-order Super-Twisting Sliding Mode Controller (Levant).

    Produces a *continuous* torque command while achieving finite-time sliding
    convergence and exact integral rejection of smooth matched disturbances
    (unknown payloads, viscous damping, and gearbox friction):

        s     = e_dot + Lambda * e
        u_st  = k1 * |s|^(1/2) * tanh(s / eps) + z
        z_dot = k2 * tanh(s / eps)
        tau   = tau_eq + u_st + k_lin * s

    Proportional gains `k1`, `k_linear` are scaled to each link's nominal inertia
    to prevent discrete-time chattering on the low-inertia wrist/elbow link while
    `k2` smoothly absorbs large unmodeled payloads and AX-12A gearbox friction.
    """

    name = "Super-Twisting SMC"
    short_name = "ST-SMC"
    control_space = "joint"

    def __init__(
        self,
        dynamics: AnalyticalDynamics,
        lam: np.ndarray | list[float] = (14.0, 14.0, 14.0),
        k1: np.ndarray | list[float] = (0.10, 0.25, 0.012),
        k2: np.ndarray | list[float] = (1.60, 2.80, 1.40),
        k_linear: np.ndarray | list[float] = (0.04, 0.12, 0.012),
        eps: np.ndarray | list[float] = (0.06, 0.06, 0.10),
        z_limit: float = 1.2,
        torque_limit: float = 1.5,
    ):
        super().__init__(torque_limit=torque_limit)
        self.dyn = dynamics
        self.lam = np.asarray(lam, dtype=np.float64)
        self.k1 = np.asarray(k1, dtype=np.float64)
        self.k2 = np.asarray(k2, dtype=np.float64)
        self.k_linear = np.asarray(k_linear, dtype=np.float64)
        self.eps = np.asarray(eps, dtype=np.float64)
        self.z_limit = float(z_limit)
        self._z = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self._z[:] = 0.0

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
        s = ed + self.lam * e

        sign_s = np.tanh(s / self.eps)
        tau_eq = self.dyn.inverse_dynamics(q, qd, ref.qdd_d + self.lam * ed)
        u_prop = self.k1 * np.sqrt(np.abs(s) + 1e-12) * sign_s + self.k_linear * s

        z_next = np.clip(self._z + self.k2 * sign_s * dt, -self.z_limit, self.z_limit)
        tau_unsat = tau_eq + u_prop + z_next
        tau = self.clip_torque(tau_unsat)

        # Anti-windup on super-twisting integral state z
        sat_same = (np.abs(tau_unsat) > self.torque_limit) & (np.sign(tau_unsat) == np.sign(s))
        self._z = np.where(sat_same, self._z, z_next)
        return tau
