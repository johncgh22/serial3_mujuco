"""Real-Time Iteration LTV-MPC & SINDy-MPC with hard physical constraints.

Solves a finite-horizon constrained optimal control problem over state
x = [q1, q2, q3, qd1, qd2, qd3]^T and torque u = [tau1, tau2, tau3]^T:

    min_{u_0..u_{N-1}}  sum_{k=0}^{N-1} (||x_k - x_ref_k||_Q^2
                                        + ||u_k - u_ref_k||_R^2
                                        + ||u_k - u_{k-1}||_{R_du}^2)
                        + ||x_N - x_ref_N||_{Q_f}^2
    s.t.  x_{k+1} = f_model(x_k, u_k + d_hat)   (RK4 linearized along warm start)
          u_min <= u_k <= u_max                 (AX-12A +/-1.5 N*m torque limits)
          -du_max <= u_k - u_{k-1} <= du_max    (torque slew-rate smoothness)
          x_min <= x_k <= x_max                 (joint angle & AX-12A speed limits)

Works identically with either:
  - `AnalyticalDynamics` (Nominal Model-Based MPC), or
  - `SINDyDynamics` (Data-Driven SINDy-MPC learned from rollouts).

Includes an optional Luenberger disturbance observer (`use_disturbance_observer`)
for offset-free tracking under unexpected external impulse/step disturbances,
and a lightweight pure-NumPy ADMM QP solver (zero external solver dependencies).
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .base import BaseController, TrajectoryReference


class DynamicsPredictor(Protocol):
    """Minimal interface required by LTVMPCController (Analytical or SINDy)."""

    def mass_matrix(self, q: np.ndarray) -> np.ndarray: ...
    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray: ...
    def forward_dynamics(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray) -> np.ndarray: ...


def _solve_qp_admm(
    H: np.ndarray,
    g: np.ndarray,
    A: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    u_init: np.ndarray,
    rho: float = 2.0,
    max_iter: int = 80,
    tol: float = 1e-4,
) -> np.ndarray:
    """Fast pure-NumPy OSQP-style ADMM solver for strictly convex QP:

        min_u  0.5 * u^T H u + g^T u   s.t.   lb <= A u <= ub
    """
    m = A.shape[0]
    K = H + rho * (A.T @ A)
    K_fac = cho_factor(K, lower=True, check_finite=False)

    u = u_init.copy()
    z = np.clip(A @ u, lb, ub)
    y = np.zeros(m, dtype=np.float64)

    for _ in range(max_iter):
        rhs = -g + rho * (A.T @ (z - y))
        u = cho_solve(K_fac, rhs, check_finite=False)
        Au = A @ u
        z_new = np.clip(Au + y, lb, ub)
        r_pri = Au - z_new
        r_dual = rho * (z_new - z)
        y += r_pri
        z = z_new
        if np.max(np.abs(r_pri)) < tol and np.max(np.abs(r_dual)) < tol:
            break

    return u


class LTVMPCController(BaseController):
    """Constrained Linear-Time-Varying Model Predictive Controller (LTV-MPC / SINDy-MPC)."""

    name = "LTV-MPC"
    short_name = "MPC"
    control_space = "joint"

    def __init__(
        self,
        dynamics: DynamicsPredictor,
        trajectory_fn: Callable[[float], TrajectoryReference] | None = None,
        horizon: int = 12,
        mpc_dt: float = 0.015,
        q_pos_weights: tuple[float, float, float] = (260.0, 320.0, 220.0),
        q_vel_weights: tuple[float, float, float] = (8.0, 12.0, 6.0),
        r_tau_weights: tuple[float, float, float] = (0.15, 0.15, 0.15),
        r_dtau_weights: tuple[float, float, float] = (0.30, 0.30, 0.30),
        terminal_scale: float = 4.0,
        torque_limit: float | np.ndarray = 1.5,
        du_limit: float = 0.35,
        q_min: tuple[float, float, float] = (-2.60, -1.55, -2.05),
        q_max: tuple[float, float, float] = (2.60, 1.55, 2.05),
        qd_max: float = 6.2,
        use_disturbance_observer: bool = True,
        dob_lv: float = 35.0,
        dob_ld: float = 280.0,
        name: str | None = None,
        short_name: str | None = None,
    ):
        super().__init__(torque_limit=torque_limit)
        self.dyn = dynamics
        self.trajectory_fn = trajectory_fn
        self.N = int(horizon)
        self.mpc_dt = float(mpc_dt)

        self.Q_diag = np.array([*q_pos_weights, *q_vel_weights], dtype=np.float64)
        self.Qf_diag = self.Q_diag * float(terminal_scale)
        self.R_diag = np.asarray(r_tau_weights, dtype=np.float64)
        self.Rdu_diag = np.asarray(r_dtau_weights, dtype=np.float64)
        self.du_limit = float(du_limit)
        self.x_min = np.array([*q_min, -qd_max, -qd_max, -qd_max], dtype=np.float64)
        self.x_max = np.array([*q_max, qd_max, qd_max, qd_max], dtype=np.float64)

        self.use_dob = bool(use_disturbance_observer)
        self.dob_lv = float(dob_lv)
        self.dob_ld = float(dob_ld)

        if name is not None:
            self.name = name
        if short_name is not None:
            self.short_name = short_name

        # Precompute constant block-diagonal cost weights and difference matrix D
        self._Q_bar = np.concatenate([np.tile(self.Q_diag, self.N - 1), self.Qf_diag])
        self._R_bar = np.tile(self.R_diag, self.N)
        self._Rdu_bar = np.tile(self.Rdu_diag, self.N)
        self._D = np.eye(3 * self.N, dtype=np.float64)
        for k in range(1, self.N):
            self._D[3 * k : 3 * (k + 1), 3 * (k - 1) : 3 * k] = -np.eye(3)

        self.reset()

    def reset(self) -> None:
        self._u_plan = np.zeros((self.N, 3), dtype=np.float64)
        self._u_last = np.zeros(3, dtype=np.float64)
        self._last_solve_t = -1e9
        self._v_hat: np.ndarray | None = None
        self._d_hat = np.zeros(3, dtype=np.float64)

    # ---- Discrete RK4 dynamics & linearization ----
    def _f_cont(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        q = x[:3]
        qd = np.clip(x[3:], -30.0, 30.0)
        qdd = self.dyn.forward_dynamics(q, qd, u + self._d_hat)
        qdd = np.clip(qdd, -500.0, 500.0)
        return np.concatenate([qd, qdd])

    def _rk4_step(self, x: np.ndarray, u: np.ndarray, substeps: int = 2) -> np.ndarray:
        h = self.mpc_dt / substeps
        xs = x
        for _ in range(substeps):
            k1 = self._f_cont(xs, u)
            k2 = self._f_cont(xs + 0.5 * h * k1, u)
            k3 = self._f_cont(xs + 0.5 * h * k2, u)
            k4 = self._f_cont(xs + h * k3, u)
            xs = xs + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return xs

    def _linearize_step(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linearize x_{k+1} = f(x, u) around (x, u): x_{k+1} ≈ A*x_k + B*u_k + c."""
        f0 = self._rk4_step(x, u)
        eps_x = 1e-5
        eps_u = 1e-5
        A = np.zeros((6, 6), dtype=np.float64)
        B = np.zeros((6, 3), dtype=np.float64)
        for i in range(6):
            dx = np.zeros(6)
            dx[i] = eps_x
            A[:, i] = (self._rk4_step(x + dx, u) - f0) / eps_x
        for j in range(3):
            du = np.zeros(3)
            du[j] = eps_u
            B[:, j] = (self._rk4_step(x, u + du) - f0) / eps_u
        c = f0 - A @ x - B @ u
        return A, B, c

    def _build_reference_horizon(self, t: float, ref: TrajectoryReference) -> tuple[np.ndarray, np.ndarray]:
        """Build stacked state reference X_ref (6*N,) and feedforward torque U_ref (3*N,)."""
        X_ref = np.zeros(6 * self.N, dtype=np.float64)
        U_ref = np.zeros(3 * self.N, dtype=np.float64)
        for k in range(self.N):
            tk = t + k * self.mpc_dt
            tk_next = t + (k + 1) * self.mpc_dt
            if self.trajectory_fn is not None:
                r_k = self.trajectory_fn(tk)
                r_next = self.trajectory_fn(tk_next)
            else:
                # Constant-acceleration extrapolation when no look-ahead trajectory is provided
                r_k = ref
                dt_k = (k + 1) * self.mpc_dt
                r_next = TrajectoryReference(
                    t=tk_next,
                    q_d=ref.q_d + ref.qd_d * dt_k,
                    qd_d=ref.qd_d,
                    qdd_d=np.zeros(3),
                    x_d=ref.x_d,
                    xd_d=ref.xd_d,
                    xdd_d=np.zeros(3),
                )
            X_ref[6 * k : 6 * (k + 1)] = np.concatenate([r_next.q_d, r_next.qd_d])
            tau_ff = self.dyn.inverse_dynamics(r_k.q_d, r_k.qd_d, r_k.qdd_d) - self._d_hat
            U_ref[3 * k : 3 * (k + 1)] = self.clip_torque(tau_ff)
        return X_ref, U_ref

    def _update_dob(self, q: np.ndarray, qd: np.ndarray, dt: float) -> None:
        """Update Luenberger torque disturbance observer without differentiating qd."""
        if not self.use_dob:
            return
        if self._v_hat is None:
            self._v_hat = qd.copy()
            return
        v_err = qd - self._v_hat
        qdd_hat = self.dyn.forward_dynamics(q, qd, self._u_last + self._d_hat)
        M = self.dyn.mass_matrix(q)
        self._v_hat = self._v_hat + (qdd_hat + self.dob_lv * v_err) * dt
        self._d_hat = np.clip(self._d_hat + (self.dob_ld * (M @ v_err)) * dt, -1.2, 1.2)

    def _solve_mpc(self, t: float, x0: np.ndarray, ref: TrajectoryReference) -> np.ndarray:
        N = self.N
        X_ref, U_ref = self._build_reference_horizon(t, ref)

        # Warm-start initial guess for ADMM
        if np.all(self._u_plan == 0.0):
            u_init = U_ref.copy()
        else:
            u_init = np.vstack([self._u_plan[1:], self._u_plan[-1:]]).ravel()

        # Linearize around the smooth reference trajectory (prevents open-loop RK4 drift
        # on low-inertia zero-damping models while remaining accurate near the tracked path)
        x_nom = np.zeros((N, 6), dtype=np.float64)
        x_nom[0] = x0
        if N > 1:
            x_nom[1:] = X_ref[: 6 * (N - 1)].reshape(N - 1, 6)
        u_nom = U_ref.reshape(N, 3)

        # Condense LTV dynamics: X = x_free + Gamma * U
        A_list, B_list, c_list = [], [], []
        for k in range(N):
            Ak, Bk, ck = self._linearize_step(x_nom[k], u_nom[k])
            A_list.append(Ak)
            B_list.append(Bk)
            c_list.append(ck)

        Gamma = np.zeros((6 * N, 3 * N), dtype=np.float64)
        x_free = np.zeros(6 * N, dtype=np.float64)
        xk_free = x0.copy()
        for k in range(N):
            Ak, Bk, ck = A_list[k], B_list[k], c_list[k]
            xk_free = Ak @ xk_free + ck
            x_free[6 * k : 6 * (k + 1)] = xk_free
            Gamma[6 * k : 6 * (k + 1), 3 * k : 3 * (k + 1)] = Bk
            for j in range(k):
                Gamma[6 * k : 6 * (k + 1), 3 * j : 3 * (j + 1)] = (
                    Ak @ Gamma[6 * (k - 1) : 6 * k, 3 * j : 3 * (j + 1)]
                )

        # Quadratic cost: 0.5 * U^T H U + g^T U
        err_free = x_free - X_ref
        Q_Gamma = self._Q_bar[:, None] * Gamma
        d0 = np.zeros(3 * N, dtype=np.float64)
        d0[:3] = self._u_last
        Rdu_D = self._Rdu_bar[:, None] * self._D

        H = Gamma.T @ Q_Gamma + np.diag(self._R_bar) + self._D.T @ Rdu_D
        H = 0.5 * (H + H.T) + 1e-8 * np.eye(3 * N)
        g = Gamma.T @ (self._Q_bar * err_free) - self._R_bar * U_ref - self._D.T @ (self._Rdu_bar * d0)

        # Linear constraints: [I; D; Gamma] U in [lb, ub]
        u_lb = np.tile(-self.torque_limit, N)
        u_ub = np.tile(self.torque_limit, N)

        du_lb = d0 - self.du_limit
        du_ub = d0 + self.du_limit

        # Relax state constraints gently if current state is near/outside boundary to ensure feasibility
        x_lb = np.tile(self.x_min, N) - x_free
        x_ub = np.tile(self.x_max, N) - x_free
        # Only enforce state constraints from k=1..N-1 (avoid infeasibility at k=0 from initial state)
        x_lb[:6] = -1e3
        x_ub[:6] = 1e3

        A_ineq = np.vstack([np.eye(3 * N), self._D, Gamma])
        lb = np.concatenate([u_lb, du_lb, x_lb])
        ub = np.concatenate([u_ub, du_ub, x_ub])

        U_opt = _solve_qp_admm(H, g, A_ineq, lb, ub, u_init)
        self._u_plan = U_opt.reshape(N, 3)
        return self.clip_torque(self._u_plan[0])

    def compute(
        self,
        t: float,
        q: np.ndarray,
        qd: np.ndarray,
        ref: TrajectoryReference,
        dt: float,
    ) -> np.ndarray:
        self._update_dob(q, qd, dt)

        # Solve MPC at mpc_dt cadence (zero-order hold between MPC ticks)
        if (t - self._last_solve_t) >= self.mpc_dt - 1e-9 or self._last_solve_t < 0.0:
            x0 = np.concatenate([q, qd])
            self._u_last = self._solve_mpc(t, x0, ref)
            self._last_solve_t = t

        return self._u_last.copy()
