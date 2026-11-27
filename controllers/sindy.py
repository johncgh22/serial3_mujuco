"""Physics-Informed Inverse-Dynamics SINDy for the 3-DOF serial manipulator.

Builds on `serial3_sindytau_v3.m` with three key upgrades for exact robotic
system identification:
  1. Well-conditioned physics-informed candidate library Theta(q, qd, qdd)
     combining trig-coupled inertia terms, Coriolis/centrifugal products,
     gravity terms, viscous + smooth Coulomb friction, and non-physical
     distractors so STLSQ genuinely discovers sparse governing equations.
  2. Column-normalized STLSQ (Sequential Thresholded Least Squares) so tiny
     physical inertias (1e-5 to 1e-3 kg*m^2) are not wiped out by an unscaled
     threshold `lambda`, followed by an unpenalized OLS refit on the active
     support.
  3. Direct algebraic reconstruction of M_hat(q) and b_hat(q, qd) from the
     sparse coefficient matrix Xi, yielding an exact forward dynamics predictor
     `qdd = M_hat(q)^-1 * (tau - b_hat(q, qd))` for SINDy-CTC and SINDy-MPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CandidateTerm:
    """One column of the SINDy inverse-dynamics library Theta(q, qd, qdd)."""
    name: str
    kind: str  # "accel" (multiplies qdd[accel_idx]) or "bias" (depends only on q, qd)
    accel_idx: int  # 0, 1, 2 for kind=="accel", else -1
    eval_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    # For accel terms, basis_fn(q) evaluates phi(q) where column = phi(q) * qdd[accel_idx]
    basis_fn: Callable[[np.ndarray], np.ndarray] | None = None


def build_candidate_library(coulomb_eps: float = 0.05) -> list[CandidateTerm]:
    """Construct the linearly independent physics-informed candidate library + distractors.

    All functions accept batched arrays q, qd, qdd of shape (N, 3) and return (N,).
    """
    terms: list[CandidateTerm] = []

    def add_accel(name: str, idx: int, basis_fn: Callable[[np.ndarray], np.ndarray]):
        terms.append(
            CandidateTerm(
                name=name,
                kind="accel",
                accel_idx=idx,
                eval_fn=lambda q, qd, qdd, i=idx, bf=basis_fn: bf(q) * qdd[:, i],
                basis_fn=basis_fn,
            )
        )

    def add_bias(name: str, fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]):
        terms.append(CandidateTerm(name=name, kind="bias", accel_idx=-1, eval_fn=fn))

    # ---- 1. Acceleration-coupled inertia candidates M(q) * qdd ----
    # Linearly independent basis: {1, sin^2(q2), sin^2(q2+q3), 2*sin(q2)*sin(q2+q3)}
    add_accel("qdd1", 0, lambda q: np.ones(len(q)))
    add_accel("sin^2(q2)*qdd1", 0, lambda q: np.sin(q[:, 1]) ** 2)
    add_accel("sin^2(q2+q3)*qdd1", 0, lambda q: np.sin(q[:, 1] + q[:, 2]) ** 2)
    add_accel("2*sin(q2)*sin(q2+q3)*qdd1", 0, lambda q: 2.0 * np.sin(q[:, 1]) * np.sin(q[:, 1] + q[:, 2]))

    add_accel("qdd2", 1, lambda q: np.ones(len(q)))
    add_accel("cos(q3)*qdd2", 1, lambda q: np.cos(q[:, 2]))
    add_accel("qdd3", 2, lambda q: np.ones(len(q)))
    add_accel("cos(q3)*qdd3", 2, lambda q: np.cos(q[:, 2]))

    # Non-physical inertia distractors (STLSQ should prune these to 0)
    add_accel("sin(q2)*qdd2", 1, lambda q: np.sin(q[:, 1]))
    add_accel("sin(q3)*qdd3", 2, lambda q: np.sin(q[:, 2]))

    # ---- 2. Coriolis / centrifugal candidates C(q, qd) * qd ----
    add_bias("sin(2*q2)*qd1*qd2", lambda q, qd, qdd: np.sin(2.0 * q[:, 1]) * qd[:, 0] * qd[:, 1])
    add_bias("sin(2*(q2+q3))*qd1*qd2", lambda q, qd, qdd: np.sin(2.0 * (q[:, 1] + q[:, 2])) * qd[:, 0] * qd[:, 1])
    add_bias("sin(2*(q2+q3))*qd1*qd3", lambda q, qd, qdd: np.sin(2.0 * (q[:, 1] + q[:, 2])) * qd[:, 0] * qd[:, 2])
    add_bias("sin(2*q2+q3)*qd1*qd2", lambda q, qd, qdd: np.sin(2.0 * q[:, 1] + q[:, 2]) * qd[:, 0] * qd[:, 1])
    add_bias("sin(q2)*cos(q2+q3)*qd1*qd3", lambda q, qd, qdd: np.sin(q[:, 1]) * np.cos(q[:, 1] + q[:, 2]) * qd[:, 0] * qd[:, 2])

    add_bias("sin(2*q2)*qd1^2", lambda q, qd, qdd: np.sin(2.0 * q[:, 1]) * (qd[:, 0] ** 2))
    add_bias("sin(2*(q2+q3))*qd1^2", lambda q, qd, qdd: np.sin(2.0 * (q[:, 1] + q[:, 2])) * (qd[:, 0] ** 2))
    add_bias("sin(2*q2+q3)*qd1^2", lambda q, qd, qdd: np.sin(2.0 * q[:, 1] + q[:, 2]) * (qd[:, 0] ** 2))
    add_bias("sin(q2)*cos(q2+q3)*qd1^2", lambda q, qd, qdd: np.sin(q[:, 1]) * np.cos(q[:, 1] + q[:, 2]) * (qd[:, 0] ** 2))

    add_bias("sin(q3)*qd2*qd3", lambda q, qd, qdd: np.sin(q[:, 2]) * qd[:, 1] * qd[:, 2])
    add_bias("sin(q3)*qd3^2", lambda q, qd, qdd: np.sin(q[:, 2]) * (qd[:, 2] ** 2))
    add_bias("sin(q3)*qd2^2", lambda q, qd, qdd: np.sin(q[:, 2]) * (qd[:, 1] ** 2))

    # Non-physical Coriolis distractors
    add_bias("qd1*qd2", lambda q, qd, qdd: qd[:, 0] * qd[:, 1])
    add_bias("qd2*qd3", lambda q, qd, qdd: qd[:, 1] * qd[:, 2])
    add_bias("qd1*qd3", lambda q, qd, qdd: qd[:, 0] * qd[:, 2])

    # ---- 3. Gravity & static configuration candidates ----
    add_bias("sin(q2)", lambda q, qd, qdd: np.sin(q[:, 1]))
    add_bias("sin(q2+q3)", lambda q, qd, qdd: np.sin(q[:, 1] + q[:, 2]))
    add_bias("cos(q2)", lambda q, qd, qdd: np.cos(q[:, 1]))
    add_bias("cos(q2+q3)", lambda q, qd, qdd: np.cos(q[:, 1] + q[:, 2]))

    # Non-physical static distractors (avoid linear q_i which is collinear with sin(q_i))
    add_bias("1", lambda q, qd, qdd: np.ones(len(q)))
    add_bias("sin(q1)", lambda q, qd, qdd: np.sin(q[:, 0]))
    add_bias("cos(q1)", lambda q, qd, qdd: np.cos(q[:, 0]))

    # ---- 4. Gearbox viscous damping & Coulomb friction candidates ----
    add_bias("qd1", lambda q, qd, qdd: qd[:, 0])
    add_bias("qd2", lambda q, qd, qdd: qd[:, 1])
    add_bias("qd3", lambda q, qd, qdd: qd[:, 2])
    add_bias("tanh(qd1/eps)", lambda q, qd, qdd, e=coulomb_eps: np.tanh(qd[:, 0] / e))
    add_bias("tanh(qd2/eps)", lambda q, qd, qdd, e=coulomb_eps: np.tanh(qd[:, 1] / e))
    add_bias("tanh(qd3/eps)", lambda q, qd, qdd, e=coulomb_eps: np.tanh(qd[:, 2] / e))

    return terms


def evaluate_library(
    terms: list[CandidateTerm],
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
) -> np.ndarray:
    """Evaluate all candidate columns on batched data (N, 3) -> Theta (N, P)."""
    q = np.atleast_2d(q)
    qd = np.atleast_2d(qd)
    qdd = np.atleast_2d(qdd)
    cols = [term.eval_fn(q, qd, qdd) for term in terms]
    return np.column_stack(cols)


def stlsq_normalized(
    Theta: np.ndarray,
    Y: np.ndarray,
    threshold: float = 0.025,
    alpha: float = 1e-3,
    max_iter: int = 25,
) -> np.ndarray:
    """Column-normalized Sequential Thresholded Least Squares (STLSQ).

    Normalizes every column of Theta to unit RMS and each target column Y[:, k]
    to unit RMS before applying threshold `threshold` (fraction of target RMS
    explained per unit normalized feature). This ensures small inertias
    (~1e-4 kg*m^2) and large gravity/damping terms (~0.3 N*m) are thresholded
    fairly on the same dimensionless scale.
    """
    n_samples, n_features = Theta.shape
    n_targets = Y.shape[1]

    col_rms = np.sqrt(np.mean(Theta ** 2, axis=0))
    col_scale = np.where(col_rms > 1e-12, col_rms, 1.0)
    Theta_n = Theta / col_scale

    y_rms = np.sqrt(np.mean(Y ** 2, axis=0))
    y_scale = np.where(y_rms > 1e-12, y_rms, 1.0)
    Y_n = Y / y_scale

    Xi_n = np.zeros((n_features, n_targets), dtype=np.float64)
    ridge = alpha * n_samples * np.eye(n_features)

    for k in range(n_targets):
        yk = Y_n[:, k]
        # Initial ridge least squares
        w = np.linalg.solve(Theta_n.T @ Theta_n + ridge, Theta_n.T @ yk)
        active = np.ones(n_features, dtype=bool)
        for _ in range(max_iter):
            new_active = np.abs(w) >= threshold
            if np.array_equal(new_active, active):
                break
            active = new_active
            w = np.zeros(n_features, dtype=np.float64)
            if not np.any(active):
                break
            A = Theta_n[:, active]
            w[active] = np.linalg.solve(
                A.T @ A + alpha * n_samples * np.eye(A.shape[1]),
                A.T @ yk,
            )
        # Final unpenalized OLS refit on discovered sparse support
        if np.any(active):
            A = Theta_n[:, active]
            w = np.zeros(n_features, dtype=np.float64)
            w[active], *_ = np.linalg.lstsq(A, yk, rcond=None)
        Xi_n[:, k] = w

    # Rescale coefficients back to physical SI units
    Xi = (Xi_n / col_scale[:, None]) * y_scale[None, :]
    return Xi


class SINDyDynamics:
    """Data-driven sparse inverse & forward dynamics discovered by SINDy.

    Exposes the same `.mass_matrix(q)`, `.bias(q, qd)`, `.inverse_dynamics()`,
    and `.forward_dynamics()` interface needed by `LTVMPCController` and
    computed-torque controllers.
    """

    def __init__(
        self,
        terms: list[CandidateTerm],
        Xi: np.ndarray,
        min_inertia: float = 5e-5,
        coulomb_eps: float = 0.05,
    ):
        self.terms = terms
        self.Xi = np.asarray(Xi, dtype=np.float64)
        self.min_inertia = float(min_inertia)
        self.coulomb_eps = float(coulomb_eps)

        # Pre-filter active terms and pack their coefficients into dense tensors so
        # M_hat and b_hat are evaluated for a whole batch of states with one
        # einsum / matmul (MPC linearization queries ~100 states per solve).
        accel_terms: list[CandidateTerm] = []
        bias_terms: list[CandidateTerm] = []
        accel_coefs: list[np.ndarray] = []
        bias_coefs: list[np.ndarray] = []
        for j, term in enumerate(terms):
            coeffs = self.Xi[j, :]
            if np.any(np.abs(coeffs) > 1e-12):
                if term.kind == "accel":
                    accel_terms.append(term)
                    accel_coefs.append(coeffs)
                else:
                    bias_terms.append(term)
                    bias_coefs.append(coeffs)
        self._accel_terms = accel_terms
        self._bias_terms = bias_terms

        # M_hat[i, accel_idx] += Xi[j, i] * phi_j(q)  ->  M = einsum("bk,ijk->bij", Phi, M_coef)
        self._M_coef = np.zeros((3, 3, len(accel_terms)), dtype=np.float64)
        for k, (term, coeffs) in enumerate(zip(accel_terms, accel_coefs)):
            self._M_coef[:, term.accel_idx, k] = coeffs
        # b_hat = Phi_bias @ B_coef
        self._B_coef = (
            np.asarray(bias_coefs, dtype=np.float64)
            if bias_coefs
            else np.zeros((0, 3), dtype=np.float64)
        )
        self._I_floor = self.min_inertia * np.eye(3)

    @property
    def n_nonzero(self) -> int:
        return int(np.sum(np.abs(self.Xi) > 1e-12))

    def format_equations(self) -> str:
        """Render discovered sparse equations in human-readable form."""
        lines = []
        for i in range(3):
            lines.append(f"tau_{i+1} =")
            active_any = False
            for j, term in enumerate(self.terms):
                c = float(self.Xi[j, i])
                if abs(c) > 1e-12:
                    active_any = True
                    sign = "+" if c >= 0 else "-"
                    lines.append(f"    {sign} {abs(c):.6f} * {term.name}")
            if not active_any:
                lines.append("    0")
        return "\n".join(lines)

    # ---- Batched evaluation: q, qd, qdd, tau of shape (B, 3) ----
    def mass_matrix_batch(self, q: np.ndarray) -> np.ndarray:
        """Reconstruct inertia matrices M_hat(q) for a batch of states -> (B, 3, 3)."""
        q = np.atleast_2d(q)
        if self._accel_terms:
            phi = np.column_stack([term.basis_fn(q) for term in self._accel_terms])
            M = np.einsum("bk,ijk->bij", phi, self._M_coef)
        else:
            M = np.zeros((len(q), 3, 3), dtype=np.float64)
        # Enforce symmetry and minimum positive-definite inertia floor
        M = 0.5 * (M + np.swapaxes(M, -1, -2))
        try:
            # Fast path: Cholesky of (M - floor*I) succeeds iff every eigenvalue
            # exceeds the floor, so the eigen-decomposition is skipped.
            np.linalg.cholesky(M - self._I_floor)
            return M
        except np.linalg.LinAlgError:
            pass
        eigvals, eigvecs = np.linalg.eigh(M)
        bad = np.any(eigvals < self.min_inertia, axis=1)
        if np.any(bad):
            w = np.maximum(eigvals[bad], self.min_inertia)
            V = eigvecs[bad]
            M[bad] = (V * w[:, None, :]) @ np.swapaxes(V, -1, -2)
        return M

    def bias_vector_batch(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Reconstruct Coriolis + gravity + friction vectors b_hat(q, qd) -> (B, 3)."""
        q = np.atleast_2d(q)
        qd = np.atleast_2d(qd)
        if not self._bias_terms:
            return np.zeros((len(q), 3), dtype=np.float64)
        qdd_zero = np.zeros_like(q)
        phi = np.column_stack([term.eval_fn(q, qd, qdd_zero) for term in self._bias_terms])
        return phi @ self._B_coef

    def forward_dynamics_batch(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Batched qdd = M_hat(q)^-1 * (tau - b_hat(q, qd)) -> (B, 3)."""
        M = self.mass_matrix_batch(q)
        rhs = np.atleast_2d(tau) - self.bias_vector_batch(q, qd)
        return np.linalg.solve(M, rhs[..., None])[..., 0]

    # ---- Single-state API ----
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Reconstruct 3x3 inertia matrix M_hat(q) from active acceleration terms."""
        return self.mass_matrix_batch(q)[0]

    def bias_vector(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Reconstruct combined Coriolis + gravity + friction vector b_hat(q, qd)."""
        return self.bias_vector_batch(q, qd)[0]

    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """Predict joint torques tau = M_hat(q)*qdd + b_hat(q, qd)."""
        return self.mass_matrix(q) @ qdd + self.bias_vector(q, qd)

    def forward_dynamics(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Predict joint accelerations qdd = M_hat(q)^-1 * (tau - b_hat(q, qd))."""
        return self.forward_dynamics_batch(q, qd, tau)[0]


def fit_sindy_model(
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    tau: np.ndarray,
    threshold: float = 0.025,
    alpha: float = 1e-3,
    coulomb_eps: float = 0.05,
) -> tuple[SINDyDynamics, dict[str, float]]:
    """Fit a sparse Physics-Informed SINDy inverse-dynamics model and return metrics."""
    terms = build_candidate_library(coulomb_eps=coulomb_eps)
    Theta = evaluate_library(terms, q, qd, qdd)
    Xi = stlsq_normalized(Theta, tau, threshold=threshold, alpha=alpha)
    model = SINDyDynamics(terms=terms, Xi=Xi, coulomb_eps=coulomb_eps)

    tau_pred = Theta @ Xi
    resid = tau - tau_pred
    rmse = np.sqrt(np.mean(resid ** 2, axis=0))
    ss_res = np.sum(resid ** 2, axis=0)
    ss_tot = np.sum((tau - np.mean(tau, axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)

    metrics = {
        "n_terms_total": len(terms) * 3,
        "n_terms_active": model.n_nonzero,
        "rmse_tau1": float(rmse[0]),
        "rmse_tau2": float(rmse[1]),
        "rmse_tau3": float(rmse[2]),
        "r2_tau1": float(r2[0]),
        "r2_tau2": float(r2[1]),
        "r2_tau3": float(r2[2]),
        "r2_mean": float(np.mean(r2)),
    }
    return model, metrics


def collect_mujoco_excitation(
    mj_model,
    duration: float = 12.0,
    subsample: int = 5,
    noise_std_q: float = 0.0,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """Collect a persistently exciting rollout from any MuJoCo plant (nominal or perturbed).

    Uses a multi-frequency rich reference tracked by a simple stabilizing PD
    controller so the unknown plant explores diverse (q, qd, qdd) states safely
    inside joint limits without needing prior knowledge of its masses/friction.
    """
    import mujoco
    from scipy.signal import savgol_filter

    data = mujoco.MjData(mj_model)
    dt = float(mj_model.opt.timestep)
    steps = int(round(duration / dt))

    data.qpos[:] = [0.0, 0.45, 0.55]
    data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, data)

    # Rich sum-of-sines excitation with incommensurate frequencies
    ts = np.zeros(steps)
    qs = np.zeros((steps, 3))
    qds = np.zeros((steps, 3))
    qdds = np.zeros((steps, 3))
    taus = np.zeros((steps, 3))

    kp = np.array([2.5, 4.0, 2.0])
    kd = np.array([0.12, 0.25, 0.08])

    for i in range(steps):
        t = float(data.time)
        q_ref = np.array([
            0.75 * np.sin(0.9 * t) + 0.35 * np.sin(2.3 * t + 0.4) + 0.15 * np.cos(3.7 * t),
            0.45 + 0.40 * np.sin(1.1 * t + 0.2) + 0.20 * np.sin(2.7 * t) + 0.10 * np.cos(4.1 * t),
            0.75 + 0.85 * np.sin(1.4 * t + 0.7) + 0.25 * np.sin(3.1 * t) + 0.12 * np.cos(4.9 * t),
        ])
        qd_ref = np.array([
            0.75 * 0.9 * np.cos(0.9 * t) + 0.35 * 2.3 * np.cos(2.3 * t + 0.4) - 0.15 * 3.7 * np.sin(3.7 * t),
            0.40 * 1.1 * np.cos(1.1 * t + 0.2) + 0.20 * 2.7 * np.cos(2.7 * t) - 0.10 * 4.1 * np.sin(4.1 * t),
            0.85 * 1.4 * np.cos(1.4 * t + 0.7) + 0.25 * 3.1 * np.cos(3.1 * t) - 0.12 * 4.9 * np.sin(4.9 * t),
        ])
        q = data.qpos.copy()
        qd = data.qvel.copy()
        tau = np.clip(kp * (q_ref - q) + kd * (qd_ref - qd), -1.5, 1.5)
        data.ctrl[:] = tau

        # Evaluate forward dynamics at (q, qd, tau) to record exact acceleration qacc
        mujoco.mj_forward(mj_model, data)
        ts[i] = t
        qs[i] = q
        qds[i] = qd
        qdds[i] = data.qacc.copy()
        taus[i] = tau

        mujoco.mj_step(mj_model, data)

    # Discard initial 0.2 s startup transient
    start = int(0.2 / dt)
    ts, qs, qds, qdds, taus = ts[start:], qs[start:], qds[start:], qdds[start:], taus[start:]

    if noise_std_q > 0.0:
        # Simulate encoder noise + Savitzky-Golay derivative estimation (realistic hardware pipeline)
        rng = np.random.default_rng(seed)
        q_noisy = qs + rng.normal(0.0, noise_std_q, size=qs.shape)
        win = 31
        qs = savgol_filter(q_noisy, window_length=win, polyorder=3, deriv=0, delta=dt, axis=0)
        qds = savgol_filter(q_noisy, window_length=win, polyorder=3, deriv=1, delta=dt, axis=0)
        qdds = savgol_filter(q_noisy, window_length=win, polyorder=3, deriv=2, delta=dt, axis=0)
        # Trim filter boundary edges
        qs, qds, qdds, taus, ts = qs[win:-win], qds[win:-win], qdds[win:-win], taus[win:-win], ts[win:-win]

    sl = slice(None, None, subsample)
    return {"t": ts[sl], "q": qs[sl], "qd": qds[sl], "qdd": qdds[sl], "tau": taus[sl]}


def load_matlab_sindy_data(folder: Path) -> dict[str, np.ndarray]:
    """Load Simulink logs (`q_serial3.mat`, `qd_serial3.mat`, `qpp_serial3.mat`, `tau_serial3.mat`)."""
    import scipy.io as sio

    q = np.asarray(sio.loadmat(folder / "q_serial3.mat")["q"], dtype=np.float64)
    qd = np.squeeze(np.asarray(sio.loadmat(folder / "qd_serial3.mat")["qd"], dtype=np.float64)).T
    qdd = np.asarray(sio.loadmat(folder / "qpp_serial3.mat")["qpp"], dtype=np.float64)
    tau = np.asarray(sio.loadmat(folder / "tau_serial3.mat")["tau"], dtype=np.float64)

    # Align lengths from the end (matching serial3_sindytau_v3.m) and trim startup transient
    min_len = min(len(q), len(qd), len(qdd), len(tau))
    q = q[-min_len:][3:]
    qd = qd[-min_len:][3:]
    qdd = qdd[-min_len:][3:]
    tau = tau[-min_len:][3:]
    t = np.linspace(0.0, 10.0, len(q))
    return {"t": t, "q": q, "qd": qd, "qdd": qdd, "tau": tau}
