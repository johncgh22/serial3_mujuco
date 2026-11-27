"""Fast pure-NumPy closed-form kinematics, Euler-Lagrange dynamics, and regressor.

Implements the exact verified equations derived symbolically in
`kinematics_dynamics_model.py` (without SymPy startup overhead) plus:
  - Analytical time-derivative of the linear Jacobian J_v_dot(q, qd) for OSC
  - Linear-in-parameters dynamic regressor Y(q, qd, qd_r, qdd_r) for Slotine-Li
    adaptive control (10 minimal base inertial parameters + optional 6 friction
    parameters for viscous & Coulomb gearbox friction).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RobotParams:
    """Inertial and geometric parameters of the 3-DOF serial manipulator."""
    l1: float
    l2: float
    l3: float
    lc1: float
    lc2: float
    lc3: float
    g: float
    m1: float
    m2: float
    m3: float
    Ixx1: float
    Iyy1: float
    Izz1: float
    Ixx2: float
    Iyy2: float
    Izz2: float
    Ixx3: float
    Iyy3: float
    Izz3: float


# Exact CAD parameters from SolidWorks Ensamble_SWML.xml / MuJoCo model.xml
# (gravity matches scene.xml: 9.80665 m/s^2)
CAD_PARAMS = RobotParams(
    l1=0.07895, l2=0.23700, l3=0.18000,
    lc1=0.06707, lc2=0.12050, lc3=0.04539,
    g=9.80665,
    m1=0.009253152908, m2=0.09412545006, m3=0.01179484791,
    Ixx1=1.649790333e-6, Iyy1=3.649503234e-6,   Izz1=4.326176249e-6,
    Ixx2=1.456124529e-5, Iyy2=1.010118916e-3,   Izz2=1.014214974e-3,
    Ixx3=3.206080635e-6, Iyy3=3.984072282e-5,   Izz3=4.201522222e-5,
)


def params_to_base_theta(p: RobotParams) -> np.ndarray:
    """Pack physical parameters into the 10 minimal linear base inertial parameters.

    theta = [
      0: Izz1,
      1: Ixx2,
      2: Iyy2 + m2*lc2^2 + m3*l2^2,
      3: Ixx3,
      4: Iyy3 + m3*lc3^2,
      5: m3*l2*lc3,
      6: Izz2 + m2*lc2^2 + m3*l2^2,
      7: Izz3 + m3*lc3^2,
      8: m2*lc2 + m3*l2,
      9: m3*lc3,
    ]
    """
    return np.array([
        p.Izz1,
        p.Ixx2,
        p.Iyy2 + p.m2 * p.lc2**2 + p.m3 * p.l2**2,
        p.Ixx3,
        p.Iyy3 + p.m3 * p.lc3**2,
        p.m3 * p.l2 * p.lc3,
        p.Izz2 + p.m2 * p.lc2**2 + p.m3 * p.l2**2,
        p.Izz3 + p.m3 * p.lc3**2,
        p.m2 * p.lc2 + p.m3 * p.l2,
        p.m3 * p.lc3,
    ], dtype=np.float64)


class AnalyticalDynamics:
    """Closed-form NumPy kinematics, dynamics, and regressor for the 3-DOF robot."""

    def __init__(self, params: RobotParams = CAD_PARAMS):
        self.params = params
        self.theta_nominal = params_to_base_theta(params)

    # ---- Kinematics (MuJoCo world frame: upright zero along +Z) ----
    def fk(self, q: np.ndarray) -> np.ndarray:
        """End-effector position [x, y, z] in MuJoCo world frame [m]."""
        p = self.params
        q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
        q23 = q2 + q3
        r = p.l2 * np.sin(q2) + p.l3 * np.sin(q23)
        return np.array([
            r * np.cos(q1),
            r * np.sin(q1),
            p.l1 + p.l2 * np.cos(q2) + p.l3 * np.cos(q23),
        ], dtype=np.float64)

    def jacobian_v(self, q: np.ndarray) -> np.ndarray:
        """3x3 linear Jacobian dp_ee/dq in MuJoCo world frame [m/rad]."""
        p = self.params
        q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
        q23 = q2 + q3
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c23, s23 = np.cos(q23), np.sin(q23)
        r = p.l2 * s2 + p.l3 * s23
        r2 = p.l2 * c2 + p.l3 * c23
        r3 = p.l3 * c23
        return np.array([
            [-r * s1,  r2 * c1,  r3 * c1],
            [ r * c1,  r2 * s1,  r3 * s1],
            [    0.0,       -r, -p.l3 * s23],
        ], dtype=np.float64)

    def jacobian_v_dot(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Time derivative d/dt J_v(q) in MuJoCo world frame [m/(rad*s)]."""
        p = self.params
        q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
        qd1, qd2, qd3 = float(qd[0]), float(qd[1]), float(qd[2])
        q23 = q2 + q3
        qd23 = qd2 + qd3
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c23, s23 = np.cos(q23), np.sin(q23)

        r = p.l2 * s2 + p.l3 * s23
        r2 = p.l2 * c2 + p.l3 * c23
        r3 = p.l3 * c23

        r_dot = r2 * qd2 + r3 * qd3
        r2_dot = -p.l2 * s2 * qd2 - p.l3 * s23 * qd23
        r3_dot = -p.l3 * s23 * qd23

        return np.array([
            [-r_dot * s1 - r * c1 * qd1, r2_dot * c1 - r2 * s1 * qd1, r3_dot * c1 - r3 * s1 * qd1],
            [ r_dot * c1 - r * s1 * qd1, r2_dot * s1 + r2 * c1 * qd1, r3_dot * s1 + r3 * c1 * qd1],
            [                       0.0,                      -r_dot,         -p.l3 * c23 * qd23],
        ], dtype=np.float64)

    # ---- Dynamics ----
    def mass_matrix(self, q: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
        """3x3 symmetric positive-definite inertia matrix M(q) [kg*m^2]."""
        th = self.theta_nominal if theta is None else theta
        q2, q3 = float(q[1]), float(q[2])
        q23 = q2 + q3
        c2, s2 = np.cos(q2), np.sin(q2)
        c23, s23 = np.cos(q23), np.sin(q23)
        c3 = np.cos(q3)

        m11 = (
            th[0]
            + th[1] * c2 * c2
            + th[2] * s2 * s2
            + th[3] * c23 * c23
            + th[4] * s23 * s23
            + 2.0 * th[5] * s2 * s23
        )
        m22 = th[6] + th[7] + 2.0 * th[5] * c3
        m23 = th[7] + th[5] * c3
        m33 = th[7]
        return np.array([
            [m11, 0.0, 0.0],
            [0.0, m22, m23],
            [0.0, m23, m33],
        ], dtype=np.float64)

    def coriolis_matrix(
        self, q: np.ndarray, qd: np.ndarray, theta: np.ndarray | None = None
    ) -> np.ndarray:
        """3x3 Christoffel Coriolis/centrifugal matrix C(q, qd) [N*m*s/rad].

        Satisfies exact passivity / skew-symmetry: (M_dot - 2*C) + (M_dot - 2*C)^T = 0.
        """
        th = self.theta_nominal if theta is None else theta
        q2, q3 = float(q[1]), float(q[2])
        qd1, qd2, qd3 = float(qd[0]), float(qd[1]), float(qd[2])
        q23 = q2 + q3
        c2, s2 = np.cos(q2), np.sin(q2)
        c23, s23 = np.cos(q23), np.sin(q23)
        s3 = np.sin(q3)

        # a2 = 0.5 * dM11/dq2, a3 = 0.5 * dM11/dq3, h = 0.5 * dM22/dq3 = dM23/dq3
        a2 = (
            (th[2] - th[1]) * s2 * c2
            + (th[4] - th[3]) * s23 * c23
            + th[5] * (c2 * s23 + s2 * c23)
        )
        a3 = (th[4] - th[3]) * s23 * c23 + th[5] * s2 * c23
        h = -th[5] * s3

        return np.array([
            [a2 * qd2 + a3 * qd3,       a2 * qd1,             a3 * qd1],
            [          -a2 * qd1,        h * qd3,    h * (qd2 + qd3)],
            [          -a3 * qd1,       -h * qd2,                  0.0],
        ], dtype=np.float64)

    def gravity_vector(self, q: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
        """3x1 gravity torque vector G(q) [N*m]."""
        th = self.theta_nominal if theta is None else theta
        g = self.params.g
        q2, q3 = float(q[1]), float(q[2])
        s2 = np.sin(q2)
        s23 = np.sin(q2 + q3)
        return np.array([
            0.0,
            -g * (th[8] * s2 + th[9] * s23),
            -g * (th[9] * s23),
        ], dtype=np.float64)

    def inverse_dynamics(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        theta: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute tau = M(q)*qdd + C(q,qd)*qd + G(q) [N*m]."""
        return (
            self.mass_matrix(q, theta) @ qdd
            + self.coriolis_matrix(q, qd, theta) @ qd
            + self.gravity_vector(q, theta)
        )

    def forward_dynamics(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        tau: np.ndarray,
        theta: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute joint accelerations qdd = M(q)^-1 * (tau - C(q,qd)*qd - G(q)) [rad/s^2]."""
        M = self.mass_matrix(q, theta)
        b = self.coriolis_matrix(q, qd, theta) @ qd + self.gravity_vector(q, theta)
        return np.linalg.solve(M, tau - b)

    def regressor(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qd_r: np.ndarray,
        qdd_r: np.ndarray,
        include_friction: bool = False,
        coulomb_eps: float = 0.05,
    ) -> np.ndarray:
        """Slotine-Li linear-in-parameters regressor Y(q, qd, qd_r, qdd_r).

        Satisfies identically:
            Y[:, :10] @ theta_inertial == M(q) @ qdd_r + C(q, qd) @ qd_r + G(q)
        If `include_friction=True`, appends 6 columns for per-joint viscous damping
        and smooth Coulomb friction: diag(qd) and diag(tanh(qd / coulomb_eps)).
        """
        q2, q3 = float(q[1]), float(q[2])
        qd1, qd2, qd3 = float(qd[0]), float(qd[1]), float(qd[2])
        qr1, qr2, qr3 = float(qd_r[0]), float(qd_r[1]), float(qd_r[2])
        qrr1, qrr2, qrr3 = float(qdd_r[0]), float(qdd_r[1]), float(qdd_r[2])
        g = self.params.g

        q23 = q2 + q3
        c2, s2 = np.cos(q2), np.sin(q2)
        c23, s23 = np.cos(q23), np.sin(q23)
        c3, s3 = np.cos(q3), np.sin(q3)

        ncols = 16 if include_friction else 10
        Y = np.zeros((3, ncols), dtype=np.float64)

        # Column 0: theta_0 = Izz1
        Y[0, 0] = qrr1

        # Column 1: theta_1 = Ixx2 (a2 = -s2*c2)
        a2_1 = -s2 * c2
        Y[0, 1] = c2 * c2 * qrr1 + a2_1 * (qd2 * qr1 + qd1 * qr2)
        Y[1, 1] = -a2_1 * qd1 * qr1

        # Column 2: theta_2 = Iyy2 + m2*lc2^2 + m3*l2^2 (a2 = +s2*c2)
        a2_2 = s2 * c2
        Y[0, 2] = s2 * s2 * qrr1 + a2_2 * (qd2 * qr1 + qd1 * qr2)
        Y[1, 2] = -a2_2 * qd1 * qr1

        # Column 3: theta_3 = Ixx3 (a2 = a3 = -s23*c23)
        a_3 = -s23 * c23
        Y[0, 3] = c23 * c23 * qrr1 + a_3 * ((qd2 + qd3) * qr1 + qd1 * (qr2 + qr3))
        Y[1, 3] = -a_3 * qd1 * qr1
        Y[2, 3] = -a_3 * qd1 * qr1

        # Column 4: theta_4 = Iyy3 + m3*lc3^2 (a2 = a3 = +s23*c23)
        a_4 = s23 * c23
        Y[0, 4] = s23 * s23 * qrr1 + a_4 * ((qd2 + qd3) * qr1 + qd1 * (qr2 + qr3))
        Y[1, 4] = -a_4 * qd1 * qr1
        Y[2, 4] = -a_4 * qd1 * qr1

        # Column 5: theta_5 = m3*l2*lc3
        a2_5 = c2 * s23 + s2 * c23
        a3_5 = s2 * c23
        h_5 = -s3
        Y[0, 5] = (
            2.0 * s2 * s23 * qrr1
            + (a2_5 * qd2 + a3_5 * qd3) * qr1
            + a2_5 * qd1 * qr2
            + a3_5 * qd1 * qr3
        )
        Y[1, 5] = (
            2.0 * c3 * qrr2
            + c3 * qrr3
            - a2_5 * qd1 * qr1
            + h_5 * qd3 * qr2
            + h_5 * (qd2 + qd3) * qr3
        )
        Y[2, 5] = c3 * qrr2 - a3_5 * qd1 * qr1 - h_5 * qd2 * qr2

        # Column 6: theta_6 = Izz2 + m2*lc2^2 + m3*l2^2
        Y[1, 6] = qrr2

        # Column 7: theta_7 = Izz3 + m3*lc3^2
        Y[1, 7] = qrr2 + qrr3
        Y[2, 7] = qrr2 + qrr3

        # Column 8: theta_8 = m2*lc2 + m3*l2 (gravity)
        Y[1, 8] = -g * s2

        # Column 9: theta_9 = m3*lc3 (gravity)
        Y[1, 9] = -g * s23
        Y[2, 9] = -g * s23

        if include_friction:
            # Viscous damping (cols 10..12) and smooth Coulomb friction (cols 13..15)
            Y[0, 10] = qd1
            Y[1, 11] = qd2
            Y[2, 12] = qd3
            Y[0, 13] = np.tanh(qd1 / coulomb_eps)
            Y[1, 14] = np.tanh(qd2 / coulomb_eps)
            Y[2, 15] = np.tanh(qd3 / coulomb_eps)

        return Y
