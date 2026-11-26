#!/usr/bin/env python3
"""Symbolic and numerical kinematic & dynamic model of the 3-DOF serial robot.

Clean Python (SymPy + NumPy) counterpart to `cd1_corrected.m`, deriving:
  1. Forward kinematics via Denavit-Hartenberg homogeneous transforms (S01..S34)
  2. Linear and angular geometric Jacobians J_v(q), J_w(q)
  3. Craig's recursive angular & linear velocity propagation (with CoM velocities)
  4. Kinetic energy K(q, qd), potential energy U(q), and Lagrangian L = K - U
  5. Euler-Lagrange inverse dynamics in standard manipulator form:
         M(q) * qdd + C(q, qd) * qd + G(q) = tau
     with Christoffel-symbol Coriolis matrix C(q, qd)
  6. Symbolic proofs of:
       - Exact decomposition: M(q)*qdd + C(q,qd)*qd + G(q) == tau
       - Skew-symmetry of passivity matrix: (M_dot - 2*C) + (M_dot - 2*C)^T == 0
  7. Numerical verification against the MuJoCo CAD model (`model.xml`).

Usage:
  ~/.venvs/mujoco/bin/python kinematics_dynamics_model.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
MODEL_XML = HERE / "model.xml"


# ---------------------------------------------------------------------------
# 1. Homogeneous transformation helpers (equivalent to MATLAB transl/troty/trotz)
# ---------------------------------------------------------------------------
def transl(x, y, z) -> sp.Matrix:
    return sp.Matrix([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ])


def trotz(theta) -> sp.Matrix:
    c, s = sp.cos(theta), sp.sin(theta)
    return sp.Matrix([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])


def troty(theta) -> sp.Matrix:
    c, s = sp.cos(theta), sp.sin(theta)
    return sp.Matrix([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ])


# ---------------------------------------------------------------------------
# 2. Physical parameter container
# ---------------------------------------------------------------------------
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

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.l1, self.l2, self.l3, self.lc1, self.lc2, self.lc3, self.g,
            self.m1, self.m2, self.m3,
            self.Ixx1, self.Iyy1, self.Izz1,
            self.Ixx2, self.Iyy2, self.Izz2,
            self.Ixx3, self.Iyy3, self.Izz3,
        )


# Exact CAD parameters extracted from SolidWorks Ensamble_SWML.xml / MuJoCo model.xml
CAD_PARAMS = RobotParams(
    l1=0.07895, l2=0.23700, l3=0.18000,
    lc1=0.06707, lc2=0.12050, lc3=0.04539,
    g=9.81,
    m1=0.00925, m2=0.09413, m3=0.01179,
    Ixx1=1.65e-6,  Iyy1=3.65e-6,    Izz1=4.33e-6,
    Ixx2=14.56e-6, Iyy2=1010.12e-6, Izz2=1014.22e-6,
    Ixx3=3.21e-6,  Iyy3=39.84e-6,   Izz3=42.02e-6,
)

# Original parameters from cd1_live.mlx (for comparison)
CD1_LIVE_PARAMS = RobotParams(
    l1=0.0728, l2=0.237, l3=0.172,
    lc1=0.0728 / 2, lc2=0.237 / 2, lc3=0.172 / 2,
    g=9.81,
    m1=0.40024, m2=0.2436, m3=0.0345,
    Ixx1=0.006608779, Iyy1=0.002231977, Izz1=0.005585125,
    Ixx2=0.006608779, Iyy2=0.002231977, Izz2=0.005585125,
    Ixx3=0.006608779, Iyy3=0.002231977, Izz3=0.005585125,
)


# ---------------------------------------------------------------------------
# 3. Symbolic Kinematics & Dynamics Derivation
# ---------------------------------------------------------------------------
class Serial3DOFModel:
    """Symbolic & numerical kinematics and Euler-Lagrange dynamics for the 3-DOF robot."""

    def __init__(self, simplify_expressions: bool = True):
        # Joint states
        self.q1, self.q2, self.q3 = sp.symbols("q1 q2 q3", real=True)
        self.q1p, self.q2p, self.q3p = sp.symbols("q1p q2p q3p", real=True)
        self.q1pp, self.q2pp, self.q3pp = sp.symbols("q1pp q2pp q3pp", real=True)
        self.q = sp.Matrix([self.q1, self.q2, self.q3])
        self.qp = sp.Matrix([self.q1p, self.q2p, self.q3p])
        self.qpp = sp.Matrix([self.q1pp, self.q2pp, self.q3pp])

        # Geometric & inertial parameters
        self.l1, self.l2, self.l3 = sp.symbols("l1 l2 l3", positive=True)
        self.lc1, self.lc2, self.lc3 = sp.symbols("lc1 lc2 lc3", real=True)
        self.g = sp.symbols("g", positive=True)
        self.m1, self.m2, self.m3 = sp.symbols("m1 m2 m3", positive=True)
        self.Ixx1, self.Iyy1, self.Izz1 = sp.symbols("Ixx1 Iyy1 Izz1", positive=True)
        self.Ixx2, self.Iyy2, self.Izz2 = sp.symbols("Ixx2 Iyy2 Izz2", positive=True)
        self.Ixx3, self.Iyy3, self.Izz3 = sp.symbols("Ixx3 Iyy3 Izz3", positive=True)

        self.param_syms = (
            self.l1, self.l2, self.l3, self.lc1, self.lc2, self.lc3, self.g,
            self.m1, self.m2, self.m3,
            self.Ixx1, self.Iyy1, self.Izz1,
            self.Ixx2, self.Iyy2, self.Izz2,
            self.Ixx3, self.Iyy3, self.Izz3,
        )

        # ---- Step 1: Forward Kinematics (DH transforms) ----
        # At q = [0, 0, 0], the arm points straight up along +Z:
        #   S01: translate l1 along +Z, yaw q1 about Z0
        #   S12: fixed Ry(pi/2)*Rz(pi) maps X2 -> +Z0 and Z2 -> +X1, pitch q2 about Z2
        #   S23: translate l2 along X2, pitch q3 about Z3
        #   S34: translate l3 along X3 to end-effector
        self.S01 = transl(0, 0, self.l1) * trotz(self.q1)
        self.S12 = troty(sp.pi / 2) * trotz(self.q2 + sp.pi)
        self.S23 = transl(self.l2, 0, 0) * trotz(self.q3)
        self.S34 = transl(self.l3, 0, 0)

        self.S02 = sp.trigsimp(self.S01 * self.S12)
        self.S03 = sp.trigsimp(self.S02 * self.S23)
        self.S04 = sp.trigsimp(self.S03 * self.S34)

        self.p_ee = sp.trigsimp(self.S04[:3, 3])
        self.J_v = sp.trigsimp(self.p_ee.jacobian(self.q))

        # Angular Jacobian columns are joint axes z_1, z_2, z_3 expressed in {0}
        z0_1 = self.S01[:3, 2]
        z0_2 = self.S02[:3, 2]
        z0_3 = self.S03[:3, 2]
        self.J_w = sp.trigsimp(sp.Matrix.hstack(z0_1, z0_2, z0_3))

        # ---- Step 2: Craig's Recursive Velocity Propagation ----
        R01, P01 = self.S01[:3, :3], self.S01[:3, 3]
        R12, P12 = self.S12[:3, :3], self.S12[:3, 3]
        R23, P23 = self.S23[:3, :3], self.S23[:3, 3]
        R34, P34 = self.S34[:3, :3], self.S34[:3, 3]

        w00 = sp.Matrix([0, 0, 0])
        v00 = sp.Matrix([0, 0, 0])
        z = sp.Matrix([0, 0, 1])

        # Correct Craig formula:
        #   w_i = R_{i-1}^i^T * w_{i-1} + qp_i * z
        #   v_i = R_{i-1}^i^T * (v_{i-1} + w_{i-1} x P_{i-1,i})
        self.w11 = sp.trigsimp(R01.T * w00 + self.q1p * z)
        self.v11 = sp.trigsimp(R01.T * (v00 + w00.cross(P01)))

        self.w22 = sp.trigsimp(R12.T * self.w11 + self.q2p * z)
        self.v22 = sp.trigsimp(R12.T * (self.v11 + self.w11.cross(P12)))

        self.w33 = sp.trigsimp(R23.T * self.w22 + self.q3p * z)
        self.v33 = sp.trigsimp(R23.T * (self.v22 + self.w22.cross(P23)))

        self.w44 = sp.trigsimp(R34.T * self.w33)
        self.v44 = sp.trigsimp(R34.T * (self.v33 + self.w33.cross(P34)))

        # Local CoM vectors in each link frame {i}
        r_c1 = sp.Matrix([0, 0, self.lc1 - self.l1])  # height lc1 from world origin {0}
        r_c2 = sp.Matrix([self.lc2, 0, 0])            # distance lc2 along link 2 (+X2)
        r_c3 = sp.Matrix([self.lc3, 0, 0])            # distance lc3 along link 3 (+X3)

        # CoM linear velocities in each link frame {i}
        self.vc1 = sp.trigsimp(self.v11 + self.w11.cross(r_c1))
        self.vc2 = sp.trigsimp(self.v22 + self.w22.cross(r_c2))
        self.vc3 = sp.trigsimp(self.v33 + self.w33.cross(r_c3))

        # ---- Step 3: Kinetic & Potential Energy, Lagrangian ----
        I1 = sp.diag(self.Ixx1, self.Iyy1, self.Izz1)
        I2 = sp.diag(self.Ixx2, self.Iyy2, self.Izz2)
        I3 = sp.diag(self.Ixx3, self.Iyy3, self.Izz3)

        half = sp.Rational(1, 2)
        self.K1 = half * self.m1 * self.vc1.dot(self.vc1) + half * (self.w11.T * I1 * self.w11)[0]
        self.K2 = half * self.m2 * self.vc2.dot(self.vc2) + half * (self.w22.T * I2 * self.w22)[0]
        self.K3 = half * self.m3 * self.vc3.dot(self.vc3) + half * (self.w33.T * I3 * self.w33)[0]
        self.K = sp.trigsimp(self.K1 + self.K2 + self.K3)

        # Inertial heights H_i of each link's CoM along +Z
        self.pc1_0 = sp.trigsimp((self.S01 * sp.Matrix([r_c1[0], r_c1[1], r_c1[2], 1]))[:3, 0])
        self.pc2_0 = sp.trigsimp((self.S02 * sp.Matrix([r_c2[0], r_c2[1], r_c2[2], 1]))[:3, 0])
        self.pc3_0 = sp.trigsimp((self.S03 * sp.Matrix([r_c3[0], r_c3[1], r_c3[2], 1]))[:3, 0])

        self.U = sp.trigsimp(self.g * (self.m1 * self.pc1_0[2] + self.m2 * self.pc2_0[2] + self.m3 * self.pc3_0[2]))
        self.L = self.K - self.U

        # ---- Step 4: Euler-Lagrange Equations & Manipulator Matrices ----
        # M(q) = Hessian of K w.r.t. qp  (identical to d(tau)/d(qpp))
        self.M = sp.Matrix(3, 3, lambda i, j: sp.diff(sp.diff(self.K, self.qp[i]), self.qp[j]))
        # G(q) = dU / dq
        self.G = sp.Matrix([sp.diff(self.U, qi) for qi in self.q])
        # C(q, qp) via Christoffel symbols of the first kind
        self.C_mat = sp.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                cij = sp.Integer(0)
                for k in range(3):
                    c_ijk = half * (
                        sp.diff(self.M[i, j], self.q[k])
                        + sp.diff(self.M[i, k], self.q[j])
                        - sp.diff(self.M[j, k], self.q[i])
                    )
                    cij += c_ijk * self.qp[k]
                self.C_mat[i, j] = cij
        self.C_vec = self.C_mat * self.qp
        self.tau = self.M * self.qpp + self.C_vec + self.G

        if simplify_expressions:
            self.M = sp.trigsimp(self.M)
            self.C_mat = sp.trigsimp(self.C_mat)
            self.C_vec = sp.trigsimp(self.C_vec)
            self.G = sp.trigsimp(self.G)

        # ---- Step 5: Verify fundamental structural properties symbolically ----
        self._verify_symbolic_identities()

        # ---- Step 6: Build fast NumPy evaluators ----
        self._fk_fn = sp.lambdify((self.q1, self.q2, self.q3, *self.param_syms), self.p_ee, "numpy")
        self._jv_fn = sp.lambdify((self.q1, self.q2, self.q3, *self.param_syms), self.J_v, "numpy")
        self._M_fn = sp.lambdify((self.q1, self.q2, self.q3, *self.param_syms), self.M, "numpy")
        self._C_fn = sp.lambdify(
            (self.q1, self.q2, self.q3, self.q1p, self.q2p, self.q3p, *self.param_syms),
            self.C_mat,
            "numpy",
        )
        self._G_fn = sp.lambdify((self.q1, self.q2, self.q3, *self.param_syms), self.G, "numpy")

    def _verify_symbolic_identities(self) -> None:
        # 1. M(q) is symmetric
        assert sp.simplify(self.M - self.M.T) == sp.zeros(3, 3), "M(q) is not symmetric!"

        # 2. Passivity / skew-symmetry: (M_dot - 2*C) + (M_dot - 2*C)^T == 0
        M_dot = sp.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                M_dot[i, j] = sum(sp.diff(self.M[i, j], self.q[k]) * self.qp[k] for k in range(3))
        N = M_dot - 2 * self.C_mat
        assert sp.trigsimp(N + N.T) == sp.zeros(3, 3), "M_dot - 2*C is not skew-symmetric!"

        # 3. Direct Euler-Lagrange derivative matches M*qpp + C*qp + G
        tau_direct = sp.zeros(3, 1)
        for i in range(3):
            dL_dqpi = sp.diff(self.L, self.qp[i])
            ddt = sum(
                sp.diff(dL_dqpi, self.q[k]) * self.qp[k] + sp.diff(dL_dqpi, self.qp[k]) * self.qpp[k]
                for k in range(3)
            )
            tau_direct[i] = ddt - sp.diff(self.L, self.q[i])
        assert sp.trigsimp(tau_direct - self.tau) == sp.zeros(3, 1), "Euler-Lagrange mismatch!"

    # ---- Numerical API ----
    def fk(self, q: np.ndarray, params: RobotParams = CAD_PARAMS, mujoco_frame: bool = True) -> np.ndarray:
        """End-effector position [x, y, z] in meters.

        If `mujoco_frame=True`, applies the +90 deg Z rotation aligning DH world
        with the MuJoCo CAD world frame: (x_mj, y_mj, z_mj) = (-y_dh, x_dh, z_dh).
        """
        p_dh = np.asarray(self._fk_fn(q[0], q[1], q[2], *params.as_tuple()), dtype=np.float64).ravel()
        if mujoco_frame:
            return np.array([-p_dh[1], p_dh[0], p_dh[2]])
        return p_dh

    def jacobian_v(self, q: np.ndarray, params: RobotParams = CAD_PARAMS, mujoco_frame: bool = True) -> np.ndarray:
        """3x3 linear Jacobian dp_ee/dq [m/rad]."""
        J_dh = np.asarray(self._jv_fn(q[0], q[1], q[2], *params.as_tuple()), dtype=np.float64)
        if mujoco_frame:
            R_z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            return R_z90 @ J_dh
        return J_dh

    def mass_matrix(self, q: np.ndarray, params: RobotParams = CAD_PARAMS) -> np.ndarray:
        """3x3 inertia matrix M(q) [kg*m^2]."""
        return np.asarray(self._M_fn(q[0], q[1], q[2], *params.as_tuple()), dtype=np.float64)

    def coriolis_matrix(self, q: np.ndarray, qd: np.ndarray, params: RobotParams = CAD_PARAMS) -> np.ndarray:
        """3x3 Coriolis/centrifugal matrix C(q, qd) [N*m*s/rad]."""
        return np.asarray(
            self._C_fn(q[0], q[1], q[2], qd[0], qd[1], qd[2], *params.as_tuple()),
            dtype=np.float64,
        )

    def gravity_vector(self, q: np.ndarray, params: RobotParams = CAD_PARAMS) -> np.ndarray:
        """3x1 gravity torque vector G(q) [N*m]."""
        return np.asarray(self._G_fn(q[0], q[1], q[2], *params.as_tuple()), dtype=np.float64).ravel()

    def inverse_dynamics(
        self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, params: RobotParams = CAD_PARAMS
    ) -> np.ndarray:
        """Compute required joint torques tau = M(q)*qdd + C(q,qd)*qd + G(q) [N*m]."""
        M = self.mass_matrix(q, params)
        C = self.coriolis_matrix(q, qd, params)
        G = self.gravity_vector(q, params)
        return M @ qdd + C @ qd + G

    def forward_dynamics(
        self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, params: RobotParams = CAD_PARAMS
    ) -> np.ndarray:
        """Compute joint accelerations qdd = M(q)^-1 * (tau - C(q,qd)*qd - G(q)) [rad/s^2]."""
        M = self.mass_matrix(q, params)
        C = self.coriolis_matrix(q, qd, params)
        G = self.gravity_vector(q, params)
        return np.linalg.solve(M, tau - C @ qd - G)


def verify_against_mujoco(sym_model: Serial3DOFModel) -> None:
    """Cross-check FK, Jacobian, M(q), bias C*qd+G, and inverse dynamics against MuJoCo."""
    import mujoco

    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    mj_model.opt.gravity[:] = [0.0, 0.0, -9.81]
    mj_model.jnt_limited[:] = 0
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mj_data = mujoco.MjData(mj_model)
    ee_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

    rng = np.random.default_rng(7)
    fk_errs, jac_errs, M_errs, bias_errs, tau_errs = [], [], [], [], []

    for _ in range(100):
        q = rng.uniform([-2.0, -1.2, -1.5], [2.0, 1.2, 1.5])
        qd = rng.uniform(-2.0, 2.0, size=3)
        qdd = rng.uniform(-3.0, 3.0, size=3)

        mj_data.qpos[:] = q
        mj_data.qvel[:] = qd
        mj_data.qacc[:] = qdd
        mujoco.mj_inverse(mj_model, mj_data)

        M_mj = np.zeros((3, 3))
        mujoco.mj_fullM(mj_model, mj_data, M_mj)
        jacp_mj = np.zeros((3, 3))
        jacr_mj = np.zeros((3, 3))
        mujoco.mj_jacSite(mj_model, mj_data, jacp_mj, jacr_mj, ee_id)

        fk_an = sym_model.fk(q, CAD_PARAMS, mujoco_frame=True)
        jac_an = sym_model.jacobian_v(q, CAD_PARAMS, mujoco_frame=True)
        M_an = sym_model.mass_matrix(q, CAD_PARAMS)
        bias_an = sym_model.coriolis_matrix(q, qd, CAD_PARAMS) @ qd + sym_model.gravity_vector(q, CAD_PARAMS)
        tau_an = sym_model.inverse_dynamics(q, qd, qdd, CAD_PARAMS)

        fk_errs.append(np.linalg.norm(fk_an - mj_data.site_xpos[ee_id]))
        jac_errs.append(np.max(np.abs(jac_an - jacp_mj)))
        M_errs.append(np.max(np.abs(M_an - M_mj)))
        bias_errs.append(np.max(np.abs(bias_an - mj_data.qfrc_bias)))
        tau_errs.append(np.max(np.abs(tau_an - mj_data.qfrc_inverse)))

    print("\n=== NUMERICAL VERIFICATION VS MUJOCO (100 random states) ===")
    print(f"  End-effector FK max error:   {max(fk_errs)*1000:.6f} mm")
    print(f"  Linear Jacobian max error:   {max(jac_errs):.6f} m/rad")
    print(f"  Mass matrix M(q) max error:  {max(M_errs):.6e} kg*m^2")
    print(f"  Bias (C*qd + G) max error:   {max(bias_errs):.6e} N*m")
    print(f"  Inverse dynamics max error:  {max(tau_errs):.6e} N*m")


def main():
    print("Deriving symbolic kinematics and Euler-Lagrange dynamics...")
    model = Serial3DOFModel(simplify_expressions=True)
    print("[PASS] Symbolic identities verified: M == M^T, (M_dot - 2C) skew-symmetric, Euler-Lagrange exact.")

    print("\n=== End-Effector Position p_ee(q) in DH frame ===")
    sp.pprint(model.p_ee)

    print("\n=== Gravity Vector G(q) ===")
    sp.pprint(model.G)

    print("\n=== Inertia Matrix M(q) ===")
    for i in range(3):
        for j in range(i, 3):
            print(f"M[{i+1},{j+1}] = {model.M[i, j]}")

    verify_against_mujoco(model)


if __name__ == "__main__":
    main()
