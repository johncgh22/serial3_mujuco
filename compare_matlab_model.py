#!/usr/bin/env python3
"""Compare the analytical MATLAB model (cd1.m / cd1_live.mlx) against MuJoCo.

Evaluates along a smooth 3-DOF trajectory:
  1. Forward kinematics: End-effector (x, y, z) from cd1_live.mlx DH equations vs MuJoCo.
  2. Inverse dynamics: Torques tau(t) = M(q)*qdd + C(q,qd) + G(q) from:
       (a) Literal cd1_live.mlx symbolic equations (with cd1_live.mlx parameters),
       (b) Corrected Euler-Lagrange analytical model with CAD/MuJoCo parameters,
       (c) MuJoCo `mj_inverse` (scene.xml / model.xml).
  3. Exports `mujoco_trajectory.mat` so you can also load and compare directly in MATLAB.

Usage:
  ~/.venvs/mujoco/bin/python compare_matlab_model.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import scipy.io as sio
import sympy as sp

HERE = Path(__file__).resolve().parent
MODEL_XML = HERE / "model.xml"
RENDER_DIR = HERE / "renders"
RENDER_DIR.mkdir(exist_ok=True)


def transl_sp(x, y, z):
    return sp.Matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def trotz_sp(a):
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def troty_sp(a):
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def build_symbolic_models():
    """Build lambdified FK and inverse dynamics functions for:
      - `literal`: exact equations from cd1_live.mlx
      - `corrected`: fixed Craig velocity propagation + link CoM vectors + Lagrangian U
    """
    q1, q2, q3 = sp.symbols("q1 q2 q3", real=True)
    qp1, qp2, qp3 = sp.symbols("qp1 qp2 qp3", real=True)
    qpp1, qpp2, qpp3 = sp.symbols("qpp1 qpp2 qpp3", real=True)
    l1, l2, l3, lc1, lc2, lc3, g = sp.symbols("l1 l2 l3 lc1 lc2 lc3 g", real=True)
    m1, m2, m3 = sp.symbols("m1 m2 m3", real=True)
    Ixx1, Iyy1, Izz1 = sp.symbols("Ixx1 Iyy1 Izz1", real=True)
    Ixx2, Iyy2, Izz2 = sp.symbols("Ixx2 Iyy2 Izz2", real=True)
    Ixx3, Iyy3, Izz3 = sp.symbols("Ixx3 Iyy3 Izz3", real=True)

    S01 = transl_sp(0, 0, l1) * trotz_sp(q1)
    S12 = troty_sp(sp.pi / 2) * trotz_sp(q2 + sp.pi)
    S23 = transl_sp(l2, 0, 0) * trotz_sp(q3)
    S34 = transl_sp(l3, 0, 0)

    S02 = S01 * S12
    S03 = S02 * S23
    S04 = S03 * S34

    R01, P01 = S01[:3, :3], S01[:3, 3]
    R12, P12 = S12[:3, :3], S12[:3, 3]
    R23, P23 = S23[:3, :3], S23[:3, 3]
    R34, P34 = S34[:3, :3], S34[:3, 3]

    z = sp.Matrix([0, 0, 1])
    I1 = sp.diag(Ixx1, Iyy1, Izz1)
    I2 = sp.diag(Ixx2, Iyy2, Izz2)
    I3 = sp.diag(Ixx3, Iyy3, Izz3)
    q_vec = [q1, q2, q3]
    qp_vec = [qp1, qp2, qp3]
    qpp_vec = [qpp1, qpp2, qpp3]

    def euler_lagrange(L):
        taus = []
        for qi, qpi in zip(q_vec, qp_vec):
            dL_dqpi = sp.diff(L, qpi)
            ddt = sum(sp.diff(dL_dqpi, qj) * qpj + sp.diff(dL_dqpi, qpj) * qppj
                      for qj, qpj, qppj in zip(q_vec, qp_vec, qpp_vec))
            taus.append(ddt - sp.diff(L, qi))
        return sp.Matrix(taus)

    # --- (A) Literal cd1_live.mlx equations ---
    w00 = sp.Matrix([0, 0, 0])
    v00 = sp.Matrix([0, 0, 0])
    w11 = R01.T * w00 + qp1 * z
    v11 = R01.T * v00 + w00.cross(P01)
    w22 = R12.T * w11 + qp2 * z
    v22 = R12.T * v11 + w11.cross(P12)
    w33 = R23.T * w22 + qp3 * z
    v33 = R23.T * v22 + w22.cross(P23)
    w44 = R34.T * w33
    P01c = P01.subs(l1, lc1)
    P12c = P12.subs(l2, lc2)  # Note: P12 is [0,0,0], so P12c == 0 in cd1_live.mlx!
    P23c = P23.subs(l3, lc3)  # Note: P23 is [l2,0,0], so P23c == [l2,0,0] in cd1_live.mlx!
    K1_lit = sp.Rational(1, 2) * m1 * (v11 + w11.cross(P01c)).dot(v11 + w11.cross(P01c)) + sp.Rational(1, 2) * (w11.T * I1 * w11)[0]
    K2_lit = sp.Rational(1, 2) * m2 * (v22 + w22.cross(P12c)).dot(v22 + w22.cross(P12c)) + sp.Rational(1, 2) * (w22.T * I2 * w22)[0]
    K3_lit = sp.Rational(1, 2) * m3 * (v33 + w44.cross(P23c)).dot(v33 + w44.cross(P23c)) + sp.Rational(1, 2) * (w44.T * I3 * w44)[0]
    H1_lit = S02[2, 3].subs(l1, lc1)
    H2_lit = S03[2, 3].subs(l2, lc2)
    H3_lit = S04[2, 3].subs(l3, lc3)
    L_lit = (K1_lit + K2_lit + K3_lit) - (-m1 * g * H1_lit - (-m2 * g * H2_lit) - (-m3 * g * H3_lit))
    tau_lit = euler_lagrange(L_lit)

    # --- (B) Corrected analytical Euler-Lagrange equations (same DH kinematics) ---
    # CoM positions in world frame directly from homogeneous transforms:
    p_c1 = (S01 * sp.Matrix([0, 0, lc1 - l1, 1]))[:3, 0]
    p_c2 = (S02 * sp.Matrix([lc2, 0, 0, 1]))[:3, 0]
    p_c3 = (S03 * sp.Matrix([lc3, 0, 0, 1]))[:3, 0]

    def vel_world(p):
        return sum((sp.diff(p, qi) * qpi for qi, qpi in zip(q_vec, qp_vec)), sp.zeros(3, 1))

    vc1, vc2, vc3 = vel_world(p_c1), vel_world(p_c2), vel_world(p_c3)
    K1_cor = sp.Rational(1, 2) * m1 * vc1.dot(vc1) + sp.Rational(1, 2) * (w11.T * I1 * w11)[0]
    K2_cor = sp.Rational(1, 2) * m2 * vc2.dot(vc2) + sp.Rational(1, 2) * (w22.T * I2 * w22)[0]
    K3_cor = sp.Rational(1, 2) * m3 * vc3.dot(vc3) + sp.Rational(1, 2) * (w33.T * I3 * w33)[0]
    U_cor = g * (m1 * p_c1[2] + m2 * p_c2[2] + m3 * p_c3[2])
    L_cor = (K1_cor + K2_cor + K3_cor) - U_cor
    tau_cor = euler_lagrange(L_cor)

    all_syms = (
        q1, q2, q3, qp1, qp2, qp3, qpp1, qpp2, qpp3,
        l1, l2, l3, lc1, lc2, lc3, g,
        m1, m2, m3, Ixx1, Iyy1, Izz1, Ixx2, Iyy2, Izz2, Ixx3, Iyy3, Izz3,
    )
    fk_fn = sp.lambdify((q1, q2, q3, l1, l2, l3), S04[:3, 3], "numpy")
    tau_lit_fn = sp.lambdify(all_syms, tau_lit, "numpy")
    tau_cor_fn = sp.lambdify(all_syms, tau_cor, "numpy")
    return fk_fn, tau_lit_fn, tau_cor_fn


def main():
    print("Building symbolic models from cd1_live.mlx equations...")
    fk_fn, tau_lit_fn, tau_cor_fn = build_symbolic_models()

    # Load pure torque-controlled MuJoCo model (no floor/limits/damping for pure dynamics comparison)
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    model.opt.gravity[:] = [0.0, 0.0, -9.81]  # Match g = 9.81 in cd1_live.mlx
    model.jnt_limited[:] = 0
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(model)

    # Smooth 3-DOF test trajectory around upright q = [0, 0, 0]
    T = 4.0
    dt = 0.01
    ts = np.arange(0.0, T + dt / 2, dt)
    amp = np.array([0.7, 0.5, 0.6])
    w = np.array([1.2, 1.5, 2.0])
    qs = amp * np.sin(w * ts[:, None])
    qds = amp * w * np.cos(w * ts[:, None])
    qdds = -amp * (w**2) * np.sin(w * ts[:, None])

    # Frame alignment between cd1_live.mlx DH convention and MuJoCo CAD convention:
    # At q = [0, 0, 0], both stand straight up along +Z!
    # In cd1_live.mlx: joint2 & joint3 rotate around +X (so +q2 pitches toward -Y).
    # In MuJoCo CAD:   joint2 & joint3 rotate around +Y (so +q2 pitches toward +X).
    # Rotating cd1_live.mlx world by +90 deg around +Z maps (x_dh, y_dh, z_dh) -> (-y_dh, x_dh, z_dh),
    # aligning the pitch plane identically! (Equivalently, q1_dh = q1_mj - pi/2.)
    def dh_to_mj_world(p_dh: np.ndarray) -> np.ndarray:
        return np.array([-p_dh[1], p_dh[0], p_dh[2]])

    # Parameters from cd1_live.mlx
    params_matlab = dict(
        l1=0.0728, l2=0.237, l3=0.172,
        lc1=0.0728 / 2, lc2=0.237 / 2, lc3=0.172 / 2,
        g=9.81,
        m1=0.40024, m2=0.2436, m3=0.0345,
        Ixx1=0.006608779, Iyy1=0.002231977, Izz1=0.005585125,
        Ixx2=0.006608779, Iyy2=0.002231977, Izz2=0.005585125,
        Ixx3=0.006608779, Iyy3=0.002231977, Izz3=0.005585125,
    )

    # Equivalent single-axis parameters extracted from the MuJoCo CAD model
    # (link1: 9.25g, link2: 94.13g with com at +0.1205m along link, link3: 11.79g with com at +0.04539m along link)
    params_cad = dict(
        l1=0.07895, l2=0.23700, l3=0.18000,
        lc1=0.06707, lc2=0.12050, lc3=0.04539,
        g=9.81,
        m1=0.00925, m2=0.09413, m3=0.01179,
        Ixx1=1.65e-6, Iyy1=3.65e-6, Izz1=4.33e-6,
        Ixx2=14.56e-6, Iyy2=1010.12e-6, Izz2=1014.22e-6,
        Ixx3=3.21e-6, Iyy3=39.84e-6, Izz3=42.02e-6,
    )

    ee_mj = np.zeros((len(ts), 3))
    ee_dh_matlab = np.zeros((len(ts), 3))
    ee_dh_cad = np.zeros((len(ts), 3))
    tau_mj = np.zeros((len(ts), 3))
    tau_lit_matlab = np.zeros((len(ts), 3))
    tau_cor_cad = np.zeros((len(ts), 3))

    def eval_tau(fn, q, qd, qdd, p):
        return np.asarray(fn(
            q[0], q[1], q[2], qd[0], qd[1], qd[2], qdd[0], qdd[1], qdd[2],
            p["l1"], p["l2"], p["l3"], p["lc1"], p["lc2"], p["lc3"], p["g"],
            p["m1"], p["m2"], p["m3"],
            p["Ixx1"], p["Iyy1"], p["Izz1"],
            p["Ixx2"], p["Iyy2"], p["Izz2"],
            p["Ixx3"], p["Iyy3"], p["Izz3"],
        ), dtype=np.float64).ravel()

    for i in range(len(ts)):
        q, qd, qdd = qs[i], qds[i], qdds[i]
        data.qpos[:] = q
        data.qvel[:] = qd
        data.qacc[:] = qdd
        mujoco.mj_inverse(model, data)
        ee_mj[i] = data.site("end_effector").xpos
        tau_mj[i] = data.qfrc_inverse

        # Note sign of q2/q3 in DH vs MuJoCo: check whether +q2/+q3 pitch in same direction after Z+90 rot!
        # In DH: S12(q2) -> x2 = [0, -sin(q2), cos(q2)]. After dh_to_mj_world (-y, x, z): [sin(q2), 0, cos(q2)] (+X for +q2).
        # Let's check MuJoCo sign at q2 > 0!
        p_dh_m = np.asarray(fk_fn(q[0], q[1], q[2], params_matlab["l1"], params_matlab["l2"], params_matlab["l3"]), dtype=np.float64).ravel()
        p_dh_c = np.asarray(fk_fn(q[0], q[1], q[2], params_cad["l1"], params_cad["l2"], params_cad["l3"]), dtype=np.float64).ravel()
        ee_dh_matlab[i] = dh_to_mj_world(p_dh_m)
        ee_dh_cad[i] = dh_to_mj_world(p_dh_c)

        tau_lit_matlab[i] = eval_tau(tau_lit_fn, q, qd, qdd, params_matlab)
        tau_cor_cad[i] = eval_tau(tau_cor_fn, q, qd, qdd, params_cad)

    # Check if q2/q3 sign in DH is opposite to MuJoCo!
    fk_err_same = np.mean(np.linalg.norm(ee_mj - ee_dh_cad, axis=1))
    ee_dh_cad_flip = np.zeros_like(ee_dh_cad)
    ee_dh_matlab_flip = np.zeros_like(ee_dh_matlab)
    tau_cor_cad_flip = np.zeros_like(tau_cor_cad)
    for i in range(len(ts)):
        q, qd, qdd = qs[i], qds[i], qdds[i]
        qf = np.array([q[0], -q[1], -q[2]])
        qdf = np.array([qd[0], -qd[1], -qd[2]])
        qddf = np.array([qdd[0], -qdd[1], -qdd[2]])
        ee_dh_cad_flip[i] = dh_to_mj_world(np.asarray(fk_fn(qf[0], qf[1], qf[2], params_cad["l1"], params_cad["l2"], params_cad["l3"]), dtype=np.float64).ravel())
        ee_dh_matlab_flip[i] = dh_to_mj_world(np.asarray(fk_fn(qf[0], qf[1], qf[2], params_matlab["l1"], params_matlab["l2"], params_matlab["l3"]), dtype=np.float64).ravel())
        tf = eval_tau(tau_cor_fn, qf, qdf, qddf, params_cad)
        tau_cor_cad_flip[i] = np.array([tf[0], -tf[1], -tf[2]])
    fk_err_flip = np.mean(np.linalg.norm(ee_mj - ee_dh_cad_flip, axis=1))

    if fk_err_flip < fk_err_same:
        print(f"Sign convention detected: q_dh = [q1, -q2, -q3] (FK err {fk_err_flip*1000:.2f} mm vs {fk_err_same*1000:.2f} mm)")
        ee_dh_cad = ee_dh_cad_flip
        ee_dh_matlab = ee_dh_matlab_flip
        tau_cor_cad = tau_cor_cad_flip
        # Also evaluate literal cd1_live.mlx at aligned signs
        for i in range(len(ts)):
            q, qd, qdd = qs[i], qds[i], qdds[i]
            tf = eval_tau(tau_lit_fn, [q[0], -q[1], -q[2]], [qd[0], -qd[1], -qd[2]], [qdd[0], -qdd[1], -qdd[2]], params_matlab)
            tau_lit_matlab[i] = np.array([tf[0], -tf[1], -tf[2]])
    else:
        print(f"Sign convention detected: q_dh = [q1, +q2, +q3] (FK err {fk_err_same*1000:.2f} mm)")

    # Print numerical summary
    fk_rmse_matlab = np.sqrt(np.mean((ee_mj - ee_dh_matlab) ** 2, axis=0)) * 1000.0
    fk_rmse_cad = np.sqrt(np.mean((ee_mj - ee_dh_cad) ** 2, axis=0)) * 1000.0
    tau_rmse_lit = np.sqrt(np.mean((tau_mj - tau_lit_matlab) ** 2, axis=0))
    tau_rmse_cor = np.sqrt(np.mean((tau_mj - tau_cor_cad) ** 2, axis=0))

    print("\n=== KINEMATICS COMPARISON (End-Effector Position RMSE) ===")
    print(f"  MuJoCo vs cd1_live.mlx (l=[0.0728, 0.237, 0.172]): RMSE = {np.round(fk_rmse_matlab, 2)} mm (3D norm={np.linalg.norm(fk_rmse_matlab):.2f} mm)")
    print(f"  MuJoCo vs DH w/ CAD lengths (l=[0.07895, 0.237, 0.180]): RMSE = {np.round(fk_rmse_cad, 2)} mm (3D norm={np.linalg.norm(fk_rmse_cad):.2f} mm)")

    print("\n=== INVERSE DYNAMICS COMPARISON (Torque RMSE) ===")
    print(f"  MuJoCo peak torque:                              {np.round(np.max(np.abs(tau_mj), axis=0), 5)} N*m")
    print(f"  cd1_live.mlx (literal) peak torque:              {np.round(np.max(np.abs(tau_lit_matlab), axis=0), 5)} N*m")
    print(f"  Corrected E-L (CAD params) peak torque:          {np.round(np.max(np.abs(tau_cor_cad), axis=0), 5)} N*m")
    print(f"  MuJoCo vs cd1_live.mlx (literal) RMSE:           {np.round(tau_rmse_lit, 5)} N*m")
    print(f"  MuJoCo vs Corrected E-L (CAD params) RMSE:       {np.round(tau_rmse_cor, 5)} N*m")

    # Plot 1: Kinematics comparison
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for j, ax_lbl in enumerate(["X [m]", "Y [m]", "Z [m]"]):
        axes[j].plot(ts, ee_mj[:, j], "k-", lw=2, label="MuJoCo (CAD)")
        axes[j].plot(ts, ee_dh_cad[:, j], "--", color="#1f77b4", lw=1.6, label="DH Analytical (CAD lengths)")
        axes[j].plot(ts, ee_dh_matlab[:, j], ":", color="#d62728", lw=1.6, label="cd1_live.mlx (original lengths)")
        axes[j].set_ylabel(ax_lbl)
        axes[j].grid(True, alpha=0.3)
        if j == 0:
            axes[j].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Forward Kinematics Comparison: MuJoCo vs MATLAB (cd1_live.mlx)")
    axes[2].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(RENDER_DIR / "matlab_vs_mujoco_kinematics.png", dpi=150)
    plt.close(fig)

    # Plot 2: Dynamics comparison
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for j, lbl in enumerate(["tau1 [N*m]", "tau2 [N*m]", "tau3 [N*m]"]):
        axes[j].plot(ts, tau_mj[:, j], "k-", lw=2, label="MuJoCo mj_inverse (CAD)")
        axes[j].plot(ts, tau_cor_cad[:, j], "--", color="#2ca02c", lw=1.6, label="Corrected Euler-Lagrange (CAD params)")
        axes[j].plot(ts, tau_lit_matlab[:, j], ":", color="#d62728", lw=1.6, label="cd1_live.mlx (original params + equations)")
        axes[j].set_ylabel(lbl)
        axes[j].grid(True, alpha=0.3)
        if j == 0:
            axes[j].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Inverse Dynamics Comparison: MuJoCo vs MATLAB (cd1_live.mlx)")
    axes[2].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(RENDER_DIR / "matlab_vs_mujoco_dynamics.png", dpi=150)
    plt.close(fig)

    # Export .mat file for direct MATLAB comparison!
    mat_path = HERE / "mujoco_comparison.mat"
    sio.savemat(
        mat_path,
        {
            "t": ts,
            "q": qs,
            "qd": qds,
            "qdd": qdds,
            "ee_mujoco": ee_mj,
            "ee_dh_cad": ee_dh_cad,
            "ee_dh_matlab": ee_dh_matlab,
            "tau_mujoco": tau_mj,
            "tau_el_cad": tau_cor_cad,
            "tau_cd1_live": tau_lit_matlab,
        },
    )
    print(f"\nSaved plots and MATLAB data file:\n  {RENDER_DIR / 'matlab_vs_mujoco_kinematics.png'}\n  {RENDER_DIR / 'matlab_vs_mujoco_dynamics.png'}\n  {mat_path}")


if __name__ == "__main__":
    main()
