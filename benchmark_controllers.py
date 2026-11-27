#!/usr/bin/env python3
"""Benchmark & interactive viewer for 3-DOF serial robot controllers in MuJoCo.

Evaluates all controllers in `controllers/` head-to-head across three tests:
  1. Nominal (`scene.xml` — exact CAD masses, zero gearbox friction):
       Smooth 3-DOF trajectory away from overhead singularity.
  2. Perturbed / Sim-to-Real (via MjSpec):
       +100 g unknown end-effector payload, +25% link2 mass mismatch,
       AX-12A 254:1 gearbox dynamics (armature=0.003, damping=0.24,
       frictionloss=0.03), +/-1.5 N*m torque limits, and an external torque
       impulse disturbance at t = 2.5..2.8 s.
  3. Cartesian 3D Circle:
       Direct end-effector circle tracking comparing OSC, Cartesian Impedance,
       CTC, and Super-Twisting SMC.

Examples:
  # Run the full headless benchmark and generate comparison plots:
  python benchmark_controllers.py

  # Watch any controller live in the interactive 3D MuJoCo viewer:
  python benchmark_controllers.py --viewer --controller st_smc --scenario perturbed
  python benchmark_controllers.py --viewer --controller osc --trajectory circle
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from controllers import (
    AnalyticalDynamics,
    BaseController,
    CAD_PARAMS,
    CartesianCircleTrajectory,
    CartesianImpedanceController,
    ComputedTorqueController,
    LTVMPCController,
    OperationalSpaceController,
    PDGravityController,
    PIDController,
    SINDyDynamics,
    SlidingModeController,
    SlotineLiAdaptiveController,
    SmoothJointTrajectory,
    SugenoFuzzyController,
    SuperTwistingSMC,
    collect_mujoco_excitation,
    fit_sindy_model,
)

HERE = Path(__file__).resolve().parent
SCENE_XML = HERE / "scene.xml"
RENDER_DIR = HERE / "renders"

# External disturbance pulse applied during the perturbed benchmark
DISTURBANCE_T_START = 2.5
DISTURBANCE_T_END = 2.8
DISTURBANCE_TORQUE = np.array([0.12, -0.20, 0.08], dtype=np.float64)


@dataclass
class SimResult:
    controller_key: str
    name: str
    short_name: str
    control_space: str
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    q_ref: np.ndarray
    x: np.ndarray
    x_ref: np.ndarray
    tau: np.ndarray
    joint_rmse_deg: np.ndarray
    ee_rmse_mm: float
    ee_max_mm: float
    ee_steady_rmse_mm: float
    peak_torque: float
    effort_int_tau2: float
    smoothness_dtau_dt: float
    theta_history: np.ndarray | None = None


def verify_analytical_core(dyn: AnalyticalDynamics) -> None:
    """Fast self-check on closed-form dynamics, regressor, J_dot, and analytical IK."""
    rng = np.random.default_rng(42)
    max_reg_err = 0.0
    max_jdot_err = 0.0
    eps = 1e-6

    for _ in range(30):
        q = rng.uniform([-2.0, 0.2, 0.2], [2.0, 1.3, 1.5])
        qd = rng.uniform(-2.0, 2.0, size=3)
        qd_r = rng.uniform(-2.0, 2.0, size=3)
        qdd_r = rng.uniform(-4.0, 4.0, size=3)

        tau_ref = dyn.mass_matrix(q) @ qdd_r + dyn.coriolis_matrix(q, qd) @ qd_r + dyn.gravity_vector(q)
        Y = dyn.regressor(q, qd, qd_r, qdd_r, include_friction=False)
        max_reg_err = max(max_reg_err, float(np.max(np.abs(Y @ dyn.theta_nominal - tau_ref))))

        Jd = dyn.jacobian_v_dot(q, qd)
        Jd_fd = (dyn.jacobian_v(q + eps * qd) - dyn.jacobian_v(q - eps * qd)) / (2.0 * eps)
        max_jdot_err = max(max_jdot_err, float(np.max(np.abs(Jd - Jd_fd))))

    circle = CartesianCircleTrajectory(dyn)
    ref0 = circle(0.7)
    ik_err = float(np.linalg.norm(dyn.fk(ref0.q_d) - ref0.x_d))

    assert max_reg_err < 1e-13, f"Regressor mismatch: {max_reg_err}"
    assert max_jdot_err < 1e-7, f"J_dot mismatch: {max_jdot_err}"
    assert ik_err < 1e-12, f"Circle analytical IK mismatch: {ik_err}"
    print(
        f"[PASS] Analytical core verified: regressor err={max_reg_err:.2e}, "
        f"J_dot err={max_jdot_err:.2e}, circle IK err={ik_err:.2e} m"
    )


def build_sim_model(scenario: str, with_target_marker: bool = False) -> mujoco.MjModel:
    """Build either the nominal or perturbed/AX-12A MuJoCo plant via MjSpec."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))

    if scenario == "perturbed":
        # 1. Add AX-12A 254:1 gearbox armature, viscous damping, and Coulomb friction
        for jname in ("joint1", "joint2", "joint3"):
            jnt = spec.joint(jname)
            jnt.armature = 0.003
            jnt.damping = [0.24, 0.0, 0.0]
            jnt.frictionloss = 0.03

        # 2. +25% unmodeled mass & inertia mismatch on link2
        link2 = spec.body("link2")
        link2.mass = float(link2.mass) * 1.25
        link2.inertia = np.asarray(link2.inertia, dtype=np.float64) * 1.25

        # 3. Unknown +100 g payload attached at the end-effector tip of link3
        link3 = spec.body("link3")
        payload = link3.add_body(name="unknown_payload", pos=[0.0, -0.18, 0.021])
        payload.add_geom(
            name="unknown_payload_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.022, 0.0, 0.0],
            mass=0.100,
            contype=0,
            conaffinity=0,
            rgba=[0.92, 0.25, 0.18, 0.90],
        )
    elif scenario != "nominal":
        raise ValueError(f"Unknown scenario: {scenario!r}")

    if with_target_marker:
        # Mocap sphere showing live desired end-effector position in viewer mode
        marker = spec.worldbody.add_body(name="target_marker", mocap=True, pos=[0.0, 0.0, 0.3])
        marker.add_geom(
            name="target_marker_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.014, 0.0, 0.0],
            contype=0,
            conaffinity=0,
            rgba=[0.15, 0.85, 0.30, 0.55],
        )

    return spec.compile()


def make_controllers(
    dyn: AnalyticalDynamics,
    trajectory_fn=None,
    sindy_dyn: SINDyDynamics | None = None,
) -> dict[str, BaseController]:
    """Instantiate the full controller suite (nominal analytical + optional SINDy models)."""
    ctrls: dict[str, BaseController] = {
        "pid": PIDController(),
        "pdg": PDGravityController(dyn),
        "ctc": ComputedTorqueController(dyn),
        "adaptive": SlotineLiAdaptiveController(dyn),
        "smc": SlidingModeController(dyn),
        "st_smc": SuperTwistingSMC(dyn),
        "fuzzy": SugenoFuzzyController(dyn),
        "osc": OperationalSpaceController(dyn),
        "impedance": CartesianImpedanceController(dyn),
        "mpc": LTVMPCController(
            dyn,
            trajectory_fn=trajectory_fn,
            use_disturbance_observer=False,
            name="LTV-MPC (Nominal)",
            short_name="MPC",
        ),
    }
    if sindy_dyn is not None:
        ctrls["sindy_ctc"] = ComputedTorqueController(
            sindy_dyn,
            kp=(260.0, 260.0, 260.0),
            kd=(32.0, 32.0, 32.0),
            name="SINDy + CTC",
            short_name="SINDy-CTC",
        )
        ctrls["sindy_mpc"] = LTVMPCController(
            sindy_dyn,
            trajectory_fn=trajectory_fn,
            use_disturbance_observer=True,
            name="SINDy-MPC (DOB)",
            short_name="SINDy-MPC",
        )
    return ctrls


def simulate_controller(
    key: str,
    controller: BaseController,
    model: mujoco.MjModel,
    trajectory,
    duration: float = 6.0,
    apply_disturbance: bool = False,
) -> SimResult:
    """Run one headless closed-loop simulation and compute tracking/effort metrics."""
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    steps = int(round(duration / dt))

    q0, qd0 = trajectory.initial_state()
    data.qpos[:] = q0
    data.qvel[:] = qd0
    mujoco.mj_forward(model, data)
    controller.reset()

    ts = np.zeros(steps)
    qs = np.zeros((steps, 3))
    qds = np.zeros((steps, 3))
    q_refs = np.zeros((steps, 3))
    xs = np.zeros((steps, 3))
    x_refs = np.zeros((steps, 3))
    taus = np.zeros((steps, 3))

    for i in range(steps):
        t = float(data.time)
        ref = trajectory(t)
        q = data.qpos.copy()
        qd = data.qvel.copy()
        x = data.site("end_effector").xpos.copy()

        tau = controller.compute(t, q, qd, ref, dt)
        data.ctrl[:] = tau

        if apply_disturbance and (DISTURBANCE_T_START <= t <= DISTURBANCE_T_END):
            data.qfrc_applied[:] = DISTURBANCE_TORQUE
        else:
            data.qfrc_applied[:] = 0.0

        ts[i] = t
        qs[i] = q
        qds[i] = qd
        q_refs[i] = ref.q_d
        xs[i] = x
        x_refs[i] = ref.x_d
        taus[i] = tau

        mujoco.mj_step(model, data)

    q_err = qs - q_refs
    ee_err = np.linalg.norm(xs - x_refs, axis=1)
    # Steady-state window: final 2 seconds (after adaptation / disturbance recovery)
    steady_mask = ts >= (duration - 2.0)

    joint_rmse_deg = np.rad2deg(np.sqrt(np.mean(q_err ** 2, axis=0)))
    ee_rmse_mm = float(np.sqrt(np.mean(ee_err ** 2)) * 1000.0)
    ee_max_mm = float(np.max(ee_err) * 1000.0)
    ee_steady_rmse_mm = float(np.sqrt(np.mean(ee_err[steady_mask] ** 2)) * 1000.0)
    peak_torque = float(np.max(np.abs(taus)))
    effort_int_tau2 = float(np.sum(np.sum(taus ** 2, axis=1)) * dt)
    smoothness_dtau_dt = float(np.mean(np.linalg.norm(np.diff(taus, axis=0), axis=1)) / dt)

    theta_hist = None
    if isinstance(controller, SlotineLiAdaptiveController) and controller.theta_history:
        theta_hist = np.asarray(controller.theta_history, dtype=np.float64)

    return SimResult(
        controller_key=key,
        name=controller.name,
        short_name=controller.short_name,
        control_space=controller.control_space,
        t=ts,
        q=qs,
        qd=qds,
        q_ref=q_refs,
        x=xs,
        x_ref=x_refs,
        tau=taus,
        joint_rmse_deg=joint_rmse_deg,
        ee_rmse_mm=ee_rmse_mm,
        ee_max_mm=ee_max_mm,
        ee_steady_rmse_mm=ee_steady_rmse_mm,
        peak_torque=peak_torque,
        effort_int_tau2=effort_int_tau2,
        smoothness_dtau_dt=smoothness_dtau_dt,
        theta_history=theta_hist,
    )


def print_table(title: str, results: list[SimResult]) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'Controller':<26} | {'Space':<9} | {'q RMSE [deg] (q1, q2, q3)':<25} | "
        f"{'EE RMSE':>8} | {'Last 2s':>8} | {'EE Max':>8} | {'Peak |τ|':>8} | {'∫‖τ‖²dt':>8} | {'Mean|dτ/dt|':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        q_str = f"{r.joint_rmse_deg[0]:5.2f}, {r.joint_rmse_deg[1]:5.2f}, {r.joint_rmse_deg[2]:5.2f}"
        print(
            f"{r.name:<26} | {r.control_space:<9} | {q_str:<25} | "
            f"{r.ee_rmse_mm:6.2f}mm | {r.ee_steady_rmse_mm:6.2f}mm | {r.ee_max_mm:6.2f}mm | "
            f"{r.peak_torque:6.3f}Nm | {r.effort_int_tau2:8.4f} | {r.smoothness_dtau_dt:9.2f}Nm/s"
        )


def plot_scenario(results: list[SimResult], title: str, out_path: Path, perturbed: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.cm.tab20(np.linspace(0, 0.95, len(results)))
    nrows = 3 if not perturbed else 4
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 3.0 * nrows), sharex=True)

    # Panel 1: End-effector 3D tracking error (mm)
    ax = axes[0]
    for c, r in zip(colors, results):
        err_mm = np.linalg.norm(r.x - r.x_ref, axis=1) * 1000.0
        ax.plot(r.t, err_mm, color=c, lw=1.4, label=f"{r.short_name} ({r.ee_rmse_mm:.2f} mm)")
    if perturbed:
        ax.axvspan(DISTURBANCE_T_START, DISTURBANCE_T_END, color="crimson", alpha=0.14, label="External impulse")
    ax.set_ylabel("EE 3D Error [mm]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8, loc="upper right")

    # Panel 2: Shoulder joint tracking error q2 (heaviest gravity load)
    ax = axes[1]
    for c, r in zip(colors, results):
        q2_err_deg = np.rad2deg(r.q[:, 1] - r.q_ref[:, 1])
        ax.plot(r.t, q2_err_deg, color=c, lw=1.2, label=r.short_name)
    if perturbed:
        ax.axvspan(DISTURBANCE_T_START, DISTURBANCE_T_END, color="crimson", alpha=0.14)
    ax.set_ylabel("Shoulder q2 Error [deg]")
    ax.grid(True, alpha=0.3)

    # Panel 3: Shoulder torque tau_2
    ax = axes[2]
    for c, r in zip(colors, results):
        ax.plot(r.t, r.tau[:, 1], color=c, lw=1.1, alpha=0.85, label=r.short_name)
    ax.axhline(1.5, color="black", ls=":", lw=1.0, alpha=0.6, label="AX-12A stall (±1.5 N·m)")
    ax.axhline(-1.5, color="black", ls=":", lw=1.0, alpha=0.6)
    if perturbed:
        ax.axvspan(DISTURBANCE_T_START, DISTURBANCE_T_END, color="crimson", alpha=0.14)
    ax.set_ylabel("Shoulder τ2 [N·m]")
    ax.grid(True, alpha=0.3)

    # Panel 4 (perturbed only): Slotine-Li online parameter adaptation
    if perturbed:
        ax = axes[3]
        adapt = next((r for r in results if r.theta_history is not None), None)
        if adapt is not None and adapt.theta_history is not None:
            th = adapt.theta_history
            ax.plot(adapt.t, th[:, 8], lw=1.6, label="θ8: m2·lc2 + m3·l2 (gravity shoulder)")
            ax.plot(adapt.t, th[:, 9], lw=1.6, label="θ9: m3·lc3 (gravity elbow + payload)")
            ax.plot(adapt.t, th[:, 11], lw=1.4, ls="--", label="d2: shoulder viscous damping (true 0.24)")
            ax.plot(adapt.t, th[:, 14], lw=1.4, ls=":", label="fc2: shoulder Coulomb friction (true 0.03)")
            ax.axhline(0.24, color="gray", ls="--", lw=0.9, alpha=0.6)
            ax.set_ylabel("Adaptive θ̂")
            ax.legend(ncol=2, fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_summary_bars(nom: list[SimResult], pert: list[SimResult], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r.short_name for r in nom]
    x = np.arange(len(labels))
    w = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    # 1. End-effector RMSE (mm)
    axes[0].bar(x - w / 2, [r.ee_rmse_mm for r in nom], w, label="Nominal", color="#3b82f6")
    axes[0].bar(x + w / 2, [r.ee_rmse_mm for r in pert], w, label="Perturbed (+100g + AX-12A)", color="#ef4444")
    axes[0].set_title("End-Effector 3D RMSE [mm] (lower = better)")
    axes[0].set_ylabel("RMSE [mm]")
    axes[0].set_xticks(x, labels, rotation=32, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    # 2. Steady-state (last 2s) EE RMSE (mm)
    axes[1].bar(x - w / 2, [r.ee_steady_rmse_mm for r in nom], w, label="Nominal", color="#3b82f6")
    axes[1].bar(x + w / 2, [r.ee_steady_rmse_mm for r in pert], w, label="Perturbed (last 2s)", color="#ef4444")
    axes[1].set_title("Steady-State EE RMSE (Last 2 s) [mm]")
    axes[1].set_ylabel("RMSE [mm]")
    axes[1].set_xticks(x, labels, rotation=32, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)

    # 3. Control effort integral ||tau||^2 dt
    axes[2].bar(x - w / 2, [r.effort_int_tau2 for r in nom], w, label="Nominal", color="#3b82f6")
    axes[2].bar(x + w / 2, [r.effort_int_tau2 for r in pert], w, label="Perturbed", color="#ef4444")
    axes[2].set_title("Control Energy ∫‖τ‖² dt [N²·m²·s]")
    axes[2].set_ylabel("Effort [N²·m²·s]")
    axes[2].set_xticks(x, labels, rotation=32, ha="right")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("3-DOF Serial Robot Controller Benchmark: Nominal vs. Sim-to-Real Perturbed", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_cartesian_circle(results_nom: list[SimResult], results_pert: list[SimResult], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    colors = ["#2563eb", "#16a34a", "#ea580c", "#9333ea", "#dc2626"]

    for ax, res_list, subtitle in (
        (axes[0], results_nom, "Nominal Plant (Y-Z Plane Projection)"),
        (axes[1], results_pert, "Perturbed Plant (+100 g Payload + AX-12A Friction)"),
    ):
        ref = res_list[0].x_ref
        ax.plot(ref[:, 1] * 1000, ref[:, 2] * 1000, "k--", lw=1.6, label="Reference Circle")
        for c, r in zip(colors, res_list):
            ax.plot(r.x[:, 1] * 1000, r.x[:, 2] * 1000, color=c, lw=1.4, label=f"{r.short_name} ({r.ee_rmse_mm:.2f} mm)")
        ax.set_aspect("equal", "box")
        ax.set_xlabel("Y [mm]")
        ax.set_ylabel("Z [mm]")
        ax.set_title(subtitle)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("3D Cartesian Circle Tracking (Operational Space vs. Joint-Space Controllers)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def run_viewer(controller_key: str, scenario: str, trajectory_kind: str) -> None:
    import mujoco.viewer

    dyn = AnalyticalDynamics(CAD_PARAMS)
    traj = SmoothJointTrajectory(dyn) if trajectory_kind == "joint" else CartesianCircleTrajectory(dyn)
    model = build_sim_model(scenario, with_target_marker=True)

    sindy_dyn = None
    if controller_key in ("sindy_ctc", "sindy_mpc"):
        print(f"Identifying SINDy model on {scenario} plant before launching viewer...")
        id_model = build_sim_model(scenario)
        exc = collect_mujoco_excitation(id_model, duration=12.0, subsample=5)
        sindy_dyn, metrics = fit_sindy_model(exc["q"], exc["qd"], exc["qdd"], exc["tau"])
        print(f"SINDy identified ({metrics['n_terms_active']} terms, R²={metrics['r2_mean']:.5f}).")

    controllers = make_controllers(dyn, trajectory_fn=traj, sindy_dyn=sindy_dyn)
    if controller_key not in controllers:
        raise ValueError(f"Unknown controller {controller_key!r}. Choose from: {list(controllers)}")
    ctrl = controllers[controller_key]
    ctrl.reset()

    data = mujoco.MjData(model)

    q0, qd0 = traj.initial_state()
    data.qpos[:] = q0
    data.qvel[:] = qd0
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    print(f"Launching viewer: controller={ctrl.name}, scenario={scenario}, trajectory={trajectory_kind}")
    print("Green sphere = live desired end-effector position. Press Esc to close.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.05, 0.0, 0.25]
        viewer.cam.distance = 0.85
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0

        while viewer.is_running():
            t0 = time.perf_counter()
            t = float(data.time)
            ref = traj(t)
            data.mocap_pos[0] = ref.x_d
            tau = ctrl.compute(t, data.qpos.copy(), data.qvel.copy(), ref, dt)
            data.ctrl[:] = tau
            if scenario == "perturbed" and (DISTURBANCE_T_START <= (t % 6.0) <= DISTURBANCE_T_END):
                data.qfrc_applied[:] = DISTURBANCE_TORQUE
            else:
                data.qfrc_applied[:] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()
            rem = dt - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)


def run_benchmark(duration: float = 6.0) -> None:
    dyn = AnalyticalDynamics(CAD_PARAMS)
    verify_analytical_core(dyn)

    joint_traj = SmoothJointTrajectory(dyn)
    circle_traj = CartesianCircleTrajectory(dyn)

    model_nom = build_sim_model("nominal")
    model_pert = build_sim_model("perturbed")

    # Identify Physics-Informed SINDy models from short 12s excitation rollouts
    print("Running Physics-Informed SINDy identification on nominal & perturbed plants...")
    exc_nom = collect_mujoco_excitation(model_nom, duration=12.0, subsample=5)
    sindy_nom, m_nom = fit_sindy_model(exc_nom["q"], exc_nom["qd"], exc_nom["qdd"], exc_nom["tau"])
    print(f"  [SINDy Nominal]   active={m_nom['n_terms_active']}/{m_nom['n_terms_total']}, R²={m_nom['r2_mean']:.5f}")

    exc_pert = collect_mujoco_excitation(model_pert, duration=12.0, subsample=5)
    sindy_pert, m_pert = fit_sindy_model(exc_pert["q"], exc_pert["qd"], exc_pert["qdd"], exc_pert["tau"])
    print(f"  [SINDy Perturbed] active={m_pert['n_terms_active']}/{m_pert['n_terms_total']}, R²={m_pert['r2_mean']:.5f}")

    # 1. Nominal joint-space + OSC + MPC + SINDy benchmark
    nom_results: list[SimResult] = []
    for key, ctrl in make_controllers(dyn, trajectory_fn=joint_traj, sindy_dyn=sindy_nom).items():
        nom_results.append(simulate_controller(key, ctrl, model_nom, joint_traj, duration=duration, apply_disturbance=False))
    print_table("BENCHMARK 1: NOMINAL PLANT (scene.xml, exact CAD, zero friction)", nom_results)

    # 2. Perturbed / Sim-to-Real benchmark (+100 g payload, +25% link2 mass, AX-12A friction, impulse disturbance)
    pert_results: list[SimResult] = []
    for key, ctrl in make_controllers(dyn, trajectory_fn=joint_traj, sindy_dyn=sindy_pert).items():
        pert_results.append(simulate_controller(key, ctrl, model_pert, joint_traj, duration=duration, apply_disturbance=True))
    print_table(
        "BENCHMARK 2: PERTURBED / SIM-TO-REAL (+100g payload, +25% m2, AX-12A friction, impulse @2.5s)",
        pert_results,
    )

    # 3. Direct 3D Cartesian Circle benchmark
    circle_keys = ("osc", "impedance", "ctc", "st_smc", "sindy_mpc")
    circle_nom: list[SimResult] = []
    circle_pert: list[SimResult] = []
    nom_ctrls_c = make_controllers(dyn, trajectory_fn=circle_traj, sindy_dyn=sindy_nom)
    pert_ctrls_c = make_controllers(dyn, trajectory_fn=circle_traj, sindy_dyn=sindy_pert)
    for key in circle_keys:
        circle_nom.append(simulate_controller(key, nom_ctrls_c[key], model_nom, circle_traj, duration=duration, apply_disturbance=False))
        circle_pert.append(simulate_controller(key, pert_ctrls_c[key], model_pert, circle_traj, duration=duration, apply_disturbance=False))
    print_table("BENCHMARK 3A: 3D CARTESIAN CIRCLE (Nominal)", circle_nom)
    print_table("BENCHMARK 3B: 3D CARTESIAN CIRCLE (Perturbed +100g + AX-12A friction)", circle_pert)

    # Generate comparison figures
    plot_scenario(
        nom_results,
        "Nominal Plant Benchmark — Smooth 3-DOF Trajectory",
        RENDER_DIR / "benchmark_nominal.png",
        perturbed=False,
    )
    plot_scenario(
        pert_results,
        "Perturbed / Sim-to-Real Benchmark (+100 g Payload, +25% Link2 Mass, AX-12A Friction, Impulse)",
        RENDER_DIR / "benchmark_perturbed.png",
        perturbed=True,
    )
    plot_summary_bars(nom_results, pert_results, RENDER_DIR / "benchmark_summary_bars.png")
    plot_cartesian_circle(circle_nom, circle_pert, RENDER_DIR / "cartesian_circle.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark or interactively view 3-DOF robot controllers.")
    parser.add_argument("--viewer", action="store_true", help="Launch interactive 3D MuJoCo viewer instead of headless benchmark.")
    parser.add_argument(
        "--controller",
        default="sindy_mpc",
        choices=["pid", "pdg", "ctc", "adaptive", "smc", "st_smc", "fuzzy", "osc", "impedance", "mpc", "sindy_ctc", "sindy_mpc"],
        help="Controller to run in --viewer mode (default: sindy_mpc).",
    )
    parser.add_argument(
        "--scenario",
        default="nominal",
        choices=["nominal", "perturbed"],
        help="Plant model for --viewer mode (default: nominal).",
    )
    parser.add_argument(
        "--trajectory",
        default="joint",
        choices=["joint", "circle"],
        help="Reference trajectory for --viewer mode (default: joint).",
    )
    parser.add_argument("--duration", type=float, default=6.0, help="Simulation duration [s] for headless benchmark.")
    args = parser.parse_args()

    if args.viewer:
        run_viewer(args.controller, args.scenario, args.trajectory)
    else:
        run_benchmark(duration=args.duration)


if __name__ == "__main__":
    main()
