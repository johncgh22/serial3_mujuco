#!/usr/bin/env python3
"""Run and test the 3-DOF serial robot in MuJoCo.

Examples:
  # 1. Interactive 3D viewer with computed-torque trajectory tracking:
  ~/.venvs/mujoco/bin/python run_demo.py --mode viewer

  # 2. Headless computed-torque + PD tracking test (saves trajectory_tracking.png):
  ~/.venvs/mujoco/bin/python run_demo.py --mode pd

  # 3. Print M(q), C(q,qd)*qd + g(q), inverse dynamics, and Jacobian J(q):
  ~/.venvs/mujoco/bin/python run_demo.py --mode dynamics

  # 4. Replay the Simscape q_serial3.mat trajectory in the interactive viewer:
  ~/.venvs/mujoco/bin/python run_demo.py --mode replay
  # 5. Interactive AX-12A position-control viewer (live Control sliders + optional Moving Speed demo):
  ~/.venvs/mujoco/bin/python run_demo.py --mode ax12a
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCENE_XML = HERE / "scene.xml"
SCENE_AX12A_XML = HERE / "scene_ax12a.xml"


def desired_trajectory(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth 3-DOF sinusoidal reference centered around the upright zero pose q=[0,0,0]."""
    q0 = np.array([0.0, 0.0, 0.0])
    amp = np.array([0.8, 0.45, 0.6])
    w = np.array([1.2, 1.5, 2.0])
    q_des = q0 + amp * np.sin(w * t)
    qd_des = amp * w * np.cos(w * t)
    qdd_des = -amp * (w**2) * np.sin(w * t)
    return q_des, qd_des, qdd_des


def computed_torque_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_des: np.ndarray,
    qd_des: np.ndarray,
    qdd_des: np.ndarray,
    kp: np.ndarray = np.array([40.0, 60.0, 40.0]),
    kd: np.ndarray = np.array([12.0, 15.0, 12.0]),
) -> np.ndarray:
    """Inverse-dynamics (computed-torque) controller: tau = M(q)*(qdd_d + Kp*e + Kd*ed) + C*qd + g."""
    e = q_des - data.qpos
    ed = qd_des - data.qvel
    v = qdd_des + kp * e + kd * ed
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    tau = M @ v + data.qfrc_bias
    return np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def run_pd_headless(duration: float = 5.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    data.qpos[:] = [0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    steps = int(duration / model.opt.timestep)
    ts = np.zeros(steps)
    qs = np.zeros((steps, 3))
    qds = np.zeros((steps, 3))
    q_refs = np.zeros((steps, 3))
    taus = np.zeros((steps, 3))

    for i in range(steps):
        t = data.time
        q_des, qd_des, qdd_des = desired_trajectory(t)
        data.ctrl[:] = computed_torque_control(model, data, q_des, qd_des, qdd_des)
        ts[i] = t
        qs[i] = data.qpos
        qds[i] = data.qvel
        q_refs[i] = q_des
        taus[i] = data.ctrl
        mujoco.mj_step(model, data)

    rmse = np.sqrt(np.mean((qs - q_refs) ** 2, axis=0))
    print(f"Computed-torque tracking ({duration:.1f}s): RMSE = {np.round(np.rad2deg(rmse), 4)} deg")
    print(f"Peak applied torques: {np.round(np.max(np.abs(taus), axis=0), 4)} N*m")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    labels = ["q1 (yaw)", "q2 (shoulder)", "q3 (elbow)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for j in range(3):
        axes[0].plot(ts, qs[:, j], color=colors[j], label=f"{labels[j]} actual")
        axes[0].plot(ts, q_refs[:, j], "--", color=colors[j], alpha=0.7, label=f"{labels[j]} ref")
        axes[1].plot(ts, taus[:, j], color=colors[j], label=f"tau{j+1}")
    axes[0].set_ylabel("Joint Angle [rad]")
    axes[0].set_title("MuJoCo 3-DOF Robot: Computed-Torque Trajectory Tracking")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set_ylabel("Torque [N*m]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out_plot = HERE / "renders" / "trajectory_tracking.png"
    out_plot.parent.mkdir(exist_ok=True)
    fig.savefig(out_plot, dpi=150)
    print(f"Saved tracking plot: {out_plot}")


def run_dynamics_demo():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    q = np.array([0.4, -0.5, -0.7])
    qd = np.array([0.5, -0.2, 0.3])
    qdd = np.array([0.1, 0.2, -0.1])

    data.qpos[:] = q
    data.qvel[:] = qd
    mujoco.mj_forward(model, data)

    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    bias = data.qfrc_bias.copy()  # C(q, qd)*qd + g(q)

    data.qacc[:] = qdd
    mujoco.mj_inverse(model, data)
    tau_inv = data.qfrc_inverse.copy()

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)

    print(f"State: q = {q}, qd = {qd}, qdd = {qdd}")
    print(f"End-effector position: {np.round(data.site('end_effector').xpos, 5)} m")
    print(f"\nMass matrix M(q) [kg*m^2]:\n{np.array2string(M, precision=6, suppress_small=True)}")
    print(f"\nBias force C(q,qd)*qd + g(q) [N*m]: {np.round(bias, 6)}")
    print(f"Inverse dynamics tau = M*qdd + C*qd + g [N*m]: {np.round(tau_inv, 6)}")
    print(f"Linear Jacobian J_p(q) [m/rad]:\n{np.array2string(jacp, precision=5, suppress_small=True)}")


def run_viewer(replay_mat: bool = False, mat_path: Path | None = None):
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    data.qpos[:] = [0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    q_traj = None
    if replay_mat:
        import scipy.io as sio
        target_mat = mat_path or (HERE / "mujoco_comparison.mat")
        if not target_mat.is_file():
            target_mat = ROOT / "q_serial3.mat"
        q_traj = np.asarray(sio.loadmat(target_mat)["q"], dtype=np.float64)
        print(f"Loaded {target_mat.name} ({len(q_traj)} samples). Replaying...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.28]
        viewer.cam.distance = 0.85
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        step_idx = 0
        while viewer.is_running():
            t0 = time.perf_counter()
            if q_traj is not None:
                i = step_idx % len(q_traj)
                data.qpos[:] = q_traj[i]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                step_idx += 5
                dt_sleep = 0.01
            else:
                q_des, qd_des, qdd_des = desired_trajectory(data.time)
                data.ctrl[:] = computed_torque_control(model, data, q_des, qd_des, qdd_des)
                mujoco.mj_step(model, data)
                dt_sleep = model.opt.timestep
            viewer.sync()
            rem = dt_sleep - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)


def run_ax12a_viewer():
    """Launch the AX-12A position-controlled scene where Control sliders directly set joint angles."""
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(SCENE_AX12A_XML))
    data = mujoco.MjData(model)
    data.qpos[:] = [0.0, 0.0, 0.0]
    data.ctrl[:] = [0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    print("AX-12A position-control mode active.")
    print("Open the right sidebar (Shift+Tab) -> 'Control' to drag joint angle sliders (rad).")
    mujoco.viewer.launch(model, data)


def main():
    parser = argparse.ArgumentParser(description="Run 3-DOF serial robot in MuJoCo.")
    parser.add_argument(
        "--mode",
        choices=["viewer", "pd", "dynamics", "replay", "ax12a"],
        default="pd",
        help="Execution mode (default: pd).",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Duration for --mode pd [s].")
    parser.add_argument("--mat", type=Path, default=None, help="Optional .mat file containing 'q' for --mode replay.")
    args = parser.parse_args()

    if args.mode == "pd":
        run_pd_headless(args.duration)
    elif args.mode == "dynamics":
        run_dynamics_demo()
    elif args.mode == "viewer":
        run_viewer(replay_mat=False)
    elif args.mode == "replay":
        run_viewer(replay_mat=True, mat_path=args.mat)
    elif args.mode == "ax12a":
        run_ax12a_viewer()


if __name__ == "__main__":
    main()
