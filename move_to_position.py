#!/usr/bin/env python3
"""Move the 3-DOF AX-12A robot smoothly from a home pose to a desired target.

Supports two target types:
  1. Joint-space vector [q1, q2, q3] (like MATLAB's homeConfiguration -> target config):
       ~/.venvs/mujoco/bin/python move_to_position.py --home 0 0 0 --target 1.0 -0.6 0.8
       ~/.venvs/mujoco/bin/python move_to_position.py --deg --home 0 0 0 --target 45 -30 60

  2. Cartesian end-effector position [x, y, z] in meters (solved via MuJoCo Jacobian IK):
       ~/.venvs/mujoco/bin/python move_to_position.py --home 0 0 0 --xyz 0.20 0.15 0.30

Add `--headless` to run without the 3D viewer and save a plot to renders/move_to_position.png.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SCENE_AX12A_XML = HERE / "scene_ax12a.xml"


def minimum_jerk_interp(q_start: np.ndarray, q_goal: np.ndarray, t: float, duration: float) -> np.ndarray:
    """5th-order minimum-jerk polynomial (zero start/end velocity and acceleration)."""
    if duration <= 0.0:
        return q_goal.copy()
    s = np.clip(t / duration, 0.0, 1.0)
    blend = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    return q_start + blend * (q_goal - q_start)


def solve_ik_xyz(
    model: mujoco.MjModel,
    target_xyz: np.ndarray,
    q_init: np.ndarray,
    max_iters: int = 300,
    tol: float = 1e-4,
    damping: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Damped least-squares inverse kinematics for the 3-DOF end_effector site."""
    data = mujoco.MjData(model)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    q_min = model.jnt_range[:, 0]
    q_max = model.jnt_range[:, 1]

    # Try starting from q_init plus a few non-singular seeds (upright [0,0,0] is singular!)
    seeds = [
        q_init.copy(),
        np.array([np.arctan2(target_xyz[1], target_xyz[0]), -0.4, -0.4]),
        np.array([np.arctan2(target_xyz[1], target_xyz[0]), 0.4, 0.4]),
        np.array([np.arctan2(target_xyz[1], target_xyz[0]) + np.pi, -0.4, 0.4]),
    ]

    best_q = q_init.copy()
    best_err = float("inf")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for seed in seeds:
        q = np.clip(seed, q_min, q_max)
        for _ in range(max_iters):
            data.qpos[:] = q
            mujoco.mj_fwdPosition(model, data)
            err_vec = target_xyz - data.site_xpos[ee_id]
            err = float(np.linalg.norm(err_vec))
            if err < best_err:
                best_err = err
                best_q = q.copy()
            if err < tol:
                return best_q, best_err
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
            # Damped pseudoinverse: dq = J^T (J J^T + lambda^2 I)^-1 e
            A = jacp @ jacp.T + (damping**2) * np.eye(3)
            dq = jacp.T @ np.linalg.solve(A, err_vec)
            # Cap step size for stability near workspace boundary
            step_norm = np.linalg.norm(dq)
            if step_norm > 0.2:
                dq *= 0.2 / step_norm
            q = np.clip(q + dq, q_min, q_max)

    return best_q, best_err


def run_move(
    q_home: np.ndarray,
    q_target: np.ndarray,
    duration: float = 2.5,
    hold_time: float = 1.0,
    headless: bool = False,
):
    model = mujoco.MjModel.from_xml_path(str(SCENE_AX12A_XML))
    data = mujoco.MjData(model)

    q_min = model.jnt_range[:, 0]
    q_max = model.jnt_range[:, 1]
    q_home = np.clip(q_home, q_min, q_max)
    q_target = np.clip(q_target, q_min, q_max)

    # Set initial (home) configuration
    data.qpos[:] = q_home
    data.qvel[:] = 0.0
    data.ctrl[:] = q_home
    mujoco.mj_forward(model, data)

    ee_start = data.site("end_effector").xpos.copy()
    print(f"Home   q = {np.round(q_home, 4)} rad ({np.round(np.rad2deg(q_home), 1)} deg), EE = {np.round(ee_start, 4)} m")
    print(f"Target q = {np.round(q_target, 4)} rad ({np.round(np.rad2deg(q_target), 1)} deg), duration = {duration:.2f} s")

    if headless:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        total_time = duration + hold_time
        steps = int(total_time / model.opt.timestep)
        ts = np.zeros(steps)
        qs = np.zeros((steps, 3))
        q_cmds = np.zeros((steps, 3))
        taus = np.zeros((steps, 3))

        for i in range(steps):
            t = data.time
            q_cmd = minimum_jerk_interp(q_home, q_target, t, duration)
            data.ctrl[:] = q_cmd
            ts[i] = t
            qs[i] = data.qpos
            q_cmds[i] = q_cmd
            taus[i] = data.actuator_force
            mujoco.mj_step(model, data)

        ee_final = data.site("end_effector").xpos.copy()
        err_deg = np.rad2deg(data.qpos - q_target)
        print(f"Final  q = {np.round(data.qpos, 4)} rad (err = {np.round(err_deg, 2)} deg), EE = {np.round(ee_final, 4)} m")

        fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
        for j, lbl in enumerate(["joint1", "joint2", "joint3"]):
            axes[0].plot(ts, qs[:, j], label=f"{lbl} actual")
            axes[0].plot(ts, q_cmds[:, j], "--", alpha=0.7, label=f"{lbl} cmd")
            axes[1].plot(ts, taus[:, j], label=f"tau{j+1}")
        axes[0].set_ylabel("Angle [rad]")
        axes[0].set_title("AX-12A Point-to-Point Move (Home -> Target)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].set_ylabel("Torque [N*m]")
        axes[1].set_xlabel("Time [s]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(ncol=3, fontsize=8)
        fig.tight_layout()
        out_path = HERE / "renders" / "move_to_position.png"
        out_path.parent.mkdir(exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")
        return

    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.26]
        viewer.cam.distance = 0.85
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0

        # Brief pause at home so the starting pose is visible
        t_hold_start = 0.5
        while viewer.is_running():
            t0 = time.perf_counter()
            t_move = max(0.0, data.time - t_hold_start)
            data.ctrl[:] = minimum_jerk_interp(q_home, q_target, t_move, duration)
            mujoco.mj_step(model, data)
            viewer.sync()
            rem = model.opt.timestep - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)


def main():
    parser = argparse.ArgumentParser(description="Move 3-DOF AX-12A robot from home to a target pose.")
    parser.add_argument("--home", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("Q1", "Q2", "Q3"),
                        help="Home joint angles (default: 0 0 0 upright).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", type=float, nargs=3, metavar=("Q1", "Q2", "Q3"),
                       help="Desired joint angles [q1, q2, q3].")
    group.add_argument("--xyz", type=float, nargs=3, metavar=("X", "Y", "Z"),
                       help="Desired end-effector Cartesian position [x, y, z] in meters.")
    parser.add_argument("--deg", action="store_true", help="Interpret --home and --target angles in degrees.")
    parser.add_argument("--duration", type=float, default=2.5, help="Move duration in seconds (default: 2.5).")
    parser.add_argument("--headless", action="store_true", help="Run without viewer and save plot.")
    args = parser.parse_args()

    q_home = np.array(args.home, dtype=np.float64)
    if args.deg:
        q_home = np.deg2rad(q_home)

    if args.target is not None:
        q_target = np.array(args.target, dtype=np.float64)
        if args.deg:
            q_target = np.deg2rad(q_target)
    else:
        model = mujoco.MjModel.from_xml_path(str(SCENE_AX12A_XML))
        xyz = np.array(args.xyz, dtype=np.float64)
        q_target, ik_err = solve_ik_xyz(model, xyz, q_home)
        print(f"IK for xyz={xyz} m -> q={np.round(q_target, 4)} rad (residual={ik_err*1000:.2f} mm)")
        if ik_err > 0.005:
            print("Warning: target may be outside the robot's reachable workspace or joint limits.")

    run_move(q_home, q_target, duration=args.duration, headless=args.headless)


if __name__ == "__main__":
    main()
