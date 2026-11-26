#!/usr/bin/env python3
"""Validate the MuJoCo 3-DOF robot model (kinematics, dynamics, energy, and visuals)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT_IMG_DIR = HERE / "renders"
OUT_IMG_DIR.mkdir(exist_ok=True)


def render_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    filename: str,
    show_col: bool = False,
    lookat=(0.0, 0.0, 0.28),
    distance: float = 0.88,
    azimuth: float = 135.0,
    elevation: float = -18.0,
):
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

    opt = mujoco.MjvOption()
    opt.geomgroup[2] = 1
    opt.geomgroup[3] = 1 if show_col else 0
    opt.sitegroup[4] = 1

    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=cam, scene_option=opt)
        img = renderer.render()
    Image.fromarray(img).save(OUT_IMG_DIR / filename)
    ee = data.site("end_effector").xpos
    print(f"Saved {filename:28s} | q={np.round(qpos, 3)} | ee_pos={np.round(ee, 4)} m")


def verify_physics(model: mujoco.MjModel, data: mujoco.MjData):
    print("\n=== PHYSICS & DYNAMICS VERIFICATION ===")
    # 1. Total mass check vs SolidWorks XML
    expected_mass = 0.207406546568
    actual_mass = float(np.sum(model.body_mass))
    assert abs(actual_mass - expected_mass) < 1e-9, f"Mass mismatch: {actual_mass} vs {expected_mass}"
    print(f"[PASS] Total mass matches SolidWorks XML exactly: {actual_mass*1000:.3f} g")

    # 2. Upright singularity check (static gravity torque == 0 when q = [0, 0, 0])
    data.qpos[:] = [0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    print(f"[PASS] Upright (0, 0, 0) static gravity torque: {np.round(data.qfrc_inverse, 6)} N*m")
    assert np.allclose(data.qfrc_inverse, 0.0, atol=1e-4)

    # 3. Forward -> Inverse dynamics round-trip (contact-free pose)
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(20):
        data.qpos[:] = [rng.uniform(-2, 2), rng.uniform(-1.0, 1.0), rng.uniform(-1.2, 1.2)]
        data.qvel[:] = rng.uniform(-2, 2, size=3)
        tau_cmd = rng.uniform(-0.5, 0.5, size=3)
        data.ctrl[:] = tau_cmd
        mujoco.mj_forward(model, data)
        assert data.ncon == 0, f"Unexpected contact at test pose {data.qpos}"
        mujoco.mj_inverse(model, data)
        max_err = max(max_err, float(np.max(np.abs(data.qfrc_inverse - tau_cmd))))
    assert max_err < 1e-10, f"Forward/inverse mismatch: {max_err}"
    print(f"[PASS] Forward <-> Inverse dynamics round-trip (20 poses): max error = {max_err:.2e} N*m")

    # 4. Energy conservation with RK4 integrator (zero damping, zero torque, no limits/contacts)
    rk4_model = mujoco.MjModel.from_xml_path(str(HERE / "model.xml"))
    rk4_model.jnt_limited[:] = 0
    rk4_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    rk4_model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    rk4_model.opt.timestep = 0.0005
    rk4_model.opt.gravity[:] = [0, 0, -9.80665]
    rk4_model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    rk4_data = mujoco.MjData(rk4_model)
    rk4_data.qpos[:] = [0.2, -0.35, -0.55]
    rk4_data.qvel[:] = [0.5, -0.3, 0.4]
    mujoco.mj_forward(rk4_model, rk4_data)
    e0 = rk4_data.energy[0] + rk4_data.energy[1]
    e_min, e_max = e0, e0
    for _ in range(2000):  # 1.0 s
        mujoco.mj_step(rk4_model, rk4_data)
        e = rk4_data.energy[0] + rk4_data.energy[1]
        e_min, e_max = min(e_min, e), max(e_max, e)
    drift = max(abs(e_max - e0), abs(e_min - e0))
    print(f"[PASS] RK4 1.0s free-swing energy conservation: E0={e0:.6f} J, max drift={drift:.2e} J")
    assert drift < 1e-4

    # 5. Check link lengths & end-effector height at upright pose q = [0, 0, 0]
    data.qpos[:] = [0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    j1_pos = data.body("link1").xpos.copy()
    j2_pos = data.body("link2").xpos.copy()
    j3_pos = data.body("link3").xpos.copy()
    ee_pos = data.site("end_effector").xpos.copy()
    print(f"[INFO] Upright kinematics (q = [0, 0, 0]):")
    print(f"       J1 = {np.round(j1_pos, 5)} m")
    print(f"       J2 = {np.round(j2_pos, 5)} m  (|J2-J1| = {np.linalg.norm(j2_pos-j1_pos):.5f} m)")
    print(f"       J3 = {np.round(j3_pos, 5)} m  (|J3-J2| = {np.linalg.norm(j3_pos-j2_pos):.5f} m)")
    print(f"       EE = {np.round(ee_pos, 5)} m  (|EE-J3| = {np.linalg.norm(ee_pos-j3_pos):.5f} m)")


def main():
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    data = mujoco.MjData(model)
    print(f"Loaded scene.xml: nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}, ngeom={model.ngeom}")

    verify_physics(model, data)

    print("\n=== RENDERING VERIFICATION POSES ===")
    render_pose(model, data, np.array([0.0, 0.0, 0.0]), "pose_upright.png")
    render_pose(model, data, np.array([0.0, 0.0, 0.0]), "pose_upright_collision.png", show_col=True)
    render_pose(model, data, np.array([0.0, 0.0, -np.pi / 2]), "pose_elbow_90.png")
    render_pose(model, data, np.array([0.01211714, 0.00780357, 1.08561127]), "pose_cad_assembly.png")


if __name__ == "__main__":
    main()
