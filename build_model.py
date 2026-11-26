#!/usr/bin/env python3
"""Build a MuJoCo MJCF model from the SolidWorks/Simscape Multibody export.

Reads:
  - ./simscape/Ensamble_SWML.xml
  - ./simscape/Ensamble_SWML_DataFile.m
  - ./meshes/*.obj (converted from SolidWorks .STEP parts in meters)

Writes:
  - ./model.xml        (pure torque-controlled MuJoCo robot asset, +Z up, base flush at z=0)
  - ./scene.xml        (scene wrapper for model.xml)
  - ./model_ax12a.xml  (Dynamixel AX-12A position-controlled robot asset)
  - ./scene_ax12a.xml  (scene wrapper for model_ax12a.xml)
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

OUT_DIR = Path(__file__).resolve().parent
SIMSCAPE_DIR = OUT_DIR / "simscape"
XML_PATH = SIMSCAPE_DIR / "Ensamble_SWML.xml"
DATAFILE_PATH = SIMSCAPE_DIR / "Ensamble_SWML_DataFile.m"

NS = {"sm": "urn:mathworks:SimscapeMultibody:import"}


def make_T(t: np.ndarray, R: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def R_to_mj_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to MuJoCo [w, x, y, z] quaternion."""
    # Ensure exact orthonormality via SVD
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
    xyzw = Rotation.from_matrix(R_clean).as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    if wxyz[0] < 0:
        wxyz = -wxyz
    # Clean tiny numerical noise
    wxyz[np.abs(wxyz) < 1e-12] = 0.0
    wxyz /= np.linalg.norm(wxyz)
    return wxyz


def fmt_vec(v: np.ndarray, digits: int = 8) -> str:
    vals = [0.0 if abs(float(x)) < 1e-12 else float(x) for x in v]
    return " ".join(f"{x:.{digits}g}" for x in vals)


def parse_datafile_rigid_transforms(path: Path) -> dict[int, tuple[str, np.ndarray]]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    rt: dict[int, tuple[str, np.ndarray]] = {}
    for blk in txt.split("%Translation Method"):
        m = re.search(
            r"smiData\.RigidTransform\((\d+)\)\.translation\s*=\s*\[([^\]]+)\].*?"
            r"\.angle\s*=\s*([^;]+);.*?"
            r"\.axis\s*=\s*\[([^\]]+)\].*?"
            r"\.ID\s*=\s*'([^']+)'",
            blk,
            re.S,
        )
        if not m:
            continue
        idx = int(m.group(1))
        t = np.array([float(x) for x in m.group(2).split()], dtype=np.float64) * 0.01  # cm -> m
        ang = float(m.group(3))
        ax = np.array([float(x) for x in m.group(4).split()], dtype=np.float64)
        ident = m.group(5)
        if abs(ang) < 1e-15 or np.linalg.norm(ax) < 1e-15:
            R = np.eye(3, dtype=np.float64)
        else:
            u = ax / np.linalg.norm(ax)
            R = Rotation.from_rotvec(u * ang).as_matrix()
        rt[idx] = (ident, make_T(t, R))
    return rt


def parse_xml_parts_and_instances(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()

    parts: dict[str, dict] = {}
    for p in root.findall(".//sm:Part", NS):
        uid = p.attrib["uid"]
        name = p.attrib["name"]
        safe = name.lower().replace("-", "_")
        mp = p.find("sm:MassProperties", NS)
        mass = float(mp.find("sm:Mass", NS).text)
        com = np.array([float(x) for x in mp.find("sm:CenterOfMass", NS).text.split()], dtype=np.float64)
        # Verified against watertight trimesh moment_inertia: [Ixx, Iyy, Izz, Iyz, Izx, Ixy]
        Ixx, Iyy, Izz, Iyz, Izx, Ixy = [float(x) for x in mp.find("sm:Inertia", NS).text.split()]
        I_mat = np.array(
            [
                [Ixx, Ixy, Izx],
                [Ixy, Iyy, Iyz],
                [Izx, Iyz, Izz],
            ],
            dtype=np.float64,
        )
        vp = p.find("sm:VisualProperties/sm:Diffuse", NS)
        rgba = [float(vp.attrib[c]) for c in ("r", "g", "b", "a")]
        parts[uid] = {
            "name": name,
            "safe": safe,
            "mass": mass,
            "com": com,
            "I": I_mat,
            "rgba": rgba,
        }

    def parse_transform(el: ET.Element) -> np.ndarray:
        tr = el.find("sm:Transform", NS)
        R = np.array([float(x) for x in tr.find("sm:Rotation", NS).text.split()], dtype=np.float64).reshape(3, 3)
        t = np.array([float(x) for x in tr.find("sm:Translation", NS).text.split()], dtype=np.float64)
        return make_T(t, R)

    def collect_leaf_parts(inst_el: ET.Element, T_parent: np.ndarray, prefix: str):
        results = []
        for child in inst_el.findall("sm:Instance", NS):
            cname = child.attrib["name"]
            euid = child.attrib["entityUid"]
            T_child = T_parent @ parse_transform(child)
            full_name = f"{prefix}_{cname}" if prefix else cname
            if euid in parts:
                results.append((full_name, euid, T_child))
            else:
                results.extend(collect_leaf_parts(child, T_child, full_name))
        return results

    top_instances = {}
    for inst in root.find(".//sm:InstanceTree", NS).findall("sm:Instance", NS):
        name = inst.attrib["name"]
        T_world = parse_transform(inst)
        leaves = collect_leaf_parts(inst, np.eye(4), "")
        top_instances[name] = {
            "T_world": T_world,
            "leaves": leaves,
        }

    return parts, top_instances


def combine_inertials(part_entries: list[tuple[np.ndarray, dict]]):
    """Combine multiple solid parts into one (mass, com, quat, diaginertia) in body frame."""
    total_mass = sum(p["mass"] for _, p in part_entries)
    com_total = np.zeros(3, dtype=np.float64)
    for T_part, p in part_entries:
        p_com_body = T_part[:3, :3] @ p["com"] + T_part[:3, 3]
        com_total += p["mass"] * p_com_body
    com_total /= total_mass

    I_total = np.zeros((3, 3), dtype=np.float64)
    for T_part, p in part_entries:
        R = T_part[:3, :3]
        p_com_body = R @ p["com"] + T_part[:3, 3]
        I_rot = R @ p["I"] @ R.T
        d = p_com_body - com_total
        # Parallel Axis Theorem: I_total += I_rot + m * ((d.d) I - d d^T)
        I_shift = p["mass"] * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        I_total += I_rot + I_shift

    I_total = 0.5 * (I_total + I_total.T)
    eigvals, eigvecs = np.linalg.eigh(I_total)
    # Ensure right-handed principal axes
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 0] *= -1
    # Guard against tiny non-positive numerical noise
    eigvals = np.maximum(eigvals, 1e-10)
    quat = R_to_mj_quat(eigvecs)
    return total_mass, com_total, quat, eigvals


def material_for_part(part_safe: str) -> str:
    if part_safe in ("ax_12a_01", "ax_12a_02", "ax_12a_03"):
        return "dynamixel_body"
    if part_safe == "ax_12a_04":
        return "dynamixel_horn"
    if part_safe == "tubo15cm":
        return "aluminum_tube"
    if part_safe == "base1":
        return "base_plate"
    return "bracket_metal"


def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")


def build():
    rt = parse_datafile_rigid_transforms(DATAFILE_PATH)
    parts, top = parse_xml_parts_and_instances(XML_PATH)

    # Extract joint transforms from Simscape DataFile:
    # Joint 1: Base1-1 --[TB1]--> Rz(q1) --[inv(TF1)]--> E1_1-1
    # Joint 2: E1_1-1  --[TB2]--> Rz(q2) --[inv(TF2)]--> E2_2-1
    # Joint 3: E2_2-1  --[TB3]--> Rz(q3) --[inv(TF3)]--> E3_3-1
    TB1, TF1 = rt[1][1], rt[2][1]
    TB2, TF2 = rt[3][1], rt[4][1]
    TB3, TF3 = rt[5][1], rt[6][1]

    # World alignment:
    # In Ensamble_SWML.xml, Base1-1 already has +Z pointing UP along Joint 1's axis!
    # Shift by a pure translation so:
    #   - Joint 1's axis passes through (x=0, y=0)
    #   - The bottom face of base1 sits flush at z = 0.000 m
    T_sw_base1 = top["Base1-1"]["T_world"]
    T_sw_J1 = T_sw_base1 @ TB1

    # Find lowest z of base1 mesh in SW world
    base1_leaf = [T_leaf for _, euid, T_leaf in top["Base1-1"]["leaves"] if parts[euid]["safe"] == "base1"][0]
    T_sw_base1_plate = T_sw_base1 @ base1_leaf
    # base1 local y=0 is bottom surface (y in [0, 0.005])
    z_bottom_sw = (T_sw_base1_plate @ np.array([0.0, 0.0, 0.0, 1.0]))[2]
    origin_shift = np.array([-T_sw_J1[0, 3], -T_sw_J1[1, 3], -z_bottom_sw], dtype=np.float64)
    T_shift = make_T(origin_shift, np.eye(3))

    # Bake +pi/2 zero-angle offset into joint2 and joint3 so q = [0, 0, 0] is upright.
    # (In raw Simscape coordinates, q_raw = [0, pi/2, pi/2] is upright; with this offset,
    # q_raw = q + [0, pi/2, pi/2] while keeping exact joint axes [0, 0, 1] and link inertias.)
    Rz_90 = make_T(np.zeros(3), Rotation.from_euler("z", np.pi / 2).as_matrix())

    # Transform of Base1-1 assembly frame inside MuJoCo root body (pos=0, quat=1 0 0 0)
    T_root_Base1 = T_shift @ T_sw_base1
    # Joint 1 frame inside MuJoCo root body:
    T_root_J1 = T_root_Base1 @ TB1
    # Link 1 body frame IS J1 frame! Inside link1, E1_1-1 assembly frame is inv(TF1).
    T_L1_E1 = np.linalg.inv(TF1)
    # Joint 2 frame inside link1 (rotated +pi/2 around Z so q2=0 points straight up):
    T_L1_J2 = T_L1_E1 @ TB2 @ Rz_90
    # Link 2 body frame IS J2 frame! Inside link2, E2_2-1 assembly frame is inv(TF2).
    T_L2_E2 = np.linalg.inv(TF2)
    # Joint 3 frame inside link2 (rotated +pi/2 around Z so q3=0 aligns straight up):
    T_L2_J3 = T_L2_E2 @ TB3 @ Rz_90
    # Link 3 body frame IS J3 frame! Inside link3, E3_3-1 assembly frame is inv(TF3).
    T_L3_E3 = np.linalg.inv(TF3)

    print("Key kinematic transforms (MuJoCo +Z up, base flush at z=0):")
    print(f"  J1 pos in root:  {np.round(T_root_J1[:3, 3], 6)}, z_axis: {np.round(T_root_J1[:3, 2], 6)}")
    print(f"  J2 pos in link1: {np.round(T_L1_J2[:3, 3], 6)}, z_axis: {np.round(T_L1_J2[:3, 2], 6)}")
    print(f"  J3 pos in link2: {np.round(T_L2_J3[:3, 3], 6)}, z_axis: {np.round(T_L2_J3[:3, 2], 6)}")

    # Compute end-effector site at the distal tip of Link 3 (BaseTubo-2 of E3_3-1)
    e3_base_tubo2 = [
        T_leaf for lname, euid, T_leaf in top["E3_3-1"]["leaves"] if "BaseTubo-2" in lname
    ][0]
    T_L3_ee = T_L3_E3 @ e3_base_tubo2
    print(f"  EE pos in link3: {np.round(T_L3_ee[:3, 3], 6)}")

    # Collect per-body part lists in each body's local frame
    bodies_spec = [
        ("serial_3dof_robot", T_root_Base1, top["Base1-1"]["leaves"], "0.20 0.75 0.35 0.45"),
        ("link1", T_L1_E1, top["E1_1-1"]["leaves"], "0.25 0.55 0.90 0.45"),
        ("link2", T_L2_E2, top["E2_2-1"]["leaves"], "0.95 0.55 0.15 0.45"),
        ("link3", T_L3_E3, top["E3_3-1"]["leaves"], "0.85 0.25 0.65 0.45"),
    ]

    body_data = {}
    for bname, T_body_asm, leaves, col_rgba in bodies_spec:
        part_entries = []
        geom_entries = []
        for lname, euid, T_asm_part in leaves:
            p = parts[euid]
            T_body_part = T_body_asm @ T_asm_part
            part_entries.append((T_body_part, p))
            geom_entries.append((sanitize_name(f"{bname}_{lname}"), p["safe"], T_body_part))
        mass, com, iquat, diag = combine_inertials(part_entries)
        body_data[bname] = {
            "mass": mass,
            "com": com,
            "iquat": iquat,
            "diag": diag,
            "geoms": geom_entries,
            "col_rgba": col_rgba,
        }
        print(
            f"  Body {bname:18s}: mass={mass*1000:6.2f} g, "
            f"com={np.round(com, 5)}, diagI={np.round(diag*1e6, 3)}e-6 kg*m^2"
        )

    total_mass = sum(d["mass"] for d in body_data.values())
    print(f"  Total robot mass: {total_mass*1000:.2f} g ({total_mass:.5f} kg)")

    # Build pure model.xml
    lines: list[str] = []
    lines.append('<mujoco model="serial_3dof_robot">')
    lines.append('  <compiler angle="radian" meshdir="meshes" autolimits="true"/>')
    lines.append("")
    lines.append("  <default>")
    lines.append('    <default class="serial_3dof_robot">')
    lines.append('      <joint type="hinge" axis="0 0 1" damping="0" armature="0"/>')
    lines.append('      <default class="visual">')
    lines.append('        <geom group="2" type="mesh" contype="0" conaffinity="0" density="0" mass="0"/>')
    lines.append("      </default>")
    lines.append('      <default class="collision">')
    lines.append(
        '        <geom group="3" type="mesh" contype="1" conaffinity="1" condim="4"'
        ' friction="0.8 0.02 0.005" solref="0.005 1" solimp="0.95 0.99 0.001"'
        ' rgba="0.20 0.75 0.35 0.45"/>'
    )
    lines.append("      </default>")
    lines.append("    </default>")
    lines.append("  </default>")
    lines.append("")
    lines.append("  <asset>")
    lines.append('    <material name="dynamixel_body" rgba="0.20 0.21 0.23 1" metallic="0.1" roughness="0.55"/>')
    lines.append('    <material name="dynamixel_horn" rgba="0.72 0.74 0.76 1" metallic="0.25" roughness="0.45"/>')
    lines.append('    <material name="bracket_metal" rgba="0.72 0.76 0.86 1" metallic="0.35" roughness="0.38"/>')
    lines.append('    <material name="aluminum_tube" rgba="0.62 0.66 0.74 1" metallic="0.55" roughness="0.32"/>')
    lines.append('    <material name="base_plate" rgba="0.68 0.72 0.82 1" metallic="0.40" roughness="0.40"/>')
    for p in sorted(parts.values(), key=lambda x: x["safe"]):
        s = p["safe"]
        lines.append(f'    <mesh name="{s}_vis" file="{s}.obj"/>')
        lines.append(f'    <mesh name="{s}_col" file="{s}.obj" maxhullvert="64"/>')
    lines.append("  </asset>")
    lines.append("")

    def emit_body_geoms(bname: str, indent: str):
        bd = body_data[bname]
        lines.append(
            f'{indent}<inertial pos="{fmt_vec(bd["com"])}" quat="{fmt_vec(bd["iquat"])}"'
            f' mass="{bd["mass"]:.10g}" diaginertia="{fmt_vec(bd["diag"], 10)}"/>'
        )
        for gname, psafe, T_part in bd["geoms"]:
            pos = fmt_vec(T_part[:3, 3])
            quat = fmt_vec(R_to_mj_quat(T_part[:3, :3]))
            mat = material_for_part(psafe)
            lines.append(
                f'{indent}<geom name="{gname}_vis" class="visual" mesh="{psafe}_vis"'
                f' material="{mat}" pos="{pos}" quat="{quat}"/>'
            )
            lines.append(
                f'{indent}<geom name="{gname}_col" class="collision" mesh="{psafe}_col"'
                f' pos="{pos}" quat="{quat}" rgba="{bd["col_rgba"]}"/>'
            )

    lines.append("  <worldbody>")
    lines.append('    <body name="serial_3dof_robot" childclass="serial_3dof_robot">')
    emit_body_geoms("serial_3dof_robot", "      ")

    # Link 1
    lines.append("")
    lines.append(
        f'      <body name="link1" pos="{fmt_vec(T_root_J1[:3, 3])}"'
        f' quat="{fmt_vec(R_to_mj_quat(T_root_J1[:3, :3]))}">'
    )
    lines.append('        <joint name="joint1" axis="0 0 1" limited="false"/>')
    emit_body_geoms("link1", "        ")

    # Link 2
    lines.append("")
    lines.append(
        f'        <body name="link2" pos="{fmt_vec(T_L1_J2[:3, 3])}"'
        f' quat="{fmt_vec(R_to_mj_quat(T_L1_J2[:3, :3]))}">'
    )
    lines.append('          <joint name="joint2" axis="0 0 1" range="-1.5707963 1.5707963"/>')
    emit_body_geoms("link2", "          ")

    # Link 3
    lines.append("")
    lines.append(
        f'          <body name="link3" pos="{fmt_vec(T_L2_J3[:3, 3])}"'
        f' quat="{fmt_vec(R_to_mj_quat(T_L2_J3[:3, :3]))}">'
    )
    lines.append('            <joint name="joint3" axis="0 0 1" range="-2.0943951 2.0943951"/>')
    emit_body_geoms("link3", "            ")
    lines.append(
        f'            <site name="end_effector" group="4" type="sphere" size="0.008"'
        f' pos="{fmt_vec(T_L3_ee[:3, 3])}" quat="{fmt_vec(R_to_mj_quat(T_L3_ee[:3, :3]))}"'
        f' rgba="0.95 0.25 0.25 0.50"/>'
    )
    lines.append("          </body>")
    lines.append("        </body>")
    lines.append("      </body>")
    lines.append("    </body>")
    lines.append("  </worldbody>")
    lines.append("")
    lines.append("  <contact>")
    lines.append('    <exclude name="base_link1_exclude" body1="serial_3dof_robot" body2="link1"/>')
    lines.append('    <exclude name="link1_link2_exclude" body1="link1" body2="link2"/>')
    lines.append('    <exclude name="link2_link3_exclude" body1="link2" body2="link3"/>')
    lines.append("  </contact>")
    lines.append("")
    lines.append("  <actuator>")
    lines.append('    <!-- Dynamixel AX-12A stall torque ~1.5 N*m -->')
    lines.append('    <motor name="motor_joint1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-1.5 1.5"/>')
    lines.append('    <motor name="motor_joint2" joint="joint2" gear="1" ctrllimited="true" ctrlrange="-1.5 1.5"/>')
    lines.append('    <motor name="motor_joint3" joint="joint3" gear="1" ctrllimited="true" ctrlrange="-1.5 1.5"/>')
    lines.append("  </actuator>")
    lines.append("")
    lines.append("  <sensor>")
    lines.append('    <jointpos name="q1" joint="joint1"/>')
    lines.append('    <jointpos name="q2" joint="joint2"/>')
    lines.append('    <jointpos name="q3" joint="joint3"/>')
    lines.append('    <jointvel name="qd1" joint="joint1"/>')
    lines.append('    <jointvel name="qd2" joint="joint2"/>')
    lines.append('    <jointvel name="qd3" joint="joint3"/>')
    lines.append('    <actuatorfrc name="tau1" actuator="motor_joint1"/>')
    lines.append('    <actuatorfrc name="tau2" actuator="motor_joint2"/>')
    lines.append('    <actuatorfrc name="tau3" actuator="motor_joint3"/>')
    lines.append('    <framepos name="ee_pos" objtype="site" objname="end_effector"/>')
    lines.append("  </sensor>")
    lines.append("")
    lines.append("  <keyframe>")
    lines.append('    <key name="upright" qpos="0 0 0"/>')
    lines.append('    <key name="elbow_90" qpos="0 0 -1.5707963268"/>')
    lines.append('    <key name="cad_assembly" qpos="0.01211714 0.00780357 1.08561127"/>')
    lines.append("  </keyframe>")
    lines.append("</mujoco>")
    lines.append("")

    model_path = OUT_DIR / "model.xml"
    model_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {model_path}")

    # Build convenient scene.xml wrapper for interactive viewing / rendering
    scene_xml = """<mujoco model="serial_3dof_robot_scene">
  <option timestep="0.001" gravity="0 0 -9.80665" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <statistic center="0 0 0.25" extent="0.6"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="135" elevation="-20" offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.88 0.91 0.95" rgb2="0.98 0.98 0.99" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.82 0.85 0.88" rgb2="0.92 0.94 0.96" markrgb="0.7 0.7 0.75" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="6 6" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="0.5 -0.6 1.2" dir="-0.4 0.5 -1" directional="true" castshadow="true"/>
    <light pos="-0.6 0.6 0.9" dir="0.5 -0.5 -1" directional="true" castshadow="false" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" condim="4" friction="0.8 0.02 0.005"/>
  </worldbody>

  <include file="model.xml"/>
</mujoco>
"""
    scene_path = OUT_DIR / "scene.xml"
    scene_path.write_text(scene_xml, encoding="utf-8")
    print(f"Wrote {scene_path}")

    # Build AX-12A position-controlled variant (model_ax12a.xml + scene_ax12a.xml)
    # Calibrated to Dynamixel AX-12A (12V) specs:
    #   - Joint mode range: +/-150 deg (joint1), +/-90 deg (joint2), +/-120 deg (joint3)
    #   - Stall torque: +/-1.5 N*m (forcerange="-1.5 1.5")
    #   - No-load speed: ~59 RPM (6.18 rad/s) via joint damping=0.24 N*m*s/rad & frictionloss=0.03 N*m
    #   - Reflected 254:1 rotor inertia: armature=0.003 kg*m^2
    #   - Position servo gains: kp=10 N*m/rad, kv=0.25 N*m*s/rad (critically damped, <1 deg gravity sag)
    ax_lines = []
    for line in lines:
        if line == '<mujoco model="serial_3dof_robot">':
            ax_lines.append('<mujoco model="serial_3dof_robot_ax12a">')
        elif 'damping="0" armature="0"' in line:
            ax_lines.append('      <joint type="hinge" axis="0 0 1" damping="0.24" armature="0.003" frictionloss="0.03"/>')
        elif '<joint name="joint1"' in line:
            ax_lines.append('        <joint name="joint1" axis="0 0 1" range="-2.6179939 2.6179939"/>')
        elif '<motor name="motor_joint1"' in line:
            ax_lines.append(
                '    <position name="pos_joint1" joint="joint1" kp="10" kv="0.25"'
                ' ctrllimited="true" ctrlrange="-2.6179939 2.6179939" forcelimited="true" forcerange="-1.5 1.5"/>'
            )
        elif '<motor name="motor_joint2"' in line:
            ax_lines.append(
                '    <position name="pos_joint2" joint="joint2" kp="10" kv="0.25"'
                ' ctrllimited="true" ctrlrange="-1.5707963 1.5707963" forcelimited="true" forcerange="-1.5 1.5"/>'
            )
        elif '<motor name="motor_joint3"' in line:
            ax_lines.append(
                '    <position name="pos_joint3" joint="joint3" kp="10" kv="0.25"'
                ' ctrllimited="true" ctrlrange="-2.0943951 2.0943951" forcelimited="true" forcerange="-1.5 1.5"/>'
            )
        elif 'actuator="motor_joint1"' in line:
            ax_lines.append('    <actuatorfrc name="tau1" actuator="pos_joint1"/>')
        elif 'actuator="motor_joint2"' in line:
            ax_lines.append('    <actuatorfrc name="tau2" actuator="pos_joint2"/>')
        elif 'actuator="motor_joint3"' in line:
            ax_lines.append('    <actuatorfrc name="tau3" actuator="pos_joint3"/>')
        elif "<!-- Dynamixel AX-12A stall torque" in line:
            ax_lines.append("    <!-- Dynamixel AX-12A Joint Mode position servos (12V: 1.5 N*m stall, ~59 RPM max) -->")
        else:
            ax_lines.append(line)

    ax_model_path = OUT_DIR / "model_ax12a.xml"
    ax_model_path.write_text("\n".join(ax_lines), encoding="utf-8")
    print(f"Wrote {ax_model_path}")

    ax_scene_xml = scene_xml.replace(
        'model="serial_3dof_robot_scene"', 'model="serial_3dof_robot_ax12a_scene"'
    ).replace('<include file="model.xml"/>', '<include file="model_ax12a.xml"/>')
    ax_scene_path = OUT_DIR / "scene_ax12a.xml"
    ax_scene_path.write_text(ax_scene_xml, encoding="utf-8")
    print(f"Wrote {ax_scene_path}")


if __name__ == "__main__":
    build()
