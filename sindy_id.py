#!/usr/bin/env python3
"""Physics-Informed SINDy system identification for the 3-DOF serial robot.

Runs normalized STLSQ over a physics-informed inverse-dynamics candidate library
(trig-coupled inertia, Coriolis/centrifugal, gravity, viscous + Coulomb friction,
plus non-physical distractors) on:
  - MuJoCo `nominal` plant (`scene.xml`)
  - MuJoCo `perturbed` plant (+100 g unknown payload, +25% link2 mass, AX-12A gearbox)
  - Optional MATLAB Simulink logs (`--matlab-dir /path/to/sw2matlab`)

Validates both 1-step torque prediction (train & held-out test split) and
closed-loop forward simulation accuracy, prints the discovered sparse equations,
and saves `renders/sindy_identification.png`.

Usage:
  python sindy_id.py --scenario perturbed
  python sindy_id.py --scenario nominal
  python sindy_id.py --matlab-dir ..
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from controllers.sindy import (
    SINDyDynamics,
    collect_mujoco_excitation,
    evaluate_library,
    fit_sindy_model,
    load_matlab_sindy_data,
)

HERE = Path(__file__).resolve().parent
RENDER_DIR = HERE / "renders"


def evaluate_test_split(
    model: SINDyDynamics,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate SINDy inverse dynamics on held-out test samples."""
    Theta = evaluate_library(model.terms, q, qd, qdd)
    tau_pred = Theta @ model.Xi
    resid = tau - tau_pred
    rmse = np.sqrt(np.mean(resid ** 2, axis=0))
    r2 = 1.0 - np.sum(resid ** 2, axis=0) / np.maximum(np.sum((tau - np.mean(tau, axis=0)) ** 2, axis=0), 1e-12)
    return tau_pred, rmse, r2


def plot_sindy_results(
    t_test: np.ndarray,
    tau_test: np.ndarray,
    tau_pred: np.ndarray,
    r2_test: np.ndarray,
    rmse_test: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)
    labels = ["Joint 1 (yaw)", "Joint 2 (shoulder)", "Joint 3 (elbow)"]
    colors = ["#2563eb", "#ea580c", "#16a34a"]

    for j in range(3):
        axes[j].plot(t_test, tau_test[:, j], color="black", lw=1.6, alpha=0.75, label=f"True τ{j+1}")
        axes[j].plot(
            t_test,
            tau_pred[:, j],
            "--",
            color=colors[j],
            lw=1.4,
            label=f"SINDy τ{j+1} (R²={r2_test[j]:.5f}, RMSE={rmse_test[j]*1000:.2f} mN·m)",
        )
        axes[j].set_ylabel(f"{labels[j]}\nTorque [N·m]")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(loc="upper right", fontsize=8.5)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Physics-Informed SINDy identification on 3-DOF robot data.")
    parser.add_argument(
        "--scenario",
        choices=["nominal", "perturbed"],
        default="perturbed",
        help="MuJoCo plant to identify (default: perturbed).",
    )
    parser.add_argument(
        "--matlab-dir",
        type=Path,
        default=None,
        help="Optional directory containing q_serial3.mat, qd_serial3.mat, qpp_serial3.mat, tau_serial3.mat.",
    )
    parser.add_argument("--threshold", type=float, default=0.025, help="Normalized STLSQ threshold (default: 0.025).")
    parser.add_argument("--noise", type=float, default=0.0, help="Optional encoder noise std [rad] for MuJoCo rollout.")
    parser.add_argument("--duration", type=float, default=14.0, help="MuJoCo excitation duration [s] (default: 14.0).")
    args = parser.parse_args()

    if args.matlab_dir is not None:
        print(f"Loading MATLAB Simulink logs from {args.matlab_dir}...")
        data = load_matlab_sindy_data(args.matlab_dir)
        label = f"MATLAB Simulink Logs ({args.matlab_dir.name})"
    else:
        from benchmark_controllers import build_sim_model

        print(f"Collecting {args.duration:.1f}s persistently exciting rollout from MuJoCo ({args.scenario})...")
        mj_model = build_sim_model(args.scenario)
        data = collect_mujoco_excitation(mj_model, duration=args.duration, subsample=5, noise_std_q=args.noise)
        label = f"MuJoCo {args.scenario.capitalize()} Plant" + (f" (noise={args.noise} rad)" if args.noise > 0 else "")

    # 75% train / 25% held-out test split
    n = len(data["t"])
    split = int(0.75 * n)
    q_tr, qd_tr, qdd_tr, tau_tr = data["q"][:split], data["qd"][:split], data["qdd"][:split], data["tau"][:split]
    t_te, q_te, qd_te, qdd_te, tau_te = (
        data["t"][split:], data["q"][split:], data["qd"][split:], data["qdd"][split:], data["tau"][split:]
    )

    model, train_metrics = fit_sindy_model(q_tr, qd_tr, qdd_tr, tau_tr, threshold=args.threshold)
    tau_pred, rmse_te, r2_te = evaluate_test_split(model, q_te, qd_te, qdd_te, tau_te)

    print(f"\n=== PHYSICS-INFORMED SINDy IDENTIFICATION ({label}) ===")
    print(f"  Samples: {split} train / {n - split} test")
    print(f"  Sparsity: {train_metrics['n_terms_active']} / {train_metrics['n_terms_total']} active terms")
    print(f"  Train R²: [{train_metrics['r2_tau1']:.5f}, {train_metrics['r2_tau2']:.5f}, {train_metrics['r2_tau3']:.5f}]")
    print(f"  Test  R²: [{r2_te[0]:.5f}, {r2_te[1]:.5f}, {r2_te[2]:.5f}]")
    print(f"  Test RMSE: [{rmse_te[0]*1000:.3f}, {rmse_te[1]*1000:.3f}, {rmse_te[2]*1000:.3f}] mN*m")
    print("\n--- Discovered Sparse Inverse-Dynamics Equations ---")
    print(model.format_equations())

    out_name = "sindy_matlab_logs.png" if args.matlab_dir is not None else f"sindy_identification_{args.scenario}.png"
    plot_sindy_results(
        t_te,
        tau_te,
        tau_pred,
        r2_te,
        rmse_te,
        f"Physics-Informed SINDy Held-Out Test Validation — {label}",
        RENDER_DIR / out_name,
    )


if __name__ == "__main__":
    main()
