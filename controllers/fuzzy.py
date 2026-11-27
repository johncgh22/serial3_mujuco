"""Sugeno Type-1 Fuzzy Controller ported from `sugenotype1_serial3.fis`.

Preserves the exact 2-input / 1-output, 9-rule zero-order Sugeno FIS from the
MATLAB workspace (`sugenotype1_serial3.fis`):
  - Input 1 (error, normalized to [0, 1] with 0.5 = zero):
      negativo: trimf(-0.4166667, 0.0, 0.4166667)
      cero:     trimf( 0.0833333, 0.5, 0.9166667)
      positivo: trimf( 0.5833333, 1.0, 1.4166667)
  - Input 2 (error rate, normalized to [0, 1] with 0.5 = zero): same 3 trimfs
  - Output (weighted average of 5 constant consequents):
      [-10, -5, 0, +5, +10]
  - 9 rules (product AND, weighted-average defuzzification):
      (N,N)->-10, (Z,N)->-5, (P,N)->0,
      (N,Z)->-5,  (Z,Z)->0,  (P,Z)->+5,
      (N,P)->0,   (Z,P)->+5, (P,P)->+10

Paired with optional analytical gravity compensation and physical per-joint
input/output scaling so it runs directly on the MuJoCo 3-DOF robot.
"""

from __future__ import annotations

import numpy as np

from .base import BaseController, TrajectoryReference
from .dynamics import AnalyticalDynamics


def _trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership function matching MATLAB's `trimf(x, [a, b, c])`."""
    left = np.where(b > a, (x - a) / (b - a), 1.0)
    right = np.where(c > b, (c - x) / (c - b), 1.0)
    return np.maximum(0.0, np.minimum(left, right))


class SugenoType1FIS:
    """Vectorized evaluator for the 9-rule Sugeno FIS in `sugenotype1_serial3.fis`."""

    MF_PARAMS = (
        (-0.416666666666667, 0.0, 0.416666666666667),   # negativo (0)
        ( 0.0833333333333333, 0.5, 0.916666666666667),  # cero     (1)
        ( 0.583333333333333, 1.0, 1.416666666666667),   # positivo (2)
    )
    # 3x3 rule consequent table: rows = input1 (N,Z,P), cols = input2 (N,Z,P)
    RULE_CONSEQUENTS = np.array([
        [-10.0, -5.0,  0.0],
        [ -5.0,  0.0,  5.0],
        [  0.0,  5.0, 10.0],
    ], dtype=np.float64)

    def evaluate(self, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        """Evaluate FIS on arrays u1, u2 in [0, 1], returning output in [-10, 10]."""
        u1_c = np.clip(u1, 0.0, 1.0)
        u2_c = np.clip(u2, 0.0, 1.0)

        mu1 = np.stack([_trimf(u1_c, a, b, c) for (a, b, c) in self.MF_PARAMS], axis=-1)  # (..., 3)
        mu2 = np.stack([_trimf(u2_c, a, b, c) for (a, b, c) in self.MF_PARAMS], axis=-1)  # (..., 3)

        # Product firing strengths for all 9 rules: shape (..., 3, 3)
        weights = mu1[..., :, None] * mu2[..., None, :]
        num = np.sum(weights * self.RULE_CONSEQUENTS, axis=(-2, -1))
        den = np.sum(weights, axis=(-2, -1))
        return np.where(den > 1e-12, num / den, 0.0)


class SugenoFuzzyController(BaseController):
    """Sugeno Type-1 Fuzzy PD + Gravity Compensation controller.

    Maps each joint's (error, error_rate) into [0, 1] centered at 0.5:
        u1 = 0.5 + 0.5 * (e / e_scale)
        u2 = 0.5 + 0.5 * (e_dot / ed_scale)
    evaluates the 9-rule Sugeno FIS (output in [-10, 10]), scales by `tau_gain`,
    and adds analytical gravity compensation G(q).
    """

    name = "Sugeno Fuzzy + G"
    short_name = "Fuzzy+G"
    control_space = "joint"

    def __init__(
        self,
        dynamics: AnalyticalDynamics,
        e_scale: np.ndarray | list[float] = (0.14, 0.12, 0.16),
        ed_scale: np.ndarray | list[float] = (1.6, 1.3, 1.8),
        tau_gain: np.ndarray | list[float] = (0.11, 0.22, 0.06),
        gravity_comp: bool = True,
        torque_limit: float = 1.5,
    ):
        super().__init__(torque_limit=torque_limit)
        self.dyn = dynamics
        self.fis = SugenoType1FIS()
        self.e_scale = np.asarray(e_scale, dtype=np.float64)
        self.ed_scale = np.asarray(ed_scale, dtype=np.float64)
        self.tau_gain = np.asarray(tau_gain, dtype=np.float64)
        self.gravity_comp = bool(gravity_comp)

    def compute(
        self,
        t: float,
        q: np.ndarray,
        qd: np.ndarray,
        ref: TrajectoryReference,
        dt: float,
    ) -> np.ndarray:
        e = ref.q_d - q
        ed = ref.qd_d - qd

        u1 = 0.5 + 0.5 * (e / self.e_scale)
        u2 = 0.5 + 0.5 * (ed / self.ed_scale)
        fis_out = self.fis.evaluate(u1, u2)  # in [-10, 10]

        tau = self.tau_gain * fis_out
        if self.gravity_comp:
            tau = tau + self.dyn.gravity_vector(q)
        return self.clip_torque(tau)
