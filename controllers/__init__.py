"""Modular controllers for the 3-DOF Dynamixel AX-12A serial manipulator."""

from .base import BaseController, TrajectoryReference
from .cartesian import CartesianImpedanceController, OperationalSpaceController
from .classical import PDGravityController, PIDController
from .dynamics import AnalyticalDynamics, CAD_PARAMS, RobotParams, params_to_base_theta
from .fuzzy import SugenoFuzzyController, SugenoType1FIS
from .mpc import LTVMPCController
from .nonlinear import ComputedTorqueController, SlotineLiAdaptiveController
from .robust import SlidingModeController, SuperTwistingSMC
from .sindy import (
    SINDyDynamics,
    collect_mujoco_excitation,
    fit_sindy_model,
    load_matlab_sindy_data,
)
from .trajectories import CartesianCircleTrajectory, SmoothJointTrajectory

__all__ = [
    "AnalyticalDynamics",
    "BaseController",
    "CAD_PARAMS",
    "CartesianCircleTrajectory",
    "CartesianImpedanceController",
    "ComputedTorqueController",
    "LTVMPCController",
    "OperationalSpaceController",
    "PDGravityController",
    "PIDController",
    "RobotParams",
    "SINDyDynamics",
    "SlidingModeController",
    "SlotineLiAdaptiveController",
    "SmoothJointTrajectory",
    "SugenoFuzzyController",
    "SugenoType1FIS",
    "SuperTwistingSMC",
    "TrajectoryReference",
    "collect_mujoco_excitation",
    "fit_sindy_model",
    "load_matlab_sindy_data",
    "params_to_base_theta",
]
