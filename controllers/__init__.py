"""Modular controllers for the 3-DOF Dynamixel AX-12A serial manipulator."""

from .base import BaseController, TrajectoryReference
from .cartesian import CartesianImpedanceController, OperationalSpaceController
from .classical import PDGravityController, PIDController
from .dynamics import AnalyticalDynamics, CAD_PARAMS, RobotParams, params_to_base_theta
from .fuzzy import SugenoFuzzyController, SugenoType1FIS
from .nonlinear import ComputedTorqueController, SlotineLiAdaptiveController
from .robust import SlidingModeController, SuperTwistingSMC
from .trajectories import CartesianCircleTrajectory, SmoothJointTrajectory

__all__ = [
    "AnalyticalDynamics",
    "BaseController",
    "CAD_PARAMS",
    "CartesianCircleTrajectory",
    "CartesianImpedanceController",
    "ComputedTorqueController",
    "OperationalSpaceController",
    "PDGravityController",
    "PIDController",
    "RobotParams",
    "SlidingModeController",
    "SlotineLiAdaptiveController",
    "SmoothJointTrajectory",
    "SugenoFuzzyController",
    "SugenoType1FIS",
    "SuperTwistingSMC",
    "TrajectoryReference",
    "params_to_base_theta",
]
