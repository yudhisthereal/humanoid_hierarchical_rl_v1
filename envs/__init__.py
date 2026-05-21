"""Environment package exports.

Keep imports lazy so lightweight consumers (like training scripts for a single
environment) do not pay the cost of importing unrelated MuJoCo-backed modules.
"""

__all__ = ["RobotHierarchicalEnv", "SanityCheckEnv", "Op3ArmBraceEnv", "Op3ArmBraceV2Env"]


def __getattr__(name):
	if name == "RobotHierarchicalEnv":
		from .robot_env import RobotHierarchicalEnv

		return RobotHierarchicalEnv
	if name == "SanityCheckEnv":
		from .sanity_check import SanityCheckEnv

		return SanityCheckEnv
	if name == "Op3ArmBraceEnv":
		from .op3_arm_brace import Op3ArmBraceEnv

		return Op3ArmBraceEnv
	if name == "Op3ArmBraceV2Env":
		from .op3_arm_brace_v2 import Op3ArmBraceEnv as Op3ArmBraceV2Env

		return Op3ArmBraceV2Env
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
