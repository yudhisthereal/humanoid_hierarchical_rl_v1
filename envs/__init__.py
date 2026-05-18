"""Environment package exports.

Keep imports lazy so lightweight consumers (like training scripts for a single
environment) do not pay the cost of importing unrelated MuJoCo-backed modules.
"""

__all__ = ["RobotHierarchicalEnv", "SanityCheckEnv", "Op3ArmBraceEnv"]


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
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
