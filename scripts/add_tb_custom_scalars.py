from __future__ import annotations

import argparse
from pathlib import Path
import sys

from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "report" / "tensorboard" / "goal_executor"


BRACE_TAGS = [
	"Goal BRACE/total_reward/by_iteration",
	"Goal BRACE/r_arm_first/by_iteration",
	"Goal BRACE/r_arm_sync/by_iteration",
	"Goal BRACE/r_knee_timing/by_iteration",
	"Goal BRACE/c_head_impact/by_iteration",
]

ROLL_TAGS = [
	"Goal ROLL/total_reward/by_iteration",
	"Goal ROLL/r_vel/by_iteration",
	"Goal ROLL/r_rot/by_iteration",
	"Goal ROLL/r_tuck/by_iteration",
	"Goal ROLL/c_ctrl/by_iteration",
	"Goal ROLL/c_impact/by_iteration",
]


def _resolve_run_dirs(log_root: Path, run: str) -> list[Path]:
	if run == "all":
		return sorted([p for p in log_root.iterdir() if p.is_dir()])
	if run == "latest":
		runs = sorted([p for p in log_root.iterdir() if p.is_dir()])
		return [runs[-1]] if runs else []
	chosen = log_root / run
	return [chosen] if chosen.is_dir() else []


def _build_layout(goal: str) -> dict[str, dict[str, list[str]]]:
	layout: dict[str, dict[str, list[str]]] = {}

	if goal in ("brace", "both"):
		layout["Goal BRACE"] = {
			"Reward Components": ["Multiline", BRACE_TAGS],
		}

	if goal in ("roll", "both"):
		layout["Goal ROLL"] = {
			"Reward Components": ["Multiline", ROLL_TAGS],
		}

	return layout


def add_custom_scalars(log_root: Path, run: str, goal: str) -> int:
	if not log_root.exists():
		print(f"[ERROR] TensorBoard log root not found: {log_root}")
		return 1

	run_dirs = _resolve_run_dirs(log_root, run)
	if not run_dirs:
		print(f"[ERROR] No runs found for --run={run} in {log_root}")
		return 1

	processed = 0
	layout = _build_layout(goal)
	if not layout:
		print("[ERROR] Empty layout. Check --goal argument.")
		return 1

	for run_dir in run_dirs:
		writer = SummaryWriter(log_dir=str(run_dir))
		writer.add_custom_scalars(layout)
		writer.flush()
		writer.close()

		processed += 1
		print(f"[OK] Added custom scalar layout to run: {run_dir.name}")

	print(f"Done. processed={processed}")
	return 0


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Add TensorBoard Custom Scalars layout so goal reward components appear in one multiline plot "
			"without retraining."
		)
	)
	parser.add_argument(
		"--log-root",
		type=Path,
		default=DEFAULT_LOG_ROOT,
		help=f"Goal executor TensorBoard root (default: {DEFAULT_LOG_ROOT})",
	)
	parser.add_argument(
		"--run",
		type=str,
		default="latest",
		help="Run folder name under log root, or 'latest', or 'all' (default: latest)",
	)
	parser.add_argument(
		"--goal",
		choices=["brace", "roll", "both"],
		default="both",
		help="Which goal chart(s) to create in Custom Scalars (default: both)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	exit_code = add_custom_scalars(log_root=args.log_root, run=args.run, goal=args.goal)
	sys.exit(exit_code)


if __name__ == "__main__":
	main()
