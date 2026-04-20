from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "report" / "tensorboard" / "goal_executor"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "report" / "plots" / "goal_executor"
DEFAULT_WINDOW_SIZE = 7
DEFAULT_MAX_EVENT_FILES = 3


BRACE_WEIGHTED_TAGS = {
	"Goal BRACE/total_reward/by_iteration": "total_reward",
	"Goal BRACE/r_arm_first/by_iteration": "r_arm_first",
	"Goal BRACE/r_arm_sync/by_iteration": "r_arm_sync",
	"Goal BRACE/r_knee_timing/by_iteration": "r_knee_timing",
	"Goal BRACE/c_head_impact/by_iteration": "c_head_impact",
}

ROLL_WEIGHTED_TAGS = {
	"Goal ROLL/total_reward/by_iteration": "total_reward",
	"Goal ROLL/r_vel/by_iteration": "r_vel",
	"Goal ROLL/r_rot/by_iteration": "r_rot",
	"Goal ROLL/r_tuck/by_iteration": "r_tuck",
	"Goal ROLL/c_ctrl/by_iteration": "c_ctrl",
	"Goal ROLL/c_impact/by_iteration": "c_impact",
}

BRACE_RAW_TAGS = {
	"Goal BRACE/raw/r_arm_first/by_iteration": "r_arm_first",
	"Goal BRACE/raw/r_arm_sync/by_iteration": "r_arm_sync",
	"Goal BRACE/raw/r_knee_timing/by_iteration": "r_knee_timing",
	"Goal BRACE/raw/c_head_impact/by_iteration": "c_head_impact",
}

ROLL_RAW_TAGS = {
	"Goal ROLL/raw/r_vel/by_iteration": "r_vel",
	"Goal ROLL/raw/r_rot/by_iteration": "r_rot",
	"Goal ROLL/raw/r_tuck/by_iteration": "r_tuck",
	"Goal ROLL/raw/c_ctrl/by_iteration": "c_ctrl",
	"Goal ROLL/raw/c_impact/by_iteration": "c_impact",
}

MEAN_REWARD_TAG_CANDIDATES = [
	"train/current_mean_reward_per_step/by_iteration",
	"train/current_mean_reward/by_iteration",
]

SUCCESS_RATE_TAG = "train/success_rate/by_iteration"


def _resolve_run_dir(log_root: Path, run: str) -> Path:
	if run == "latest":
		runs = sorted([p for p in log_root.iterdir() if p.is_dir()])
		if not runs:
			raise FileNotFoundError(f"No run directories found in: {log_root}")
		return runs[-1]

	run_dir = log_root / run
	if not run_dir.is_dir():
		raise FileNotFoundError(f"Run directory not found: {run_dir}")
	return run_dir


def _resolve_event_files(run_dir: Path, max_event_files: int) -> list[Path]:
	event_files = [p for p in run_dir.glob("events.out.tfevents.*") if p.is_file()]
	event_files.sort(key=lambda path: (path.stat().st_size, path.stat().st_mtime), reverse=True)
	return event_files[:max_event_files] if max_event_files > 0 else event_files


def _load_selected_scalars(
	run_dir: Path,
	target_tags: set[str],
	max_event_files: int,
) -> dict[str, list[tuple[int, float]]]:
	series: dict[str, list[tuple[int, float]]] = {tag: [] for tag in target_tags}
	event_files = _resolve_event_files(run_dir, max_event_files=max_event_files)

	if not event_files:
		return series

	for event_file in event_files:
		loader = RawEventFileLoader(str(event_file))
		for raw_event in loader.Load():
			event = event_pb2.Event.FromString(raw_event)
			if not event.HasField("summary"):
				continue
			step = int(event.step)
			for value in event.summary.value:
				tag = value.tag
				if tag not in series:
					continue
				if value.HasField("simple_value"):
					series[tag].append((step, float(value.simple_value)))

	for tag in series:
		if series[tag]:
			series[tag].sort(key=lambda x: x[0])

	return series


def _moving_average(points: list[tuple[int, float]], window_size: int) -> list[tuple[int, float]]:
	if not points:
		return []
	if window_size <= 1:
		return points[:]

	window: list[float] = []
	window_sum = 0.0
	window_count = 0
	averaged_points: list[tuple[int, float]] = []
	for step, value in points:
		if math.isfinite(value):
			window.append(value)
			window_sum += value
			window_count += 1
		else:
			averaged_points.append((step, value))
			continue

		if len(window) > window_size:
			removed = window.pop(0)
			window_sum -= removed
			window_count -= 1

		averaged_points.append((step, window_sum / window_count if window_count else value))

	return averaged_points


def _running_max(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
	current_max = float("-inf")
	running_points: list[tuple[int, float]] = []
	for step, value in points:
		if math.isfinite(value):
			current_max = value if current_max == float("-inf") else max(current_max, value)
		running_points.append((step, current_max))
	return running_points


def _plot_goal(
	series: dict[str, list[tuple[int, float]]],
	tag_to_label: dict[str, str],
	title: str,
	out_file: Path,
	window_size: int,
) -> bool:
	fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
	has_any = False
	color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

	for idx, (tag, label) in enumerate(tag_to_label.items()):
		events = series.get(tag, [])
		if not events:
			continue
		x = [p[0] for p in events]
		y = [p[1] for p in events]
		color = color_cycle[idx % len(color_cycle)] if color_cycle else None
		ax.plot(x, y, linewidth=1.5, alpha=0.4, color=color)
		moving_avg_points = _moving_average(events, window_size)
		sx = [p[0] for p in moving_avg_points]
		sy = [p[1] for p in moving_avg_points]
		ax.plot(sx, sy, linewidth=2.3, color=color, label=f"{label} smoothed")
		has_any = True

	if not has_any:
		plt.close(fig)
		return False

	ax.set_title(title)
	ax.set_xlabel("iteration")
	ax.set_ylabel("value")
	ax.grid(alpha=0.25)
	ax.legend(loc="best", frameon=True)
	ax.text(0.02, 0.02, f"Transparent lines: raw values. Smoothing: moving average (window={window_size})",
			transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
			bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
	fig.tight_layout()
	out_file.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_file)
	plt.close(fig)
	return True


def _plot_paired_metric(
	series: dict[str, list[tuple[int, float]]],
	current_tag: str,
	title: str,
	out_file: Path,
	window_size: int,
	current_label: str,
	max_label: str,
) -> bool:
	points = series.get(current_tag, [])
	if not points:
		return False

	max_points = _running_max(points)
	fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
	color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
	current_color = color_cycle[0] if color_cycle else None
	max_color = color_cycle[1] if len(color_cycle) > 1 else current_color

	current_x = [p[0] for p in points]
	current_y = [p[1] for p in points]
	current_moving_avg = _moving_average(points, window_size)
	current_sx = [p[0] for p in current_moving_avg]
	current_sy = [p[1] for p in current_moving_avg]

	max_x = [p[0] for p in max_points]
	max_y = [p[1] for p in max_points]
	max_moving_avg = _moving_average(max_points, window_size)
	max_sx = [p[0] for p in max_moving_avg]
	max_sy = [p[1] for p in max_moving_avg]

	ax.plot(current_x, current_y, linewidth=1.5, alpha=0.4, color=current_color)
	ax.plot(
		current_sx,
		current_sy,
		linewidth=2.3,
		color=current_color,
		label=f"{current_label} smoothed",
	)
	ax.plot(max_x, max_y, linewidth=1.5, alpha=0.4, color=max_color)
	ax.plot(
		max_sx,
		max_sy,
		linewidth=2.3,
		color=max_color,
		label=f"{max_label} smoothed",
	)

	ax.set_title(title)
	ax.set_xlabel("iteration")
	ax.set_ylabel("value")
	ax.grid(alpha=0.25)
	ax.legend(loc="best", frameon=True)
	ax.text(0.02, 0.02, f"Transparent lines: raw values. Smoothing: moving average (window={window_size})",
			transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
			bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
	fig.tight_layout()
	out_file.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_file)
	plt.close(fig)
	return True


def _first_available_tag(series: dict[str, list[tuple[int, float]]], candidates: list[str]) -> str | None:
	for tag in candidates:
		if series.get(tag):
			return tag
	return None


def _scaled_points(points: list[tuple[int, float]], scale: float) -> list[tuple[int, float]]:
	return [(step, value * scale) for step, value in points]


def _has_any(series: dict[str, list[tuple[int, float]]], tags: set[str]) -> bool:
	for tag in tags:
		if series.get(tag):
			return True
	return False


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot saved goal reward components with distinct colors and legend (no retraining)."
	)
	parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
	parser.add_argument("--run", type=str, default="latest", help="Run folder name or 'latest'")
	parser.add_argument("--goal", choices=["brace", "roll", "both"], default="both")
	parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
	parser.add_argument(
		"--window-size",
		type=int,
		default=DEFAULT_WINDOW_SIZE,
		help="Trailing moving average window size (default: 7)",
	)
	parser.add_argument(
		"--max-event-files",
		type=int,
		default=DEFAULT_MAX_EVENT_FILES,
		help="How many event files to scan, largest first (default: 3)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	run_dir = _resolve_run_dir(args.log_root, args.run)
	target_tags = (
		set(BRACE_WEIGHTED_TAGS.keys())
		| set(ROLL_WEIGHTED_TAGS.keys())
		| set(BRACE_RAW_TAGS.keys())
		| set(ROLL_RAW_TAGS.keys())
		| set(MEAN_REWARD_TAG_CANDIDATES)
		| {SUCCESS_RATE_TAG}
	)
	series = _load_selected_scalars(run_dir, target_tags, max_event_files=args.max_event_files)

	brace_raw_present = _has_any(series, set(BRACE_RAW_TAGS.keys()))
	roll_raw_present = _has_any(series, set(ROLL_RAW_TAGS.keys()))

	brace_weighted_series = series
	roll_weighted_series = series
	brace_raw_tags_for_plot = BRACE_RAW_TAGS
	roll_raw_tags_for_plot = ROLL_RAW_TAGS

	# Legacy fallback: old runs have only Goal BRACE/* and Goal ROLL/* (unweighted components).
	# Convert legacy components to weighted contributions for weighted plots.
	if not brace_raw_present:
		brace_weighted_series = dict(series)
		if series.get("Goal BRACE/r_arm_sync/by_iteration"):
			brace_weighted_series["Goal BRACE/r_arm_sync/by_iteration"] = _scaled_points(
				series["Goal BRACE/r_arm_sync/by_iteration"], 0.8
			)
		if series.get("Goal BRACE/c_head_impact/by_iteration"):
			brace_weighted_series["Goal BRACE/c_head_impact/by_iteration"] = _scaled_points(
				series["Goal BRACE/c_head_impact/by_iteration"], -1.0
			)
		brace_raw_tags_for_plot = {
			"Goal BRACE/r_arm_first/by_iteration": "r_arm_first",
			"Goal BRACE/r_arm_sync/by_iteration": "r_arm_sync",
			"Goal BRACE/r_knee_timing/by_iteration": "r_knee_timing",
			"Goal BRACE/c_head_impact/by_iteration": "c_head_impact",
		}

	if not roll_raw_present:
		roll_weighted_series = dict(series)
		if series.get("Goal ROLL/r_tuck/by_iteration"):
			roll_weighted_series["Goal ROLL/r_tuck/by_iteration"] = _scaled_points(
				series["Goal ROLL/r_tuck/by_iteration"], 0.5
			)
		if series.get("Goal ROLL/c_ctrl/by_iteration"):
			roll_weighted_series["Goal ROLL/c_ctrl/by_iteration"] = _scaled_points(
				series["Goal ROLL/c_ctrl/by_iteration"], -0.5
			)
		if series.get("Goal ROLL/c_impact/by_iteration"):
			roll_weighted_series["Goal ROLL/c_impact/by_iteration"] = _scaled_points(
				series["Goal ROLL/c_impact/by_iteration"], -1.0
			)
		roll_raw_tags_for_plot = {
			"Goal ROLL/r_vel/by_iteration": "r_vel",
			"Goal ROLL/r_rot/by_iteration": "r_rot",
			"Goal ROLL/r_tuck/by_iteration": "r_tuck",
			"Goal ROLL/c_ctrl/by_iteration": "c_ctrl",
			"Goal ROLL/c_impact/by_iteration": "c_impact",
		}

	out_dir = args.out_root / run_dir.name
	if args.goal in ("brace", "both"):
		ok = _plot_goal(
			series=brace_weighted_series,
			tag_to_label=BRACE_WEIGHTED_TAGS,
			title=f"Goal BRACE weighted reward components ({run_dir.name})",
			out_file=out_dir / "brace_components.png",
			window_size=args.window_size,
		)
		print(f"[{'OK' if ok else 'SKIP'}] {out_dir / 'brace_components.png'}")

		ok = _plot_goal(
			series=series,
			tag_to_label=brace_raw_tags_for_plot,
			title=f"Goal BRACE raw components ({run_dir.name})",
			out_file=out_dir / "brace_components_raw.png",
			window_size=args.window_size,
		)
		print(f"[{'OK' if ok else 'SKIP'}] {out_dir / 'brace_components_raw.png'}")

	if args.goal in ("roll", "both"):
		ok = _plot_goal(
			series=roll_weighted_series,
			tag_to_label=ROLL_WEIGHTED_TAGS,
			title=f"Goal ROLL weighted reward components ({run_dir.name})",
			out_file=out_dir / "roll_components.png",
			window_size=args.window_size,
		)
		print(f"[{'OK' if ok else 'SKIP'}] {out_dir / 'roll_components.png'}")

		ok = _plot_goal(
			series=series,
			tag_to_label=roll_raw_tags_for_plot,
			title=f"Goal ROLL raw components ({run_dir.name})",
			out_file=out_dir / "roll_components_raw.png",
			window_size=args.window_size,
		)
		print(f"[{'OK' if ok else 'SKIP'}] {out_dir / 'roll_components_raw.png'}")

	mean_reward_tag = _first_available_tag(series, MEAN_REWARD_TAG_CANDIDATES)
	if mean_reward_tag is None:
		print("[SKIP] mean_reward_pair.png (no mean reward tag found)")
	else:
		is_per_step = "per_step" in mean_reward_tag
		title = (
			f"Current vs running max mean reward per step ({run_dir.name})"
			if is_per_step
			else f"Current vs running max mean reward ({run_dir.name})"
		)
		label = "current mean reward per step" if is_per_step else "current mean reward"
		max_label = "running max mean reward per step" if is_per_step else "running max mean reward"
		ok = _plot_paired_metric(
			series=series,
			current_tag=mean_reward_tag,
			title=title,
			out_file=out_dir / "mean_reward_pair.png",
			window_size=args.window_size,
			current_label=label,
			max_label=max_label,
		)
		print(f"[{'OK' if ok else 'SKIP'}] {out_dir / 'mean_reward_pair.png'}")

	ok = _plot_paired_metric(
		series=series,
		current_tag=SUCCESS_RATE_TAG,
		title=f"Current vs running max win rate ({run_dir.name})",
		out_file=out_dir / "win_rate_pair.png",
		window_size=args.window_size,
		current_label="current win rate",
		max_label="running max win rate",
	)
	print(f"[{'OK' if ok else 'SKIP'}] {out_dir / 'win_rate_pair.png'}")


if __name__ == "__main__":
	main()
