from __future__ import annotations

import argparse
import ast
from typing import Sequence

from scripts.test import run_test
from scripts.train import train


def _parse_push_forces(value: str) -> list[float]:
	"""Parse --push-f values like '[100,75,30]' or '[75]' into a non-empty float list."""
	try:
		parsed = ast.literal_eval(value)
	except (SyntaxError, ValueError) as exc:
		raise argparse.ArgumentTypeError(
			"Invalid --push-f format. Use a Python-style list, e.g. --push-f='[100,75,30]'"
		) from exc

	if isinstance(parsed, (int, float)):
		values: Sequence[float] = [float(parsed)]
	elif isinstance(parsed, (list, tuple)):
		if len(parsed) == 0:
			raise argparse.ArgumentTypeError("--push-f must not be empty.")
		try:
			values = [float(v) for v in parsed]
		except (TypeError, ValueError) as exc:
			raise argparse.ArgumentTypeError("--push-f must contain numeric values only.") from exc
	else:
		raise argparse.ArgumentTypeError(
			"Invalid --push-f value. Expected a number or list of numbers, e.g. '[75]'."
		)

	if any(v < 0.0 for v in values):
		raise argparse.ArgumentTypeError("--push-f values must be non-negative.")
	return list(values)


def main() -> None:
	parser = argparse.ArgumentParser()
	sub = parser.add_subparsers(dest="mode", required=True)

	train_parser = sub.add_parser("train")
	train_parser.add_argument("--env", choices=["selector", "executor", "sanity_check"], default="executor")
	train_parser.add_argument("--state_tracking", choices=["reward", "success"], default="reward")
	train_parser.add_argument(
		"--push-f",
		type=_parse_push_forces,
		default=None,
		help="Initial push-force choices, e.g. --push-f='[100,75,30]' or --push-f='[75]'.",
	)

	test_parser = sub.add_parser("test")
	test_parser.add_argument("--checkpoint", type=str, required=True)
	test_parser.add_argument("--steps", type=int, default=1000)
	test_parser.add_argument(
		"--push-f",
		type=_parse_push_forces,
		default=None,
		help="Initial push-force choices, e.g. --push-f='[100,75,30]' or --push-f='[75]'.",
	)

	args = parser.parse_args()
	if args.mode == "train":
		# Convert "success" to "success_rate" for the train function
		state_tracking = "success_rate" if args.state_tracking == "success" else args.state_tracking
		train(args.env, best_state_tracking=state_tracking, push_force_choices=args.push_f)
	else:
		run_test(args.checkpoint, args.steps, push_force_choices=args.push_f)


if __name__ == "__main__":
	main()