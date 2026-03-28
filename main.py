from __future__ import annotations

import argparse

from scripts.test import run_test
from scripts.train import train


def main() -> None:
	parser = argparse.ArgumentParser()
	sub = parser.add_subparsers(dest="mode", required=True)

	train_parser = sub.add_parser("train")
	train_parser.add_argument("--env", choices=["selector", "executor"], default="executor")
	train_parser.add_argument("--state_tracking", choices=["reward", "success"], default="reward")

	test_parser = sub.add_parser("test")
	test_parser.add_argument("--checkpoint", type=str, required=True)
	test_parser.add_argument("--steps", type=int, default=1000)

	args = parser.parse_args()
	if args.mode == "train":
		# Convert "success" to "success_rate" for the train function
		state_tracking = "success_rate" if args.state_tracking == "success" else args.state_tracking
		train(args.env, best_state_tracking=state_tracking)
	else:
		run_test(args.checkpoint, args.steps)


if __name__ == "__main__":
	main()