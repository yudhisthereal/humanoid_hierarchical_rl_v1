from __future__ import annotations

import argparse

from scripts.test import run_test
from scripts.train import train


def main() -> None:
	parser = argparse.ArgumentParser()
	sub = parser.add_subparsers(dest="mode", required=True)

	sub.add_parser("train")

	test_parser = sub.add_parser("test")
	test_parser.add_argument("--checkpoint", type=str, required=True)
	test_parser.add_argument("--steps", type=int, default=1000)

	args = parser.parse_args()
	if args.mode == "train":
		train()
	else:
		run_test(args.checkpoint, args.steps)


if __name__ == "__main__":
	main()