"""Inspect a checkpoint's legal turn plan without changing the live web bot."""
import argparse
from dataclasses import asdict
import json

import torch

from .mcts import MCTS
from .model import TorchEvaluator, load_checkpoint
from .rules import e
from .train import device_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    position = parser.add_mutually_exclusive_group()
    position.add_argument("--fen")
    position.add_argument("--state")
    parser.add_argument("--time-ms", type=float, default=2400)
    parser.add_argument("--simulations", type=int, default=100000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = device_name(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    board = e.BaseBoard.deserialize(args.state) if args.state else e.BaseBoard(args.fen or e.STARTING_FEN)
    result = MCTS(TorchEvaluator(model, device), simulations=args.simulations).plan_turn(board, args.time_ms)
    print(json.dumps({"prototype": True, "value_semantics": "uncalibrated W-minus-L, current-player perspective",
                      **asdict(result)}, indent=2))


if __name__ == "__main__":
    main()
