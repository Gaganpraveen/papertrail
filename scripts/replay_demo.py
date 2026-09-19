"""Replay a real asciicast v2 recording using only the Python standard library."""

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, nargs="?", default=Path("examples/demo.cast"))
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Playback speed; does not alter recorded timing."
    )
    parser.add_argument(
        "--max-pause",
        type=float,
        default=3.0,
        help="Cap idle pauses during replay. Original timings remain in the recording.",
    )
    args = parser.parse_args()
    if args.speed <= 0 or args.max_pause < 0:
        parser.error("Speed must be positive and max-pause nonnegative.")
    with args.recording.open() as stream:
        header = json.loads(next(stream))
        print(
            f"Recorded demo: {header['title']} | {args.speed:g}x | idle pauses capped at {args.max_pause:g}s"
        )
        previous = 0.0
        for line in stream:
            timestamp, kind, text = json.loads(line)
            time.sleep(min(max(timestamp - previous, 0) / args.speed, args.max_pause))
            if kind == "o":
                sys.stdout.write(text)
                sys.stdout.flush()
            previous = timestamp


if __name__ == "__main__":
    main()
