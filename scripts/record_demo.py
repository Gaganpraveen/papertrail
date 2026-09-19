"""Record actual CLI subprocess output in asciicast v2 format; no canned responses."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", default="1706.03762")
    parser.add_argument(
        "--session", help="Show a previously completed session, then record fresh QA."
    )
    parser.add_argument("--output", type=Path, default=Path("examples/demo.cast"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    transcript = []
    with args.output.open("w") as recording:
        recording.write(
            json.dumps(
                {
                    "version": 2,
                    "width": 110,
                    "height": 34,
                    "timestamp": int(time.time()),
                    "title": "PaperTrail: real local briefing and QA",
                    "env": {"TERM": "xterm-256color"},
                }
            )
            + "\n"
        )

        def output(text):
            transcript.append(text)
            recording.write(
                json.dumps([round(time.monotonic() - started, 3), "o", text.replace("\n", "\r\n")])
                + "\n"
            )
            recording.flush()
            print(text, end="", flush=True)

        def run(command):
            output("\n$ papertrail " + " ".join(command) + "\n")
            process = subprocess.Popen(
                [sys.executable, "-m", "papertrail.cli", *command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "COLUMNS": "110", "NO_COLOR": "1"},
            )
            result = []
            for line in process.stdout:
                result.append(line)
                output(line)
            if process.wait() != 0:
                raise SystemExit(
                    "A real demo command failed; inspect the recording. No success was fabricated."
                )
            return "".join(result)

        result = run(["show", args.session] if args.session else ["digest", args.paper])
        match = re.search(r"Saved session: ([a-f0-9]{12})", result)
        if not match:
            raise SystemExit("Could not locate the created session ID.")
        session = match[1]
        questions = [
            "How many GPUs were used, and how long was the big model trained?",
            "What does the paper report about label smoothing?",
            "What was the total training cost in US dollars?",
        ]
        for question in questions:
            run(["ask", session, question])
        run(["export", session, "--output", str(args.output.parent / "attention")])
        output(f"\nCompleted real session: {session}\n")
    args.output.with_suffix(".txt").write_text("".join(transcript))


if __name__ == "__main__":
    main()
