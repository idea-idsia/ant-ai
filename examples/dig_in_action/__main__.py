"""Run the demo: `uv run python -m examples.dig_in_action`.

Opens a browser on a run that is already going. The two buttons in the header
restart it with and without `DigToHeal`, which is the comparison worth watching:
the same cast, the same schedule, the same clock, and a different answer.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn

from examples.dig_in_action.server import create_app, default_factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-heal",
        action="store_true",
        help="Start on the unhealed condition: routing only, no detectors.",
    )
    parser.add_argument(
        "--think",
        type=float,
        default=1.2,
        help="Seconds a scripted turn takes. The figure's x axis is wall clock, "
        "so this is how fast the diagram draws itself.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Partial results the coordinator settles for out of five. Raise it "
        "to 5 and the flaw stops being a premature submit and becomes a stall.",
    )
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument(
        "--rsp",
        action="store_true",
        help="Put the Repeated Subproblem detector back. It reports the whole "
        "cohort on a broadcast assignment, and its advisory then wakes every "
        "counter — worth seeing once, and why it is off by default.",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    app = create_app(
        default_factory(
            think=args.think,
            patience=args.patience,
            max_rounds=args.max_rounds,
            repeated_subproblem=args.rsp,
        ),
        heal=not args.no_heal,
    )

    url = f"http://{args.host}:{args.port}/"
    print(f"DIG in action — {url}")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
