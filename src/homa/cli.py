from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict


Handler = Callable[[], int]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="homa",
        description="Command-line helpers for the homa package.",
    )
    parser.add_argument(
        "command",
        help="Name of the command to run, e.g. cache:remove.",
    )
    return parser


def _command_table() -> Dict[str, Handler]:
    return {
        "cache:remove": _handle_cache_remove,
    }


def _handle_cache_remove() -> int:
    root = Path.cwd()
    removed = 0
    errors: list[str] = []

    for candidate in root.rglob("__pycache__"):
        if not candidate.is_dir():
            continue
        try:
            shutil.rmtree(candidate)
            removed += 1
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")

    print(f"Removed {removed} __pycache__ director{'ies' if removed != 1 else 'y'}.")

    if errors:
        print("Failed to remove the following paths:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    commands = _command_table()
    handler = commands.get(args.command)
    if handler is None:
        parser.error(f"Unknown command '{args.command}'.")
        return 2

    return handler()


if __name__ == "__main__":
    sys.exit(main())
