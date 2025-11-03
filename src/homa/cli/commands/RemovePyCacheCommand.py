import shutil
import sys
from .Command import Command
from pathlib import Path


class RemovePyCacheCommand(Command):
    def __call__(self):
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

        print(
            f"Removed {removed} __pycache__ director{'ies' if removed != 1 else 'y'}."
        )

        if errors:
            print("Failed to remove the following paths:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1

        return 0
