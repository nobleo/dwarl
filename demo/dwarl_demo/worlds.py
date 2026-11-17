"""Compatibility entry point; benchmark geometry lives in dwarl.testing.worlds."""

from dwarl.testing.worlds import WORLDS, build_world, build_worlds, office, warehouse

__all__ = ["WORLDS", "build_world", "build_worlds", "office", "warehouse"]


if __name__ == "__main__":
    import sys

    print(build_worlds(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
