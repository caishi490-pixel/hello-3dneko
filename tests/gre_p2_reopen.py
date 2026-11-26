# SPDX-License-Identifier: GPL-3.0-or-later
"""Reopen a P2 working .blend and compare it with the expected native geometry."""

import argparse
import sys
from pathlib import Path

import bpy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    sys.path.insert(0, str(args.p0.resolve()))
    from hash_asset import coordinates, fingerprint, mesh_object

    working = mesh_object("Table")
    data, hashes = coordinates(working.data), fingerprint(working)
    assert bpy.ops.wm.open_mainfile(filepath=str(args.expected.resolve()), load_ui=False, use_scripts=False) == {"FINISHED"}
    expected = mesh_object("Table")
    assert data.tobytes() == coordinates(expected.data).tobytes()
    assert hashes == fingerprint(expected)
    print(f"Disk reopen {args.expected.name}: every stored coordinate and P0 fingerprint match", flush=True)


if __name__ == "__main__":
    main()
