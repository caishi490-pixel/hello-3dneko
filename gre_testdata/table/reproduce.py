# SPDX-License-Identifier: Apache-2.0
"""Run GRE P0 from a fresh directory using Blender as a separate process."""

import argparse
import hashlib
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def run(blender: Path, script: str, *args: str, blend: Path | None = None) -> None:
    command = [str(blender), "-b", "--factory-startup"]
    if blend is not None:
        command.append(str(blend))
    command.extend(["--python-exit-code", "1", "-t", "4", "--python",
                    str(Path(__file__).resolve().with_name(script))])
    if args:
        command.extend(("--", *args))
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    blender = args.blender.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be a fresh or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    run(blender, "test_noise_field.py")
    run(blender, "build_table.py", "--output-dir", str(output))
    clean = output / "table_clean.blend"
    source_digest = hashlib.sha256(clean.read_bytes()).hexdigest()
    labels = output / "table_annotations.json"
    run(blender, "hash_asset.py", blend=clean)
    with TemporaryDirectory(prefix="gre-p0-repeat-") as repeat:
        for directory in (str(output), repeat):
            for ratio in ("0.5", "1", "3"):
                run(blender, "perturb.py", "--annotations", str(labels), "--output-dir", directory,
                    "--amplitude", ratio, "--wavelength", "0.32", "--seed", "42", blend=clean)
        run(blender, "verify_table.py", "--data-dir", str(output), "--repeat-dir", repeat)
    assert hashlib.sha256(clean.read_bytes()).hexdigest() == source_digest, "Clean .blend changed on disk"
    print(f"Clean file unchanged, SHA-256: {source_digest}")


if __name__ == "__main__":
    main()
