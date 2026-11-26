# SPDX-License-Identifier: Apache-2.0
"""Reproduce P3 detection, recovery, bad relation rejection and disk verification."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def run(command: list[str], root: Path) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("blender", "numpy-python", "data", "output-dir"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scene = args.data.resolve(strict=True)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("--output-dir must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    immutable = {path: digest(path) for path in sorted(scene.glob("cabinet_*")) if path.suffix in {".blend", ".json"}}
    numpy = str(args.numpy_python.resolve(strict=True))
    run([numpy, "-m", "unittest", "discover", "-s", "tests", "-p", "test_gre_p3.py", "-v"], root)
    run([numpy, "-m", "unittest", "discover", "-s", "tests", "-p", "test_gre_p2.py", "-v"], root)
    base = [str(args.blender.resolve(strict=True)), "-b", "--factory-startup", "--python-exit-code", "1", "-t", "4", "--python"]
    module = [*base, "gre/p3_cli.py", "--"]
    input_tilted = str(scene / "cabinet_tilted.blend")
    annotations = str(scene / "cabinet_annotations.json")
    detection = out / "detection"
    run([*module, "--source", input_tilted, "--annotations", annotations, "--report", str(detection), "--detect-only"], root)
    detected = json.loads((detection / "metrics.json").read_text(encoding="utf-8"))
    assert detected["status"] == "candidate" and len(detected["detected"]) == 8
    assert all(row["coverage"] >= 0.60 for row in detected["detected"].values())

    for mode in ("tilted", "noisy"):
        folder = out / mode
        report = folder / "report"
        candidate = folder / "cabinet_repaired.blend"
        flags = ["--stability-seeds", "0", "1", "7", "42", "100"] if mode == "tilted" else []
        run([*module, "--source", str(scene / f"cabinet_{mode}.blend"), "--annotations", annotations,
             "--report", str(report), "--out", str(candidate), *flags], root)
        metrics = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["status"] == "accepted" and metrics["qa"]["status"] == "accepted" and candidate.is_file(), metrics.get("reason")
        assert digest(candidate) == metrics["output_file_sha256"]
        assert all(row["coordinates_exact"] and row["maximum_m"] == 0 for row in metrics["protected"].values())
        assert metrics["junctions"]["line_core_vertices"] > 0 and metrics["junctions"]["corner_core_vertices"] == 1
        assert all(row["deviation_deg"] < 0.1 for row in metrics["relations_after_full"]["relations"])
        if flags:
            assert len(metrics["stability"]["seeds"]) == 5
            worst = max(metrics["stability"]["region_max_pairwise_normal_deg"].values())
            assert worst < 0.05
            assert all(metrics["stability"]["region_identical_support"].values())
            print(f"SEEDS maximum normal deviation {worst:.9f} deg", flush=True)
        for name in ("before.png", "after.png", "wireframe.png", "uv_checker.png", "displacement.png", "report.md", "metrics.json"):
            assert (report / "p2" / name).stat().st_size
        print(f"GOOD {mode}: P2 accepted, exact protected vertices and relations below 0.1 deg", flush=True)

    wrong = out / "wrong_relation"
    reject = wrong / "cabinet_repaired.blend"
    run([*module, "--source", input_tilted, "--annotations", str(scene / "cabinet_wrong_relation.json"),
         "--report", str(wrong / "report"), "--out", str(reject), "--no-images"], root)
    result = json.loads((wrong / "report" / "metrics.json").read_text(encoding="utf-8"))
    assert result["status"] == "rejected" and "displacement" in result["qa"]["hard_failures"] and not reject.exists()
    displacement = result["qa"]["checks"]["displacement"]
    print(f"WRONG relation: rejected at {1000 * displacement['maximum_m']:.6f} mm / {1000 * displacement['budget_m']:.6f} mm budget", flush=True)

    with TemporaryDirectory(prefix="gre-p3-regen-") as repeated:
        run([*base, "gre_testdata/cabinet/build_cabinet.py", "--", "--seed", "42", "--output-dir", repeated], root)
        run([*base, "tests/gre_p3_disk.py", "--", "--data", str(scene), "--repeat", repeated, "--results", str(out)], root)
    for path, sha in immutable.items():
        assert digest(path) == sha
        print(f"INPUT unchanged {path.name} {sha}", flush=True)
    print("P3 ACCEPTANCE: PASS", flush=True)


if __name__ == "__main__":
    main()
