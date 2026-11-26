# SPDX-License-Identifier: Apache-2.0
"""Full P1 acceptance from fresh outputs using the native Blender CLI."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def run(command: list[str], cwd: Path) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--numpy-python", required=True, type=Path, help="A Python interpreter with NumPy")
    parser.add_argument("--data", required=True, type=Path, help="P0 fixture directory with hash_asset.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    blender = str(args.blender.resolve(strict=True))
    data = args.data.resolve(strict=True)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("--output-dir must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    baselines = {file: hashlib.sha256(file.read_bytes()).hexdigest() for file in data.glob("*.blend")}
    run([str(args.numpy_python.resolve(strict=True)), "-m", "unittest", "discover", "-s", "tests",
         "-p", "test_gre_math.py", "-v"], root)
    base = [blender, "-b", "--factory-startup"]
    cli = ["--python-exit-code", "1", "-t", "4", "--python", "gre/cli.py", "--",
           "--annotations", str(data / "table_annotations.json")]
    cases = [("a0.5", "table_noisy_s42_a0.5.blend", "tabletop", "accepted", ""),
             ("a1", "table_noisy_s42_a1.blend", "tabletop", "accepted", ""),
             ("a3", "table_noisy_s42_a3.blend", "tabletop", "rejected", "not_planar"),
             ("clean", "table_clean.blend", "tabletop", "no_action", "already_at_reference"),
             ("protected_region", "table_noisy_s42_a1.blend", "bevel", "unsupported", "Select an editable")]
    for name, source, region, status, reason in cases:
        work = out / name
        target = work / "table_repaired.blend"
        report = work / "report"
        run([*base, str(data / source), *cli, "--region", region, "--out", str(target), "--report", str(report)], root)
        metrics = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["status"] == status and metrics.get("reason", "").startswith(reason), metrics
        assert target.is_file() == (status == "accepted"), target
        if status == "accepted":
            assert metrics["target_plane"]["source"] == "locked_rim" and metrics["locked_rim_anchor"]["vertices"] == 256
            assert metrics["reference_after_m"]["rms"] < 0.00005
            assert metrics["rim_face_normal_deviation_deg"]["after"]["p95"] < 0.5
            for filename in metrics["images"]["before"], metrics["images"]["after"], metrics["images"]["heatmap"]:
                assert (report / filename).stat().st_size > 0
        print(f"CASE {name}: {status}", flush=True)

    # Refuse an in-place write even when the caller passes the source path explicitly.
    source = data / "table_noisy_s42_a1.blend"
    report = out / "in_place_guard" / "report"
    run([*base, str(source), *cli, "--region", "tabletop", "--out", str(source), "--report", str(report)], root)
    assert json.loads((report / "metrics.json").read_text(encoding="utf-8"))["status"] == "unsupported"

    # Warp a single locked rim vertex on a scratch copy to exercise production fallback.
    with TemporaryDirectory(prefix="gre-p1-warped-rim-") as scratch:
        scratch = Path(scratch)
        warped = scratch / "warped_rim.blend"
        expression = (
            "import bpy\n"
            "mesh = bpy.data.objects['Table'].data\n"
            "corner = min((v for v in mesh.vertices if v.co.z == 0.75), key=lambda v: (v.co.x, v.co.y))\n"
            "corner.co.z += 0.004\n"
            "mesh.update()\n"
            f"bpy.ops.wm.save_as_mainfile(filepath={str(warped)!r}, copy=True, check_existing=False)\n"
        )
        run([*base, str(source), "--python-exit-code", "1", "--python-expr", expression], root)
        target, report = scratch / "fallback_repaired.blend", scratch / "report"
        run([*base, str(warped), *cli, "--region", "tabletop", "--out", str(target), "--report", str(report)], root)
        metrics = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["status"] == "accepted" and metrics["target_plane"]["source"] == "area_fit" and target.is_file()
        assert metrics["projection_passes"] == 8 and metrics["locked_coordinates_exact"]
        print("CASE warped_rim: area-weighted fallback with exact locked coordinates", flush=True)

    run([*base, "--python-exit-code", "1", "-t", "4", "--python", "tests/gre_p1_blender_checks.py", "--",
         "--data", str(data), "--results", str(out)], root)
    for file, digest in baselines.items():
        assert hashlib.sha256(file.read_bytes()).hexdigest() == digest, file
        print(f"Source SHA-256 unchanged: {file.name} {digest}", flush=True)
    print("P1 ACCEPTANCE: PASS", flush=True)


if __name__ == "__main__":
    main()
