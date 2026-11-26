# SPDX-License-Identifier: Apache-2.0
"""Exercise P2 gates and native Blender commit/rollback without installing packages."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], root: Path, expected: int = 0) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    status = subprocess.run(command, cwd=root, check=False).returncode
    if status != expected:
        raise RuntimeError(f"Expected subprocess status {expected}, got {status}")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--numpy-python", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--p1-results", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data, p1, profile = (path.resolve(strict=True) for path in (args.data, args.p1_results, args.profile))
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("--output-dir must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    workspace = out / "workspace"
    blender = [str(args.blender.resolve(strict=True)), "-b", "--factory-startup"]
    exit_flag = ["--python-exit-code", "1", "-t", "4"]
    qa = [*blender, *exit_flag, "--python", "gre/qa_cli.py", "--", "--workspace", str(workspace)]
    photos = ("before.png", "after.png", "wireframe.png", "uv_checker.png", "displacement.png")
    tracked = set(data.glob("*.blend")) | {p1 / f"a{ratio}" / "table_repaired.blend" for ratio in ("0.5", "1")}
    original = {path: digest(path) for path in sorted(tracked)}
    run([str(args.numpy_python.resolve(strict=True)), "-m", "unittest", "discover", "-s", "tests",
         "-p", "test_gre_p2.py", "-v"], root)

    def job(action: str, source: Path, candidate: Path, folder: Path, *extra: str) -> dict:
        run([*qa, action, "--source", str(source), "--candidate", str(candidate), "--annotations",
             str(data / "table_annotations.json"), "--region", "tabletop", "--profile", str(profile),
             "--report", str(folder), *extra], root)
        return read(folder / "metrics.json")

    def inspect(working: Path, expected: Path) -> None:
        run([*blender, str(working), *exit_flag, "--python", "tests/gre_p2_reopen.py", "--",
             "--p0", str(data), "--expected", str(expected)], root)

    for ratio in ("0.5", "1"):
        source = data / f"table_noisy_s42_a{ratio}.blend"
        candidate = p1 / f"a{ratio}" / "table_repaired.blend"
        folder = out / f"good_a{ratio}"
        result = job("check", source, candidate, folder)
        assert result["status"] in {"accepted", "needs_review"} and not result["hard_failures"], result
        assert all((folder / image).stat().st_size for image in photos)
        target = result["checks"]["target_error"]
        assert target["approved_plane_m"]["after"]["rms"] < target["approved_plane_m"]["before"]["rms"]
        assert target["clean_reference_m"]["after"]["rms"] < target["clean_reference_m"]["before"]["rms"]
        print(f"GOOD a{ratio}: {result['status']} {result['review_reasons']}", flush=True)

    source = data / "table_noisy_s42_a1.blend"
    candidate = p1 / "a1" / "table_repaired.blend"
    filename = "table_1h_trial.blend"
    working = workspace / "working" / filename
    work_args = ("--no-images", "--working-file", filename)
    blocked = job("commit", source, candidate, out / "unapproved_commit", *work_args)
    assert blocked["status"] == "needs_review" and blocked["transaction"]["status"] == "not_committed"
    assert not working.exists()
    posted = job("commit", source, candidate, out / "reviewed_commit", *work_args, "--review-note",
                 "P2 rollback drill on the isolated working copy. UV stretch remains needs_review.")
    receipt = posted["transaction"]
    assert posted["status"] == "needs_review" and receipt["status"] == "committed"
    expected = receipt["committed_file_sha256"]
    assert digest(working) == expected
    inspect(working, candidate)
    shutil.copyfile(working, out / "committed_trial.blend")

    rollback = [*qa, "rollback", "--audit-id", receipt["audit_id"], "--expected-current-hash"]
    run([*rollback, digest(source)], root, expected=1)
    assert digest(working) == expected
    with working.open("ab") as stream:
        stream.write(b"\nP2 simulated out-of-band change\n")
    changed = digest(working)
    run([*rollback, expected], root, expected=1)
    assert digest(working) == changed
    shutil.copyfile(out / "committed_trial.blend", working)
    run([*rollback, expected], root)
    assert digest(working) == digest(source)
    inspect(working, source)
    run([*rollback, expected], root, expected=1)
    assert digest(working) == digest(source)
    print(f"ROLLBACK EXACT {digest(working)}", flush=True)

    edits = {
        "locked_on_disk": ("min((v for v in mesh.vertices if v.co.z == 0.75), key=lambda v: (v.co.x, v.co.y)).co.z += 0.000001", "locked"),
        "uv_on_disk": ("mesh.uv_layers[0].uv[0].vector.x += 0.00001", "integrity"),
    }
    with TemporaryDirectory(prefix="gre-p2-bad-native-") as temporary:
        for name, (edit, expected_gate) in edits.items():
            bad = Path(temporary) / f"{name}.blend"
            expression = ("import bpy\nmesh = bpy.data.objects['Table'].data\n" + edit + "\nmesh.update()\n"
                          f"bpy.ops.wm.save_as_mainfile(filepath={str(bad)!r}, copy=True, check_existing=False)\n")
            run([*blender, str(source), *exit_flag, "--python-expr", expression], root)
            result = job("check", source, bad, out / name, "--no-images")
            assert result["status"] == "rejected" and expected_gate in result["hard_failures"], result
            print(f"BAD NATIVE {name}: {expected_gate} -> rejected", flush=True)

    for path, sha in original.items():
        assert digest(path) == sha
        print(f"Input unchanged {path}: {sha}", flush=True)
    print("P2 ACCEPTANCE: PASS", flush=True)


if __name__ == "__main__":
    main()
