# SPDX-License-Identifier: Apache-2.0
"""Human-readable and machine-readable P2 gate results."""

from pathlib import Path

from gre.snapshot import write_json


def _observations(check: str, result: dict) -> str:
    def millimeters(value: float) -> str:
        return f"{1000 * value:.6f} mm"

    if check == "locked" and "locked_count" in result:
        return f"{result['changed']} changed / {result['locked_count']} locked, max {millimeters(result['maximum_m'])}"
    if check == "displacement" and "budget_m" in result:
        return f"{millimeters(result['maximum_m'])} / {millimeters(result['budget_m'])}"
    if check == "degenerate" and "new_zero" in result:
        return f"{result['new_zero']} new zero-area, {result['new_small']} new small-angle"
    if check == "flip" and "count" in result:
        return f"{result['count']} reversed, {result['uncertain']} uncertain"
    if check == "self_intersection" and "new_pairs" in result:
        return f"{result['baseline_pairs']} baseline pairs, {result['new_pairs']} new"
    if check == "uv_stretch" and "layers" in result:
        return " / ".join(f"{name} p95 {row['p95_log']:.5f}, max {row['max_log']:.5f}"
                          if row['p95_log'] is not None else f"{name} {row['reason']}"
                          for name, row in result["layers"].items())
    if check == "area_volume" and "components" in result:
        rows = []
        for row in result["components"]:
            area = row["area_relative_change"]
            volume = row["volume"].get("relative_change")
            area_text = f"{area:.2%}" if area is not None else "n/a"
            volume_text = f"{volume:.2%}" if volume is not None else "n/a"
            rows.append(f"component {row['component']}, area {area_text}, volume {volume_text}")
        return " / ".join(rows)
    if check == "target_error" and "clean_reference_m" in result:
        values = result['clean_reference_m']
        return f"clean RMS {millimeters(values['before']['rms'])} to {millimeters(values['after']['rms'])}"
    return result.get("reason") or ""


def write_report(folder: Path, metrics: dict) -> None:
    write_json(folder / "metrics.json", metrics)
    lines = ["# GRE P2 QA", "", f"Decision **{metrics['status']}**", "",
             "| Check | Status | Observation |", "| --- | --- | --- |"]
    for name, row in metrics["checks"].items():
        evidence = _observations(name, row).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {name} | {row['status']} | {evidence} |")
    if metrics.get("transaction"):
        lines.extend(("", f"Transaction `{metrics['transaction']['status']}`"))
    lines.extend(("", "[Complete measurements](metrics.json)", ""))
    for name in ("before", "after", "wireframe", "uv_checker", "displacement"):
        if (folder / f"{name}.png").is_file():
            lines.extend((f"![{name}]({name}.png)", ""))
    with (folder / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
