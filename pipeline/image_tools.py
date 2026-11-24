"""Pure-Pillow image helpers: turnaround slicing, breakdown slicing, UV pixel audit, QC grids."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _active_mask_grid(img: Image.Image, threshold: int = 246, step: int = 2) -> tuple[int, int, list[bool]]:
    """Return a downsampled boolean mask where pixels are darker than threshold."""
    rgb = img.convert("RGB")
    sw = max(1, rgb.width // step)
    sh = max(1, rgb.height // step)
    small = rgb.resize((sw, sh), Image.Resampling.NEAREST)
    raw = small.tobytes()
    mask = [
        raw[i] < threshold or raw[i + 1] < threshold or raw[i + 2] < threshold
        for i in range(0, len(raw), 3)
    ]
    return sw, sh, mask


def _pad_square_rgb(pil_im: Image.Image, size: int = 1024, fill_ratio: float = 0.90) -> Image.Image:
    """Center-fit RGB image onto a pure #FFFFFF square canvas."""
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    target = max(1, int(size * fill_ratio))
    scale = min(target / max(1, pil_im.width), target / max(1, pil_im.height))
    nw = max(1, int(round(pil_im.width * scale)))
    nh = max(1, int(round(pil_im.height * scale)))
    resized = pil_im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def slice_turnaround_views(
    src_path: Path,
    out_dir: Path,
    view_names: tuple[str, ...] = ("01_front.png", "02_side.png", "03_back.png"),
    threshold: int = 246,
    min_width: int = 30,
    merge_gap: int = 45,
    size: int = 1024,
    fill_ratio: float = 0.90,
) -> list[Path]:
    """Split a horizontal multi-view turnaround sheet into #FFFFFF square view PNGs."""
    src = Image.open(src_path).convert("RGB")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 2
    sw, sh, mask = _active_mask_grid(src, threshold=threshold, step=step)

    col_active = [any(mask[y * sw + x] for y in range(sh)) for x in range(sw)]
    segs: list[tuple[int, int]] = []
    in_seg = False
    start_x = 0
    for sx, active in enumerate(col_active):
        if active and not in_seg:
            in_seg, start_x = True, sx * step
        elif not active and in_seg:
            in_seg = False
            end_x = sx * step
            if end_x - start_x >= min_width:
                segs.append((start_x, end_x))
    if in_seg and src.width - start_x >= min_width:
        segs.append((start_x, src.width))

    merged: list[tuple[int, int]] = []
    for x0, x1 in segs:
        if merged and x0 - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], x1)
        else:
            merged.append((x0, x1))

    if not merged:
        raise ValueError(f"No foreground views detected in turnaround image: {src_path}")

    saved: list[Path] = []
    for idx, (x0, x1) in enumerate(merged):
        name = view_names[idx] if idx < len(view_names) else f"{idx + 1:02d}_view.png"
        sx0, sx1 = max(0, x0 // step), min(sw, (x1 + step - 1) // step)
        active_ys = [
            sy for sy in range(sh) if any(mask[sy * sw + sx] for sx in range(sx0, sx1))
        ]
        if not active_ys:
            continue
        y0 = max(0, min(active_ys) * step - 12)
        y1 = min(src.height, (max(active_ys) + 1) * step + 12)
        crop = src.crop((max(0, x0 - 12), y0, min(src.width, x1 + 12), y1))
        dst = out_dir / name
        _pad_square_rgb(crop, size=size, fill_ratio=fill_ratio).save(dst, "PNG")
        saved.append(dst)
    return saved


def _dilate_mask(sw: int, sh: int, mask: list[bool], radius: int) -> list[bool]:
    if radius <= 0:
        return list(mask)
    offsets = [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]
    out = [False] * (sw * sh)
    for y in range(sh):
        row_off = y * sw
        for x in range(sw):
            if not mask[row_off + x]:
                continue
            for dx, dy in offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < sw and 0 <= ny < sh:
                    out[ny * sw + nx] = True
    return out


def slice_breakdown_components(
    src_path: Path,
    out_dir: Path,
    prefix: str = "slice",
    labels_hint: list[str] | None = None,
    threshold: int = 247,
    bridge_px: int = 4,
    min_area: int = 1200,
    size: int = 1024,
    fill_ratio: float = 0.88,
) -> list[dict[str, Any]]:
    """Extract isolated garment/prop cutouts from a modular breakdown sheet."""
    src = Image.open(src_path).convert("RGB")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 2
    sw, sh, base_mask = _active_mask_grid(src, threshold=threshold, step=step)
    dilated = _dilate_mask(sw, sh, base_mask, max(1, bridge_px // step))

    labels = [0] * (sw * sh)
    comps: list[dict[str, Any]] = []
    cid = 0
    for y in range(sh):
        for x in range(sw):
            idx = y * sw + x
            if not dilated[idx] or labels[idx] != 0:
                continue
            cid += 1
            q = deque([(x, y)])
            labels[idx] = cid
            x0 = x1 = x
            y0 = y1 = y
            pixels = 0
            while q:
                cx, cy = q.popleft()
                pixels += 1
                x0, x1 = min(x0, cx), max(x1, cx)
                y0, y1 = min(y0, cy), max(y1, cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < sw and 0 <= ny < sh:
                        nidx = ny * sw + nx
                        if dilated[nidx] and labels[nidx] == 0:
                            labels[nidx] = cid
                            q.append((nx, ny))
            area_px = pixels * step * step
            if area_px >= min_area:
                comps.append(
                    {
                        "cid": cid,
                        "sbox": (x0, y0, x1 + 1, y1 + 1),
                        "box": (
                            x0 * step,
                            y0 * step,
                            min(src.width, (x1 + 1) * step),
                            min(src.height, (y1 + 1) * step),
                        ),
                        "area_px": area_px,
                    }
                )

    comps.sort(key=lambda item: (item["box"][0], item["box"][1]))
    outputs: list[dict[str, Any]] = []
    white_full = Image.new("RGB", src.size, (255, 255, 255))
    for idx, comp in enumerate(comps):
        cid_val = comp["cid"]
        mask_bytes = bytes(255 if v == cid_val else 0 for v in labels)
        mask_small = Image.frombytes("L", (sw, sh), mask_bytes)
        mask_full = mask_small.resize(src.size, Image.Resampling.NEAREST)
        x0, y0, x1, y1 = comp["box"]
        pad = 16
        crop_box = (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(src.width, x1 + pad),
            min(src.height, y1 + pad),
        )
        composited = Image.composite(src, white_full, mask_full).crop(crop_box)
        slug = (
            labels_hint[idx]
            if labels_hint and idx < len(labels_hint)
            else f"{prefix}_{idx + 1:02d}"
        )
        out_path = out_dir / f"{idx + 1:02d}_{slug}.png"
        _pad_square_rgb(composited, size=size, fill_ratio=fill_ratio).save(out_path, "PNG")
        outputs.append({"name": out_path.name, "path": str(out_path), "box": comp["box"], "area_px": comp["area_px"]})
    return outputs


def audit_uv_atlas(
    texture_path: Path,
    regions: dict[str, dict[str, Any]],
    out_dir: Path,
    starved_threshold_px: int = 48,
) -> dict[str, Any]:
    """Inspect texture pixel dimensions and export upscaled zoom crops per focal region."""
    tex_path = Path(texture_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(tex_path).convert("RGB")
    w, h = im.size
    results: dict[str, Any] = {
        "texture_path": str(tex_path),
        "atlas_size": [w, h],
        "starved_threshold_px": starved_threshold_px,
        "regions": {},
    }

    for name, spec in regions.items():
        if "pil_box" in spec:
            x0, y0, x1, y1 = [int(v) for v in spec["pil_box"]]
        elif "uv_box" in spec:
            u0, v0, u1, v1 = [float(v) for v in spec["uv_box"]]
            x0 = int(round(u0 * w))
            x1 = int(round(u1 * w))
            y0 = int(round((1.0 - max(v0, v1)) * h))
            y1 = int(round((1.0 - min(v0, v1)) * h))
        elif "pil_center" in spec:
            cx, cy = [int(v) for v in spec["pil_center"]]
            rad = int(spec.get("radius", 24))
            x0, y0, x1, y1 = cx - rad, cy - rad, cx + rad, cy + rad
        else:
            raise ValueError(f"Region '{name}' needs pil_box, uv_box, or pil_center.")

        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(w, max(cx0 + 1, x1)), min(h, max(cy0 + 1, y1))
        crop = im.crop((cx0, cy0, cx1, cy1))
        bw, bh = crop.size
        zoom_size = 512
        zoom = crop.resize((zoom_size, zoom_size), Image.Resampling.NEAREST)
        zoom_path = out_dir / f"zoom_{name}.png"
        zoom.save(zoom_path, "PNG")
        results["regions"][name] = {
            "pil_box": [cx0, cy0, cx1, cy1],
            "pixel_size": [bw, bh],
            "min_dim_px": min(bw, bh),
            "starved": min(bw, bh) < starved_threshold_px,
            "zoom_crop": str(zoom_path),
        }

    audit_json = out_dir / "uv_audit.json"
    audit_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def make_contact_sheet(
    image_paths: list[Path],
    out_path: Path,
    cols: int = 3,
    cell_size: int = 512,
    labels: list[str] | None = None,
) -> Path:
    """Arrange diagnostic images into a clean white-background grid sheet."""
    paths = [Path(p) for p in image_paths if Path(p).is_file()]
    if not paths:
        raise ValueError("No valid images provided to make_contact_sheet.")
    cols = max(1, min(cols, len(paths)))
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_size, rows * cell_size), (248, 248, 252))
    draw = ImageDraw.Draw(sheet)

    for idx, p in enumerate(paths):
        r, c = divmod(idx, cols)
        ox, oy = c * cell_size, r * cell_size
        im = Image.open(p).convert("RGB")
        tile = _pad_square_rgb(im, size=cell_size - 16, fill_ratio=0.95)
        sheet.paste(tile, (ox + 8, oy + 8))
        label = labels[idx] if labels and idx < len(labels) else p.name
        draw.rectangle((ox + 12, oy + 12, ox + 12 + min(cell_size - 24, len(label) * 7 + 14), oy + 32), fill=(24, 24, 34))
        draw.text((ox + 18, oy + 15), label, fill=(245, 245, 255))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "PNG")
    return out_path
