# SPDX-License-Identifier: Apache-2.0
"""CLI orchestration and image registration for the HD retexture pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from .blender_runner import run_blender_script
from .config import DEFAULT_BLENDER_EXE, REPO_ROOT


VIEWS = ("front", "back", "left", "right")
PHOTO_VIEWS = (*VIEWS, "detail")
OBLIQUES = ("front_high", "back_high", "left_low", "right_low")
LINING = ("front_rim", "back_rim", "left_hem", "right_hem")
ALL_VIEWS = (*PHOTO_VIEWS, *OBLIQUES, *LINING)
WORKER = REPO_ROOT / "blender_scripts" / "uv_hd_retex_toolkit.py"
PROMPT = (
    "Retouch only surface texture of this orthographic violet corset render. Freeze canvas, scale, camera, "
    "silhouettes and exact pixel positions of the scrollwork, eyelets, lacing, bow, bell and tassel. "
    "Sharpen existing matte violet fine weave, gilt filigree, satin ribbon and tassel fibers, keeping softly "
    "lit anime collectible shading and original local colors. Remove smear and stains. Keep all existing "
    "parts, with no new stitches, ornament, text, shadows or shapes. Preserve a transparent background "
    "where supported, otherwise use pure white. Keep all panels and spacing if supplied a 2x2 sheet."
)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def white(image):
    image = image.convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")


def prepare_inputs(root):
    target = root / "inputs"
    target.mkdir(exist_ok=True)
    photos = {}
    metadata = read_json(root / "cameras.json")["views"]
    for view in metadata:
        with Image.open(root / "before" / f"{view}.png") as image:
            photos[view] = white(image)
        photos[view].save(target / f"{view}.png")
    tile = photos["front"].width
    grids = ((VIEWS, "view_grid", "preview"), (OBLIQUES, "coverage_grid", "coverage_preview"),
             (LINING, "lining_grid", "lining_preview"))
    for views, name, preview in grids:
        if set(views).issubset(photos):
            grid = Image.new("RGB", (tile * 2, tile * 2), "white")
            for i, view in enumerate(views):
                grid.paste(photos[view], ((i % 2) * tile, (i // 2) * tile))
            grid.save(target / f"{name}.png")
            grid.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
            grid.save(target / f"{preview}.png")
    (target / "prompt.txt").write_text(PROMPT + "\n", encoding="utf-8")


def alpha_mask(image, threshold=128):
    if "A" in image.getbands() and image.getchannel("A").getextrema()[0] < 20:
        return image.getchannel("A").point(lambda value: 255 if value >= threshold else 0)
    red, green, blue = image.convert("RGB").split()
    lowest = ImageChops.darker(red, ImageChops.darker(green, blue))
    return lowest.point(lambda value: 255 if value < 231 else 0)


def gold_mask(image, mask):
    red, _, blue = image.convert("RGB").split()
    yellow = ImageChops.subtract(red, blue).point(lambda value: 255 if value >= 15 else 0)
    return ImageChops.multiply(yellow, mask)


def intersection_ratio(a, b):
    intersection = ImageChops.darker(a, b).histogram()[255]
    union = ImageChops.lighter(a, b).histogram()[255]
    return intersection / max(1, union)


def warp(image, transform, mode=Image.Resampling.NEAREST):
    sx, sy, tx, ty = transform
    return image.transform(image.size, Image.Transform.AFFINE, (sx, 0, tx, 0, sy, ty), mode)


def register_mask(reference, candidate):
    target_box, source_box = reference.getbbox(), candidate.getbbox()
    if not target_box or not source_box:
        return 0.0, (1, 1, 0, 0)
    width, height = reference.size
    # Fit only a bounded global reframing. Never warp individual ornaments.
    scale_x = (source_box[2] - source_box[0]) / (target_box[2] - target_box[0])
    scale_y = (source_box[3] - source_box[1]) / (target_box[3] - target_box[1])
    start = [(1, 1, 0, 0)]
    if 0.93 <= scale_x <= 1.07 and 0.93 <= scale_y <= 1.07:
        start.append((scale_x, scale_y, source_box[0] - target_box[0] * scale_x,
                      source_box[1] - target_box[1] * scale_y))
    mini_size = (192, 192)
    a = reference.resize(mini_size, Image.Resampling.NEAREST)
    b = candidate.resize(mini_size, Image.Resampling.NEAREST)

    def score(t):
        sx, sy, tx, ty = t
        return intersection_ratio(a, warp(b, (sx, sy, tx * 192 / width, ty * 192 / height)))

    transform = max(start, key=score)
    for fraction in (0.012, 0.006, 0.003):
        for index, step in enumerate((fraction, fraction, fraction * width, fraction * height)):
            probes = [transform]
            for sign in (-1, 1):
                probe = list(transform)
                probe[index] += step * sign
                if 0.93 <= probe[0] <= 1.07 and 0.93 <= probe[1] <= 1.07:
                    if abs(probe[2]) <= width * 0.08 and abs(probe[3]) <= height * 0.08:
                        probes.append(tuple(probe))
            transform = max(probes, key=score)
    raw = intersection_ratio(reference, candidate)
    final = intersection_ratio(reference, warp(candidate, transform))
    return (final, transform) if final > raw else (raw, (1, 1, 0, 0))


def align_views(root, grid_path=None, views_dir=None, threshold=0.89, provider="external"):
    if bool(grid_path) == bool(views_dir):
        raise ValueError("Supply exactly one of --grid or --views-dir")
    aligned = root / "aligned"
    aligned.mkdir(exist_ok=True)
    sources = {}
    if grid_path:
        with Image.open(grid_path) as opened:
            grid = opened.copy()
        width, height = grid.size
        if width != height or width % 2:
            raise ValueError("Expected an even square 2x2 sheet")
        tile = width // 2
        sources = {name: grid.crop((i % 2 * tile, i // 2 * tile, (i % 2 + 1) * tile, (i // 2 + 1) * tile))
                   for i, name in enumerate(VIEWS)}
    else:
        calibrated = read_json(root / "cameras.json")["views"]
        for name in calibrated:
            path = views_dir / f"{name}.png"
            if path.is_file():
                with Image.open(path) as image:
                    sources[name] = image.copy()
    views = {}
    for name, image in sources.items():
        if image.width != image.height or image.width < 400:
            raise ValueError(f"{name} needs a square PNG at least 400 px wide")
        before = root / "before" / f"{name}.png"
        with Image.open(before) as render:
            reference = render.convert("RGBA").resize(image.size, Image.Resampling.LANCZOS)
        original_mask = alpha_mask(reference)
        generated_mask = alpha_mask(image)
        iou, transform = register_mask(original_mask, generated_mask)
        raw_iou = intersection_ratio(original_mask, generated_mask)
        image = image.convert("RGBA")
        image.putalpha(image.getchannel("A") if image.getchannel("A").getextrema()[0] < 20 else generated_mask)
        image = warp(image, transform, Image.Resampling.BICUBIC)
        aligned_mask = alpha_mask(image)
        ref_gold = gold_mask(reference, original_mask)
        ai_gold = gold_mask(image, aligned_mask)
        agreement = intersection_ratio(ref_gold, ai_gold)
        has_gold = ref_gold.histogram()[255] > image.width ** 2 * 0.005
        approved = iou >= threshold and (not has_gold or agreement >= 0.46)
        ref_inner = original_mask.filter(ImageFilter.MinFilter(5))
        new_inner = aligned_mask.filter(ImageFilter.MinFilter(5))
        joint = ImageChops.darker(ref_inner, new_inner)
        gold_mismatch = ImageChops.difference(ref_gold, ai_gold).filter(ImageFilter.MaxFilter(5))
        confidence = ImageChops.subtract(joint, gold_mismatch).filter(ImageFilter.GaussianBlur(1.5))
        confidence = ImageChops.multiply(confidence, joint)
        out_image = aligned / f"{name}.png"
        image.save(out_image)
        # Confidence uses a grayscale PNG to preserve a straight, independent opacity channel.
        confidence.save(aligned / f"{name}_confidence.png")
        diagnostic = Image.new("RGB", (image.width * 2, image.height * 2), "white")
        diagnostic.paste(white(reference), (0, 0))
        diagnostic.paste(white(image), (image.width, 0))
        overlay = Image.merge("RGB", (original_mask, aligned_mask, aligned_mask))
        diagnostic.paste(overlay, (0, image.height))
        diff = ImageChops.difference(white(reference), white(image))
        diff = ImageChops.multiply(diff, Image.merge("RGB", (joint, joint, joint)))
        diagnostic.paste(diff, (image.width, image.height))
        diagnostic.save(root / f"alignment_check_{name}.png")
        views[name] = {
            "size": list(image.size), "raw_iou": round(raw_iou, 5), "iou": round(iou, 5),
            "gold_iou": round(agreement, 5), "approved": approved, "affine_output_to_input": transform,
            "foreground_confidence_fraction": round(
                (image.width * image.height - confidence.histogram()[0]) / max(1, original_mask.histogram()[255]), 5),
            "rgb_sha256": digest(out_image),
            "confidence_sha256": digest(aligned / f"{name}_confidence.png"),
            "before_sha256": digest(before),
        }
    report = {"provider": provider, "min_iou": threshold, "camera_sha256": digest(root / "cameras.json"), "views": views}
    write_json(root / "alignment_report.json", report)
    rejected = [name for name in VIEWS if not views.get(name, {}).get("approved")]
    if rejected:
        raise RuntimeError(f"Alignment gate rejected: {', '.join(rejected)}. See alignment_report.json.")
    return report


def comparison(root):
    tile = 840
    header = 34
    sheet = Image.new("RGB", (tile * 2, (tile + header) * 3), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for row, view in enumerate(("front", "back", "detail")):
        for col, stage in enumerate(("before", "after")):
            with Image.open(root / stage / f"{view}.png") as image:
                image = white(image).resize((tile, tile), Image.Resampling.LANCZOS)
            x, y = col * tile, row * (tile + header)
            sheet.paste(image, (x, y + header))
            draw.text((x + 20, y + 9), f"{view.upper()}  /  {stage.upper()}", fill="#241d36", font=font)
    path = root / "before_after_comparison.png"
    sheet.save(path)
    return path


def invoke(root, action, **config):
    root.mkdir(parents=True, exist_ok=True)
    return run_blender_script(DEFAULT_BLENDER_EXE, WORKER, {"out": str(root), "action": action, **config},
                              root / "work" / f"{action}.json", root / "work" / f"{action}.log", timeout=2400)


def execute(args):
    root = args.out.resolve()
    command = args.retex_action
    try:
        if command == "prepare":
            front = args.front
            if not all(math.isfinite(n) for n in front) or math.hypot(*front[:2]) < 1e-6:
                raise ValueError("Front direction requires a finite horizontal vector")
            if not 512 <= args.resolution <= 4096 or not 16 <= args.bleed <= 64:
                raise ValueError("Resolution must be 512–4096 and bleed must be 16–64 pixels")
            if not 0.3 <= args.detail_scale <= 1.0 or not all(math.isfinite(n) for n in args.detail_offset):
                raise ValueError("Invalid detail camera scale or offset")
            common = {"input": str(args.input.resolve(strict=True)), "front": front}
            profile_file = args.region_profile or REPO_ROOT / "templates" / "uv_regions" / (
                "corset.json" if args.corset_regions else "generic.json"
            )
            profile = read_json(profile_file)
            if not isinstance(profile.get("rules"), list) or not isinstance(profile.get("weights"), dict):
                raise ValueError("Region profile requires rules and weights")
            if any(not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0
                   for weight in profile["weights"].values()):
                raise ValueError("Region weights must be positive finite numbers")
            invoke(root, "uv", **common, size=args.size, bleed=args.bleed, region_profile=profile)
            invoke(root, "photo", **common, resolution=args.resolution,
                   detail_offset=args.detail_offset, detail_scale=args.detail_scale, obliques=args.obliques,
                   lining=args.lining)
            prepare_inputs(root)
            result = {"status": "prepared", "preview": str(root / "inputs" / "preview.png")}
        elif command == "align":
            result = align_views(root, args.grid, args.views_dir, args.min_iou, args.provider)
        elif command == "bake":
            alignment = read_json(root / "alignment_report.json")
            if alignment["camera_sha256"] != digest(root / "cameras.json"):
                raise ValueError("Camera calibration changed since alignment")
            approved = [view for view in ALL_VIEWS if alignment["views"].get(view, {}).get("approved")]
            if not set(VIEWS).issubset(approved):
                raise ValueError("All four main views must pass alignment before projection")
            for view in approved:
                if digest(root / "before" / f"{view}.png") != alignment["views"][view]["before_sha256"]:
                    raise ValueError(f"Reference render changed after alignment: {view}")
                for suffix, check in (("", "rgb_sha256"), ("_confidence", "confidence_sha256")):
                    path = root / "aligned" / f"{view}{suffix}.png"
                    if digest(path) != alignment["views"][view][check]:
                        raise ValueError(f"Aligned input changed: {path.name}")
            invoke(root, "bake", input=str(root / "work" / "uv.blend"),
                   high=str(args.high.resolve(strict=True)), views=approved)
            result = {"status": "baked", "comparison": str(comparison(root)),
                      **read_json(root / "projection_report.json")}
        else:
            from .comfy_retexture import generate
            prompt = args.prompt_file.read_text(encoding="utf-8-sig").strip() if args.prompt_file else PROMPT
            if not prompt:
                raise ValueError("A nonempty Comfy Cloud prompt is required")
            result = generate(root, args, prompt)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error))
        return 1


def add_parser(subparsers):
    parser = subparsers.add_parser("uv-hd-retex", help="Prepare, align and rebake fixed-camera garment textures")
    steps = parser.add_subparsers(dest="retex_action", required=True)
    prepare = steps.add_parser("prepare", help="Regenerate UVs and prepare camera-locked inputs")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--size", type=int, choices=(2048, 4096), default=4096)
    prepare.add_argument("--resolution", type=int, default=1536)
    prepare.add_argument("--bleed", type=int, default=16)
    prepare.add_argument("--front", nargs=3, type=float, default=(0, -1, 0))
    prepare.add_argument("--detail-offset", nargs=2, type=float, default=(0.04, 0.13))
    prepare.add_argument("--detail-scale", type=float, default=0.66)
    regions = prepare.add_mutually_exclusive_group()
    regions.add_argument("--region-profile", type=Path, help="JSON UV region rules and weights")
    regions.add_argument("--corset-regions", action="store_true", help="Use the bundled corset region profile")
    prepare.add_argument("--obliques", action="store_true", help="Add four elevated/lowered oblique views")
    prepare.add_argument("--lining", action="store_true", help="Add four steeper lining and hem views")
    align = steps.add_parser("align", help="Check silhouettes, ornaments and projection confidence")
    inputs = align.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--grid", type=Path)
    inputs.add_argument("--views-dir", type=Path)
    align.add_argument("--min-iou", type=float, choices=(0.89, 0.90, 0.92, 0.95, 0.97), default=0.89)
    align.add_argument("--provider", choices=("external", "builtin-imagegen", "comfy-cloud"), default="external")
    bake = steps.add_parser("bake", help="Bake the aligned views and export packed Blender and glTF assets")
    bake.add_argument("--high", type=Path, required=True)
    cloud = steps.add_parser("comfy", help="Submit one confirmed Comfy Cloud grid or single-view generation")
    cloud.add_argument("--view", choices=(*ALL_VIEWS, "view_grid"), default="view_grid")
    cloud.add_argument("--key-file", type=Path)
    cloud.add_argument("--prompt-file", type=Path)
    cloud.add_argument("--confirm-preview", action="store_true")
    cloud.add_argument("--confirmed-cost-cny", type=float)
    cloud.add_argument("--resume", action="store_true")
    for step in (prepare, align, bake, cloud):
        step.add_argument("--out", type=Path, required=True)
        step.set_defaults(func=execute)
