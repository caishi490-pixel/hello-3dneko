"""Generate an annotated Character Proportion & Craft Blueprint PNG from craft_analysis_report.json and orthogonal renders."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int, bold: bool = False):
    candidates = (
        [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc"]
        if bold
        else [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\segoeui.ttf"]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def build_blueprint(study_dir: Path, out_path: Path) -> Path:
    report = json.loads((study_dir / "craft_analysis_report.json").read_text(encoding="utf-8"))
    props = report["proportions_and_landmarks"]
    craft = report["manual_craft_secrets"]
    hm = props["height_metrics"]
    lp = props["landmark_z_percentages"]
    cs = props["cross_sections"]
    kr = props["key_proportion_ratios"]
    rings = craft["circumferential_quad_loop_rings"]
    budget = craft["basemesh_poly_budget"]
    bones = craft["armature_136_bones_taxonomy"]

    ren_dir = study_dir / "renders"
    img_tex = Image.open(ren_dir / "03_basemesh_textured_front.png").convert("RGB")
    img_wire_f = Image.open(ren_dir / "02_basemesh_clay_wire_front.png").convert("RGB")
    img_wire_s = Image.open(ren_dir / "02_basemesh_clay_wire_side.png").convert("RGB")

    scale = 1.12
    panel_h = round(768 * scale)  # 860
    panel_y0 = 112
    canvas_w, canvas_h = 2500, 1020

    # Horizontal crops keeping full fingertip span in front views and tight profile in side view
    front_crop = (100, 0, 668, 768)
    side_crop = (230, 0, 538, 768)
    front_w = round((front_crop[2] - front_crop[0]) * scale)  # 636
    side_w = round((side_crop[2] - side_crop[0]) * scale)   # 345

    pan_tex = img_tex.crop(front_crop).resize((front_w, panel_h), Image.Resampling.LANCZOS)
    pan_wf = img_wire_f.crop(front_crop).resize((front_w, panel_h), Image.Resampling.LANCZOS)
    pan_ws = img_wire_s.crop(side_crop).resize((side_w, panel_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (241, 244, 249))
    draw = ImageDraw.Draw(canvas)

    # Top header bar
    draw.rectangle([(0, 0), (canvas_w, 84)], fill=(20, 27, 45))
    f_title = _load_font(30, bold=True)
    f_sub = _load_font(17, bold=False)
    f_card_h = _load_font(20, bold=True)
    f_body = _load_font(16, bold=False)
    f_tick = _load_font(15, bold=True)
    f_tick_s = _load_font(14, bold=False)

    draw.text(
        (32, 16),
        f"二次元人型素模拆解与人工精华蓝图  |  真实身高 {hm['anatomical_height_m']:.3f}m · {hm['head_to_body_ratio']:.2f} 头身 · 标准 T-Pose",
        fill=(248, 250, 252),
        font=f_title,
    )
    src_name = Path(report.get("source_glb", "basemesh.glb")).name
    draw.text(
        (32, 54),
        f"基准源：{src_name}（已剥离重复 Ghost 与杂物；保留完整躯干皮肤壳 + 面部材质槽 + 人形骨骼）",
        fill=(148, 163, 184),
        font=f_sub,
    )

    x_p1 = 28
    x_p2 = x_p1 + front_w + 16
    x_p3 = x_p2 + front_w + 16
    x_ruler = x_p3 + side_w + 18
    x_cards = x_ruler + 296

    for px, pw, im, cap in [
        (x_p1, front_w, pan_tex, "正面素模贴图 (袜纹直绘皮肤UV)"),
        (x_p2, front_w, pan_wf, "正面拓扑线框 (肩宽:腰:胯 = 1.04 : 0.53 : 1.00)"),
        (x_p3, side_w, pan_ws, "侧面 S 曲线剖面"),
    ]:
        canvas.paste(im, (px, panel_y0))
        draw.rectangle([(px, panel_y0), (px + pw, panel_y0 + panel_h)], outline=(180, 190, 206), width=2)
        draw.rectangle([(px, panel_y0 - 26), (px + pw, panel_y0)], fill=(30, 41, 59))
        draw.text((px + 10, panel_y0 - 23), cap, fill=(241, 245, 249), font=f_tick)

    # Exact orthographic camera projection inverse used by extract_and_analyze_basemesh._render_study_views
    sole_z = hm["sole_z"]
    crown_z = hm["skin_crown_z"]
    center_z = 0.5 * (sole_z + crown_z)
    span = max(hm["anatomical_height_m"], hm["fingertip_armspan_m"], 0.1)
    ortho_scale = span * 1.18

    def z_to_y(z_val: float) -> int:
        y_raw = 768.0 * (0.5 - (z_val - center_z) / ortho_scale)
        return round(panel_y0 + y_raw * scale)

    landmarks = [
        ("100.0% 头顶", crown_z, f"身高 {hm['anatomical_height_m']:.3f}m", (220, 38, 38)),
        ("85.5% 下巴", hm["chin_z"], f"头高 {hm['head_height_chin_to_crown_m']*100:.1f}cm (1/{hm['head_to_body_ratio']:.1f})", (37, 99, 235)),
        ("80.9% 肩线", cs["torso_shoulder_deltoid_root"]["z"], f"去臂肩宽 {cs['torso_shoulder_deltoid_root']['width_x']*100:.1f}cm", (15, 118, 110)),
        ("74.0% 胸峰", cs["bust_upper_chest"]["z"], f"胸深 {cs['bust_upper_chest']['depth_y']*100:.1f}cm", (217, 119, 6)),
        ("68.2% 最细腰", cs["waist_narrowest"]["z"], f"蜂腰宽 {cs['waist_narrowest']['width_x']*100:.1f}cm", (220, 38, 38)),
        ("51.9% 胯最宽", cs["pelvis_hips_widest"]["z"], f"胯宽 {cs['pelvis_hips_widest']['width_x']*100:.1f}cm", (124, 58, 237)),
        ("32.4% 膝盖", cs["single_knee"]["z"], f"大腿/小腿 ≈ {kr['thigh_to_calf_bone_ratio']:.2f}:1", (37, 99, 235)),
        ("5.1% 脚踝", lp["ankle"] * hm["anatomical_height_m"] / 100.0 + sole_z, f"单踝宽 {cs['single_ankle']['width_x']*100:.1f}cm", (71, 85, 105)),
        ("0.0% 脚底", sole_z, "基准地面 Z=0", (15, 23, 42)),
    ]

    ruler_x_end = x_ruler + 282
    draw.rectangle([(x_ruler - 6, panel_y0), (ruler_x_end, panel_y0 + panel_h)], fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    draw.line([(x_ruler + 4, z_to_y(crown_z)), (x_ruler + 4, z_to_y(sole_z))], fill=(100, 116, 139), width=3)

    for label, zv, detail, col in landmarks:
        yp = z_to_y(zv)
        # Dashed guide across all 3 views
        for sx in range(x_p1, x_p3 + side_w, 12):
            draw.line([(sx, yp), (min(sx + 6, x_p3 + side_w), yp)], fill=col, width=1)
        draw.line([(x_p3 + side_w, yp), (x_ruler + 12, yp)], fill=col, width=2)
        draw.ellipse([(x_ruler, yp - 4), (x_ruler + 8, yp + 4)], fill=col)
        draw.text((x_ruler + 16, yp - 14), label, fill=col, font=f_tick)
        draw.text((x_ruler + 132, yp - 13), detail, fill=(51, 65, 85), font=f_tick_s)

    # Right column: 4 Craft Cards
    cards = [
        (
            "1. 三围与剪影大关系",
            (30, 64, 175),
            [
                f"• 头身比：{hm['head_to_body_ratio']:.2f} 头身 (日系美少女黄金区间)",
                f"• 臂展比：{hm['fingertip_armspan_m']:.2f}m / {hm['anatomical_height_m']:.2f}m = {hm['armspan_to_height_ratio']:.2f} (短紧凑腕臂)",
                f"• 正面宽脸：{cs['head_face_level']['width_x']*100:.1f}宽 × {cs['head_face_level']['depth_y']*100:.1f}深cm (偏扁平)",
                f"• 沙漏腰胯：腰/胯 = {kr['waist_to_hip_width_ratio']:.2f}，肩/胯 = {kr['shoulder_torso_to_hip_width_ratio']:.2f}",
                f"• 细腿渐变：大腿15.5 → 膝7.8 → 踝5.3cm",
            ],
        ),
        (
            "2. 手工四边面环线圈数标准",
            (15, 118, 110),
            [
                f"• 上臂：{rings['left_upper_arm_quad_loop_edges']} 边   前臂：{rings['left_forearm_quad_loop_edges']} 边（标准动画极简管）",
                f"• 大腿：{rings['left_thigh_quad_loop_edges']} 边   小腿：{rings['left_calf_quad_loop_edges']} 边（12~14边防扭曲）",
                f"• 脖子：{rings['neck_mid_quad_loop_edges']} 边（高密度对接头部防扭脖瘪陷）",
                f"• 细腰：{rings['waist_quad_loop_edges']} 边   胯臀：{rings['hips_quad_loop_edges']} 边（{rings['hips_quad_loop_edges']}→{rings['waist_quad_loop_edges']}→12×2 分流）",
                f"• 全身连续闭环仅颈顶88边界线，躯干无掏空",
            ],
        ),
        (
            "3. 面数与材质分层经济学",
            (180, 83, 9),
            [
                f"• 素模总预算：{budget['total_basemesh_raw_tris_polys']:,} 三角面（还原≈8.9k四边面）",
                f"• 头部+手部占 {budget['hands_plus_face_share_of_basemesh_pct']:.1f}%：脸{budget['face_head_share_pct']:.1f}% + 手{100.0*budget['body_skin_regions_raw_polys']['hands_and_fingers']/max(1,budget['total_basemesh_raw_tris_polys']):.1f}%",
                "• 脸部分 8 套材质：肤色+口腔+白眼+虹膜+高光+眉+睫毛+眼线",
                "• 手部含 10 片独立指甲微壳（每片32三角面）",
                "• 贴身长袜与金环直接画在身体UV，零额外建模面数",
            ],
        ),
        (
            "4. 骨骼与物理绑定结构 (136 骨)",
            (109, 40, 217),
            [
                f"• 核心脊柱头部 {bones['breakdown']['core_spine_head']} 骨 + 四肢 {bones['breakdown']['biped_limbs']} 骨 + 30 指骨",
                f"• 次级物理骨：{bones['breakdown']['secondary_bust_physics']} 胸(739顶点权重) + {bones['breakdown']['secondary_skirt_physics']} 裙 + {bones['breakdown']['secondary_hair_physics']} 头发",
                "• 身体每顶点 ≤ 4 骨权重，100% 归一化（无爆皮风险）",
                "• 服装仅在外扩剪影件建模（3件共仅 2,637 三角面）",
                "• 标准 T-Pose 90° 开臂 + 双腿微张 2.4° 方便自动刷权重",
            ],
        ),
    ]

    cy = panel_y0
    card_w = canvas_w - x_cards - 24
    card_h = 202
    for title, hdr_col, bullets in cards:
        draw.rectangle([(x_cards, cy), (x_cards + card_w, cy + card_h)], fill=(255, 255, 255), outline=(203, 213, 225), width=2)
        draw.rectangle([(x_cards, cy), (x_cards + card_w, cy + 36)], fill=hdr_col)
        draw.text((x_cards + 14, cy + 6), title, fill=(255, 255, 255), font=f_card_h)
        by = cy + 48
        for line in bullets:
            draw.text((x_cards + 14, by), line, fill=(30, 41, 59), font=f_body)
            by += 29
        cy += card_h + 16

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study-dir", required=True)
    p.add_argument("--out", default="")
    args = p.parse_args()
    study_dir = Path(args.study_dir)
    out = Path(args.out) if args.out else (study_dir / "renders" / "04_proportion_blueprint.png")
    res = build_blueprint(study_dir, out)
    print("Proportion blueprint saved ->", res)


if __name__ == "__main__":
    main()
