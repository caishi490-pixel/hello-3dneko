# UV baking and HD retexture

## Shared-atlas PBR baking

Run `python run_pipeline.py uv-bake --high "<high>" --low "<low>" --out "<work>"`. The worker fits each part's islands into disjoint padded atlas cells, merges a temporary target, and bakes Diffuse, Normal, Roughness, Metallic, and AO once each. Original parts keep their geometry and share the atlas material. The report records the temporary face ranges, bake count, and device.

Review `ray_valid_hit_pct`, `cross_layer_mis_hit_pct`, `slope_decode_angle_error_deg`, and `tangent_sign_flipped_pct` independently. Correct measured wrong-layer hits through cage and ray settings. Use vector inversion only for a confirmed tangent-sign issue with no wrong-layer hits. A non-calibrated slope reports null. Zero ray distance uses the Blender unlimited-ray setting independently of cage extrusion.

For HD retexture, `prepare --region-profile "<profile.json>"` accepts ordered rules and positive weights. See `templates/uv_regions/generic.json` for the default and `corset.json` for an opt-in garment profile.

## Visual polish and focal UVs

Audit texel density and matched-camera closeups before editing. If the eye, a motif, or another focal island lacks pixels, give it a dedicated material and UV region. Keep welded geometry and face-corner UV seams intact during edits. Use the reference render and approved 2D design to fix framing and feature placement during image editing, then inspect alpha and landmark alignment before reprojection. Use the texture channels or Toon shader to control metallic accents and specular glare. Verify the result by reloading the baked glTF and rendering the same cameras.

## HD reprojection

当服装或配件的低模拓扑已经达标（例如四边面率 > 99%），但原有贴图发糊、需要**严格保留原几何轮廓与布线**并换上一套高清纹理时，按本流程执行多视角拍摄 → 锁轮廓透明重绘 → 遮挡安全反投烘焙。

## 五步核心工程纪律

1. **留住原 UV，再展开一套干净的新 UV**
   - 优先把接缝藏到侧面和内侧；若仅按侧缝展开导致饰品、金纹或细绳挤在一起出现折叠，改用智能展开并将少量打架的面单独拆开，把可检测重叠压到 `0`。
   - 必看 UV 布局图与像素密度报告，确认小装饰分到了足够的纹素（Texel Density）。
2. **先拍清楚底图，再做透明重绘**
   - 从同一模型固定相机拍摄正、背、左、右及需要的近照（柔光无硬影），保存完整相机参数。
   - 亲眼检查拍摄底图：确认没有裁断流苏、没有搞反正背面，再生成透明 PNG 重绘图。正、背面优先单张出图，避免网格拼图瓜分有效像素。
3. **让重绘只改表面细节，严锁几何轮廓**
   - 提示词明确锁住轮廓、机位、装饰位置（金纹、抽绳、铃铛等）。
   - 用透明图的 Alpha 通道与底图比对轮廓，用半透明叠图检查细边和花纹；轮廓通过后仍有局部小错位时，单独遮罩挡掉错位像素。
4. **只把安全可见的区域反投回去**
   - 某块布料被前景装饰遮挡时，自动交由能看清该区域的另一机位接管。
   - 用法线夹角与视线遮挡检查筛选安全像素；所有机位都采不到的死角从原高模/原材质回退补齐，最后做 **16 px UV 边缘扩色（Dilation）**防黑缝。
   - 实战经验：仅 4 个水平机位时安全覆盖率约 `49.6%`，补齐上下斜视角（`--obliques`）和内衬机位（`--lining`）后可提升至 `88.4%+`。先查覆盖率热力图哪里漏，再把相机往哪里加。
5. **交真正能直接打开验收的资产**
   - 同机位输出正面、背面、局部、侧面与内衬对比图；重新打开导出的 `.blend` 与 `.glb` 核对网格与打包贴图完整性，并跑离线测试。原网格自带的几何缺口需在验收报告中明确指出位置。

## CLI 实战命令

```powershell
# 1. 准备新 UV 与固定机位柔光底图（按需加 --obliques / --lining / --corset-regions）
python -B run_pipeline.py uv-hd-retex prepare --input low.blend --out projects/retexture_demo --size 4096

# 2. 将审图通过的正/背/左/右透明重绘图放入 projects/retexture_demo/generated/selected/
#    命名：front.png / back.png / left.png / right.png（斜视角与内衬按 manifest 对应命名）
python -B run_pipeline.py uv-hd-retex align --out projects/retexture_demo --views-dir projects/retexture_demo/generated/selected

# 3. 执行法线与遮挡安全的多视角反投 + 原高模死角补齐 + 16px 扩边烘焙
python -B run_pipeline.py uv-hd-retex bake --high high.glb --out projects/retexture_demo
```

> [!NOTE]
> 高模与低模必须在同一坐标系下；若调用外部付费生图接口，务必先人工核对输入底图、落盘任务号，且失败不自动重试。
