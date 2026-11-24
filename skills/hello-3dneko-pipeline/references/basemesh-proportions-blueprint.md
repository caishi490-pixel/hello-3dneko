# Anime Character Basemesh Proportions & Manual Craft Blueprint

Reference structural blueprint derived via `blender_scripts/extract_and_analyze_basemesh.py` and `blender_scripts/render_proportion_blueprint.py` from a standardized `6.90`-head stylized anime character proportion benchmark (no copyrighted source meshes bundled).

## 1. Macro Character Proportions

Measured on clean anatomical basemesh topology (`Body_SKIN` + `Face`, excluding high hair accessories and garments):

- **Anatomical Height**: `1.6542 m` (sole `Z = 0.000 m`)
- **Head Height (`chin -> crown`)**: `0.2397 m` (`chin Z = 1.4145 m`)
- **Head-to-Body Ratio**: **`6.90` heads**
- **Binding Rest Pose**:
  - Arms: **`90.0°` horizontal T-Pose** (`fingertip_armspan = 1.2845 m`, armspan-to-height ratio **`0.777`**, `upper_to_lower_arm = 1.074`)
  - Legs: **`2.4°` slight vertical splay**

### Vertical Z-Landmark Rhythm (`0%` Sole -> `100%` Crown)

| Landmark | World Z (m) | Height % | Key Anatomical Metric |
| :--- | :---: | :---: | :--- |
| Crown | `1.654` | `100.00%` | Anatomical head top |
| Head Bone (`Head`) | `1.448` | `87.56%` | Neck->Head pivot |
| Chin | `1.415` | `85.51%` | Head occupies top `14.49%` (`1 / 6.90`) |
| Neck Base | `1.375` | `83.10%` | Slender anime neck (`6.4 cm` wide × `8.5 cm` deep) |
| Shoulder Joint | `1.339` | `80.92%` | Bone span `0.175 m`; deltoid-root torso width `0.331 m` |
| Bust Peak | `1.223` | `73.95%` | Front depth `0.205 m` (`width = 0.240 m`) |
| Narrowest Waist | `1.128` | `68.20%` | Narrowest pinch **`0.170 m`** wide × `0.131 m` deep |
| Hips Bone | `0.978` | `59.14%` | Pelvis root |
| Widest Hips | `0.858` | `51.88%` | Widest trochanter flare **`0.318 m`** (`waist/hip = 0.534`) |
| Knee | `0.536` | `32.42%` | `thigh / calf` bone ratio **`0.977 : 1`** |
| Ankle | `0.084` | `5.06%` | Single ankle `5.3 cm` wide × `7.2 cm` deep |
| Sole | `0.000` | `0.00%` | Ground plane contact |

### Lateral Silhouette Craft Rules

- **Front-Flat Stylized Face**: `21.6 cm` wide × `13.3 cm` deep (`1.63 : 1` flat front ratio; head width / torso shoulder width = `0.65`).
- **Hourglass Torso Rhythm**: Shoulder root (`33.1 cm`) : Narrowest waist (`17.0 cm`) : Widest hip (`31.8 cm`) = **`1.04 : 0.53 : 1.00`**.
- **Leg Taper (`single leg`)**: Upper thigh `15.5 cm` -> Knee `7.8 cm` -> Mid calf `7.1 cm` -> Ankle `5.3 cm` (`~3:1` taper).

## 2. Manual Topology, Polygon Budget & Layering Craft

1. **Full-Body Watertight Basemesh + Micro Fingernail Shells**:
   - `Body_SKIN` (`5,870` verts, `10,022` raw tris -> `5,537` recovered faces at `80.93%` quads, `82.67%` valence-4, `0` non-manifold edges) with only the clean `neck_top_rim` boundary meeting `Face`.
   - Separate micro-shells for fingernails keep hand phalanx loops clean.
2. **Recovered Quad Circumferential Loop Ring Standard**:
   - Upper arm / Forearm: **`12` quad edges**
   - Thigh / Calf: **`12` quad edges** (thigh) / **`14` quad edges** (calf)
   - Neck: **`24` quad edges**
   - Waist / Hips: **`30` quad edges** at waist / **`42` quad edges** around pelvis
3. **Asymmetric Polygon Allocation (`Face + Hands = 55.6%` of Basemesh)**:
   - Allocate dense geometry to the face (`8` material slots for skin, oral cavity, sclera, iris, highlight, eyeline, brow, eyelash) and articulated fingers while keeping tight stockings painted in body UV (`0` extra polys).
4. **Clean Humanoid Rigging (`<= 4` Weights/Vertex)**:
   - Strict glTF weight contract: `0` vertices over 4 influences, `0` unnormalized vertices, plus clean secondary chains (`Bust`, `Skirt`, `Hair`).

## 3. Reproducible Extractor Command

```powershell
blender -b --python blender_scripts/extract_and_analyze_basemesh.py -- --input "<basemesh.glb>" --out-dir "<out_dir>"
python blender_scripts/render_proportion_blueprint.py --study-dir "<out_dir>"
```
