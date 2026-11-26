# GRE P0 木桌测试集

`table_clean.blend` 是原生 Blender 基准，包含单一网格对象 `Table`。外形尺寸为 1.60 × 0.85 × 0.75 m，桌面厚 65 mm，上下沿为 3 mm 倒角，顶面为 64 × 64 四边面。四条桌腿与桌面组成封闭网格。`TableUV` 生成后固定。

| 文件 | 用途 |
| --- | --- |
| `table_clean.blend` | 干净基准 |
| `table_noisy_s42_a{0.5,1,3}.blend` | 同一扰动形态的三档峰值，seed 42，标称波长 0.32 m |
| `table_annotations.json` | v1 标注。`gre_vid` 为 POINT 域，`gre_fid` 为 FACE 域。`regions[].face_ids` 保存 `gre_fid` |
| `metrics.json` | 从磁盘重新打开模型后计算的验收数据 |
| `build_table.py`、`perturb.py`、`noise_field.py` | 建模、扰动入口与纯 NumPy 数学 |
| `hash_asset.py`、`verify_table.py`、`test_noise_field.py` | 哈希 CLI、场景验收及数学单元测试 |
| `reproduce.py` | 调用 Blender 子进程完成全量复现。外层只用 Python 标准库 |

## 复现

在此目录运行。将 Blender 路径改为本机安装位置，输出目录使用一个新的或空的目录。

```powershell
$blender = 'X:/path/to/blender.exe'
python ./reproduce.py --blender $blender --output-dir ./reproduced
```

单个样本和哈希也可以直接从 Blender 命令行生成。

```powershell
& $blender -b --factory-startup ./table_clean.blend --python-exit-code 1 --python ./perturb.py -- --annotations ./table_annotations.json --output-dir ./my_samples --amplitude 1 --wavelength 0.32 --seed 42
& $blender -b --factory-startup ./table_clean.blend --python-exit-code 1 --python ./hash_asset.py -- --object Table
```

`--amplitude` 以局部中位边长 `h` 为单位，取所有所选面中去重后的边计算中位数。幅度表示绝对法向位移峰值。`--wavelength` 使用局部米制单位。可用 `--face-ids` 传入一组可编辑 `gre_fid` 缩小选择。与非选中面共享的顶点锁定，边界内两个过渡环用局部边长构成的测地距离衰减。这里的资产自包含，输出无需重定位外部资源。

## P1 交接

- 使用 `a1` 作为第一组平面恢复输入。按 `gre_vid` 将候选坐标与 `table_clean.blend` 对齐，并参照 `metrics.json` 的输入 RMS。`a0.5` 和 `a3` 可用于检查对不同输入幅度的表现。
- 读取 `regions` 的角色并锁定所有接触受保护面的顶点。`bevel`、`tabletop_body`、`legs` 均为受保护面。`matrix_world` 为单位矩阵，`unit_scale` 为 1.0。
- 标注 `source_hash` 绑定 clean 基准。匹配 noisy 与标注时，使用 clean 的 `source_hash` 校验标注归属，再用拓扑哈希和稳定 ID 核对 noisy 的对应关系。noisy 的坐标哈希属于其自身场景。
- 本批为程序化数据。真实 AI 木桌样本待提供。
