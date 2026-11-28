# GRE P1 平面精修

在仓库根目录用 Blender 启动。运行时使用 Blender 自带 Python 与 NumPy。`--report` 指向一个新目录，`--out` 指向该目录之外的新 `.blend`。`--clean` 默认为标注文件旁的 `table_clean.blend`。

```powershell
$blender = $env:HELLO_3DNEKO_BLENDER_EXE
$data = './gre_testdata/table'
& $blender -b --factory-startup "$data/table_noisy_s42_a1.blend" --python-exit-code 1 --python ./gre/cli.py -- --annotations "$data/table_annotations.json" --region tabletop --out ./outputs/table_1h_repaired.blend --report ./outputs/gre_a1
```

报告记录 `accepted`、`rejected`、`no_action`、`unsupported` 或 `solver_failed`。批处理以 `metrics.json` 的 `status` 作为判断依据，`--python-exit-code` 用于 Blender 脚本异常。只有 `accepted` 会写候选 `.blend`，并生成相同机位的 `before.png`、`after.png` 与 `displacement.png`。热图独立归一化，深蓝为零、红色对应 `images.heatmap_peak_m`。

## 数据约定

- 标注以稳定的 `gre_fid` 选择具备 `plane` 语义的可编辑面，以干净基准的指纹校验来源。写入前重新验证输入场景 `source_hash`，补丁使用 `gre_vid`。
- 只在选定面集内做面积加权 SVD 与局部法向投影。两环权重来自局部空间测地距离。共用非选中面或边的顶点硬锁。若输入边缘是完整闭环且全部锁定，且边缘平面与全区拟合一致，则将边缘平面用作目标。其余输入使用全区拟合平面。
- `target_plane.source` 记录实际目标来自 `locked_rim` 或 `area_fit`。对固定平面，`k` 次 smoothstep 投影等价于一次系数为 `1 - (1 - w)^k` 的计算。边缘一致时用 `k=16` 抑制环内波纹，回退时沿用 `k=8`。
- 默认平面性门槛 `λ_min / λ_middle ≤ 0.005`。它是 P1 的木桌初值，可经 `--max-planarity-ratio` 显式调整。P1 用独立 clean 基准检查 50% RMS 改善。位移预算只记录，P2 接入完整 QA。
- 误差用 `gre_vid` 对齐后按对象局部米制计算。输出未经 Apply Transform，报告同时记录 `matrix_world` 与 `unit_scale`。

## 全量验收

在仓库根目录执行。`--numpy-python` 可指向 Blender 安装中 `python/bin` 下的解释器，整个验收无需安装软件包。

```powershell
python ./tests/run_gre_p1.py --blender $env:HELLO_3DNEKO_BLENDER_EXE --numpy-python $env:HELLO_3DNEKO_NUMPY_PYTHON --data ./gre_testdata/table --output-dir ./outputs/p1_reproduced
```

验收重开两份候选，使用 P0 的读数器复核基准误差、全部 UV 指纹及锁点坐标。它也检查单个边缘点翘曲时回退到全区拟合，并检查 3h 拒绝、干净基准无需处理、保护区拒绝、原件覆盖拒绝、共享网格、形态键、修改器、骨骼父级、过期哈希和第二层 UV。

## P2 独立验收与回滚

P2 从原件与候选读取稳定 ID、位置和原始属性，固定基准三角化，独立评估九项门禁。`check` 输出 `metrics.json`、`report.md` 和固定机位几何、线框、UV 棋盘格及位移图片。

```powershell
$blender = $env:HELLO_3DNEKO_BLENDER_EXE
$data = './gre_testdata/table'
& $blender -b --factory-startup --python-exit-code 1 --python ./gre/qa_cli.py -- --workspace ./outputs/p2-work check --source "$data/table_noisy_s42_a1.blend" --candidate ./p1_results/a1/table_repaired.blend --annotations "$data/table_annotations.json" --region tabletop --profile "$data/p2_profile.json" --report ./outputs/p2-a1
```

`--profile` 可选填 `max_displacement_m`、`min_angle_deg`、`uv_p95`、`uv_max`、`area_limit`、`volume_limit`、`pair_limit` 和用于单独报告误差的 `approved_plane`。`--reference` 必须与标注指纹匹配，木桌默认读取标注旁的 `table_clean.blend`。单独使用其他资产时显式传入干净基准。木桌位移预算默认为 `min(1.0h, 0.011D)`，其他资产默认 `min(0.1h, 0.002D)`。注释中的位移上限继续生效。

`commit` 使用同一套输入参数，并需要 `--working-file trial.blend`。它从原件重放通过校验的坐标补丁，只在工作区 `working/` 创建新文件。`needs_review` 须附显式 `--review-note`；硬失败和未完成碰撞检查不可提交。工作区保留 `snapshots/<asset_id>/<source_sha256>.npz`、对应 `.meta.json` 和原生 `.blend`，以及 `audit/` 下的提交与回滚记录。工作区需在支持硬链接的本地文件系统上。

```powershell
& $blender -b --factory-startup --python-exit-code 1 --python ./gre/qa_cli.py -- --workspace ./outputs/p2-work rollback --audit-id '<commit audit ID>' --expected-current-hash '<committed_file_sha256>'
```

```powershell
python ./tests/run_gre_p2.py --blender $env:HELLO_3DNEKO_BLENDER_EXE --numpy-python $env:HELLO_3DNEKO_NUMPY_PYTHON --data ./gre_testdata/table --p1-results ./p1_results --profile ./gre_testdata/table/p2_profile.json --output-dir ./outputs/p2_reproduced
```

## P3 平面族与关系约束

样本与标注在 `gre_testdata/cabinet/`。`--detect-only` 只输出候选平面列表、覆盖率、支持面的 `gre_fid` 和迭代次数。求解只使用 `relations` 与 `junctions` 中明确批准的关系，支持 `parallel`、`perpendicular` 和有符号 `offset`。间距的正方向取第一个平面的法向，其绝对值最大的分量统一朝正。

```powershell
$blender = $env:HELLO_3DNEKO_BLENDER_EXE
$data = './gre_testdata/cabinet'
& $blender -b --factory-startup --python-exit-code 1 --python ./gre/p3_cli.py -- --source "$data/cabinet_tilted.blend" --annotations "$data/cabinet_annotations.json" --out ./outputs/cabinet_repaired.blend --report ./outputs/p3-cabinet --stability-seeds 0 1 7 42 100
```

法线与间距由面积加权检测和 NumPy LM 联合拟合。共享锐边、角点分别求平面交线、交点的最近坐标。保护区零位移，两道过渡环按测地距离衰减。`approved_targets_for_qa` 只供独立评估。`--reference` 默认为标注旁的 `cabinet_clean.blend`。P2 九项通过后才会保存 `.blend`，新文件写出后按同一组稳定 ID 重开核验。

```powershell
python ./tests/run_gre_p3.py --blender $env:HELLO_3DNEKO_BLENDER_EXE --numpy-python $env:HELLO_3DNEKO_NUMPY_PYTHON --data ./gre_testdata/cabinet --output-dir ./outputs/p3_reproduced
```

## P4 特征保护去噪

`gre_testdata/vase/` 包含同拓扑的干净花瓶和 seed 42 噪声花瓶。标注提供浮雕保护版与未标注对照版。局部过滤在基于 `gre_vid`／`gre_fid` 的编辑掩码内工作，保护非流形边、超阈值二面角端点和显式标注特征点，两道过渡环用测地距离加权。

```powershell
python ./tests/run_gre_p4.py --blender $env:HELLO_3DNEKO_BLENDER_EXE --numpy-python $env:HELLO_3DNEKO_NUMPY_PYTHON --data ./gre_testdata/vase --output-dir ./outputs/p4_reproduced
```

`gre/p4_cli.py` 接收 `--source`、`--reference`、`--marked`、`--unmarked`、`--truth` 和 `--output-dir`。`--iters` 控制三种更新方法的共同轮数。`--sigma-s-h`、`--sigma-r-h` 表示双边空间与法向高度尺度相对于局部中位边长 `h` 的倍数，`--bilateral-relaxation` 控制每轮更新幅度。独立评估用参考网格与 `vase_truth.json`，算法只计算噪声几何、邻域与授权掩码。每轮以原始坐标核对累计位移，超预算保留上一个可行轮次。九项 P2 门禁照常运行。方案缺少显式完整特征审阅时，产生改动的候选进入待审。

## P5 批跑与人工计划

```powershell
python -m gre.pipeline batch --blender $env:HELLO_3DNEKO_BLENDER_EXE --output-dir ./gre_p5_reproduced
```

批跑按 `gre_testdata/p5_catalog.json` 对当前目录全部原生输入执行已列明的相容组合。每组使用独立 Blender 工作进程，默认超时 60 秒，可用 `--timeout` 覆盖。工作根目录会生成 `metrics.json`、`results.csv`、`GRE_MVP_总结.md` 与 `audit/`，而冻结计划、逐阶段记录、预览 SVG、坐标差分、日志及通过门禁的候选在 `runs/`。失败索引在 `failures/`，真实演练的工作副本在 `working/`。

人工流程先用 `inspect` 生成待审 E3 计划，再审核并将 `approval_state` 明确填为 `approved`。工作目录内每次运行使用新 `--case-id`。

```powershell
$b = $env:HELLO_3DNEKO_BLENDER_EXE
$common = @('--blender',$b,'--output-dir','./manual','--source','./gre_testdata/cabinet/cabinet_tilted.blend','--annotations','./gre_testdata/cabinet/cabinet_annotations.json','--reference','./gre_testdata/cabinet/cabinet_clean.blend')
python -m gre.pipeline inspect @common --case-id draft --semantic planar_assembly --algorithm plane_family
# 将 manual/runs/draft/plan.json 复制并审阅为 approved.json 后再提交工作副本
python -m gre.pipeline run @common --case-id approved-trial --plan ./approved.json --commit
python -m gre.pipeline rollback --blender $b --output-dir ./manual --case-id restored --audit-id '<commit id>' --expected-current-hash '<commit sha256>'
```

计划用文件 SHA-256 绑定源和证据，长度预算写明 `m`，坐标空间为 `object_local`。有机夹具可传 `--truth` 绑定独立评估依据。`run` 必须携带显式批准计划，`accepted` 只发布验证过的新候选，`--commit` 写隔离原生副本。回滚需要预期当前 SHA-256，恢复后重开核对快照。

```powershell
python ./tests/verify_gre_p5.py --root ./gre_p5_reproduced --data ./gre_testdata
```
