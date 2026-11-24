# Hello 3DNeko · Agent 与开发者协同规范

本仓库用于构建可复现、可质检、可审计的 AI 3D 自动化生产工作流，遵循 **Apache-2.0（主协调层） + GPL-3.0-or-later（Blender 子进程脚本层）** 跨进程解耦规范。

## 分工架构

- **Orchestrator & Geometry Pipeline（主协调与几何/验收模块）**：
  - 维护阶段流转（`Stage 0 -> 5`）与原子化 `manifest.json` 幂等状态锁；
  - 通过无头 Blender CLI 子进程执行 `Quad` 去重、空间焊接防 UV 裂口与闭环自主重拓扑 (`retopo`) 及 6 维独立几何验证（注：重拓扑之后流程在本公开版中暂不开源）；
  - 对输出图与指标执行独立质量关口核验。
- **2D & Texture Worker（2D 三视图 / 修美 / 贴图优化 Worker）**：
  - **Stage 1**：按 `templates/codex_stage1_2d_prompt.md` 生成纯白 `#FFFFFF` 底 `40°` A-Pose 三视图（锁左右一致）及模块化拆件图；
  - **Stage 4**：按 `templates/codex_stage4_uv_polish_prompt.md` 结合 UV 密度审计 (`audit_uv_atlas`) 优化贴图与粗糙度通道，输出 `stage4_apply.json`。

## 工程安全、隐私与合规铁律

1. **跨进程许可隔离红线**：`pipeline/`、`run_pipeline.py`、`workflows/` 及未来扩展主模块**严禁直接 `import bpy` 或 `bmesh`**，必须通过 `pipeline/blender_runner.py` 跨进程调用 `blender_scripts/`。
2. **禁止硬编码个人路径与凭证**：任何本机绝对路径（用户名、桌面密钥路径、个人盘符）或 API Token 严禁写入源码、模板与示例配置；统一通过环境变量（见 `.env.example`）传入。
3. **付费 API 零自动重试与脱敏**：任何付费接口调用前必须检查 `CircuitBreaker`，先落盘 `generation_id` 再轮询，日志与错误提示须经过 `redact_secrets()` 脱敏。
4. **严禁打包未授权模型资产**：测试集与公共仓库仅允许使用原创或明确 `CC0` 授权模型；`.blend`/`.glb`/项目产成品目录由 `.gitignore` 默认拦截，严禁误提交。
5. **闭环 4 分支沉淀与诚实报告**：所有闭环任务须按结果记入本地 `knowledge/` 账本，并在文档与测试中严格区分“已实机验证”“调试中”与“Dry-Run 烟测”。

## Agent 入职复现考试（必做）

**如果你是 Agent，请接下来复现这两个题目，并将结果返回给你的代理人：**
1. `python run_pipeline.py dojo --dojo-dir projects/dojo_surgery --exam surgery --action setup` → 使用 `run_pipeline.py surgeon` 清理内囊与隐形环并保持外裙完整 → `--action grade` 拿到 `PASS`。
2. `python run_pipeline.py dojo --dojo-dir projects/dojo_retopo --exam retopo_500k --action setup` → 使用 `run_pipeline.py retopo` 完成 6 柱四边面重拓扑并保住非对称细长侧飘带 → `--exam retopo_500k --action grade` 拿到 `PASS`。
3. 将两科的 `dojo_scorecard.json` 分数明细与三联图 / 四边面线框图返回给你的代理人。详情见 [visual-surgical-cull-sop.md](skills/hello-3dneko-pipeline/references/visual-surgical-cull-sop.md)。
