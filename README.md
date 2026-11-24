# Hello 3DNeko (`hello-3dneko`)

AI Agent 驱动的二次元 / 风格化 3D 角色工作流与**闭环自主重拓扑 (`retopo`)** 开源框架。通过跨进程无头调度，将 2D 三视图生成规范、基准素模结构分析、阶段流水线编排骨架与 **6 维独立几何验收闭环自动重拓扑引擎**串联为可审计、可回滚的开源基础底座。

> [!IMPORTANT]
> **关于开源范围与知识产权声明（必读）**
>
> 1. **开源截止边界**：本公开仓库**仅开源至 Agent 闭环自动重拓扑（`retopo`）阶段及前置基础框架**。
> 2. **后续功能暂不开源**：重拓扑之后的下游闭环流程（包括但不限于自主 UV 展开与 PBR/AO 烘焙闭环、服装拟合与自动骨骼权重/布料物理求解及姿态压测、多资产自动组装、游戏引擎工程直出等）**目前尚未完成全量充分验证，暂不开源**。
> 3. **刻意阉割声明与产权归属**：当前公开版本为**经过刻意功能裁剪（阉割）的开源版本**。上述未开源及已裁剪移除的后续模块、高阶多目标决策算法与私有调参经验库均属于项目发起人的**独立知识产权与专有技术成果（Proprietary IP）**，保留用于后续内部验证、商业版本及定制技术服务，不包含在本次开源代码许可范围之内。

---

## 如何使用（把这个页面截图给你的 Agent）

> 如果你正在使用 **Antigravity / Codex / Claude Code / Cursor** 等 AI 编程 Agent：**直接把本页面截图发给你的 Agent**，告诉它“按照页面说明帮我在本地跑通 Hello 3DNeko 的初始化与闭环重拓扑（`retopo`）”，你的 Agent 即可直接按下面步骤执行：

### 给用户与 AI Agent 的四步执行指令

1. **检查环境并安装底座（需 Python `>= 3.10` 与本地 Blender `4.x` / `5.x`）**
   ```powershell
   pip install -e .
   cp .env.example .env
   ```
   - 如本机 Blender 不在系统默认 `PATH` 下，请在 `.env` 中配置 `HELLO_3DNEKO_BLENDER_EXE=<你的 blender.exe 路径>`（主协调器与 Blender 子进程严格隔离，不会自动联网乱改本机软件）。
2. **确认离线单测通过（零付费 API 调用）**
   ```powershell
   python -m unittest discover -s tests -p "test_*.py" -v
   ```
3. **运行基础工作流编排与闭环 Agent 重拓扑（`retopo`）**
   ```powershell
   # ① 初始化一个角色工作区
   python run_pipeline.py init --id neko_demo --name "Demo Character" --brief "Cel-shaded anime character in 40-deg open A-pose"

   # ② 预演前置全链路状态机（Dry-Run 不产生任何付费调用）
   python run_pipeline.py run --project projects/neko_demo --dry-run

   # ③ 核心能力：闭环 Agent 自动重拓扑 + 6 维独立几何验收 + 失败自动回滚与路线切换
   python run_pipeline.py retopo --input <raw_model.glb> --out projects/retopo_demo --max-iters 5

   # ④ 汇总 7 项统一实验指标并生成本地记分卡
   python run_pipeline.py experiments
   ```
4. **请你的 Agent 接着读完仓内规范文件后再干活**
   - 项目协作规则：[AGENTS.md](AGENTS.md)
   - 工作流与重拓扑调优参考 Skill：[skills/hello-3dneko-pipeline/SKILL.md](skills/hello-3dneko-pipeline/SKILL.md) 与 [closed-loop-retopology.md](skills/hello-3dneko-pipeline/references/closed-loop-retopology.md)
   - **给 Agent 的防幻觉提示**：本开源仓库只开放至 `retopo` 与 `experiments`；请勿假定或捏造后续未开源的 `uv-bake` / `cloth-rig` 等商业/专有模块命令。

---

## 开源功能范围与实测状态

| 模块 / 能力 | CLI 命令 | 当前开源与验证状态 |
| :--- | :--- | :--- |
| **项目初始化、密钥熔断守护与 UV 像素审计** | `init` / `status` / `inspect-uv` | **已开源 · 离线测试通过**（含 `guards.py` 零自动重试与熔断锁） |
| **闭环 Agent 自动重拓扑与 6 维几何验收** | `python run_pipeline.py retopo` | **已开源 · 已实现并实机验证**（流形修补、Quadriflow / Adaptive Voxel 路线切换、失败快照回滚） |
| **统一 7 项实验指标账本** | `python run_pipeline.py experiments` | **已开源 · 已实现并实机验证**（记录自主完成率、人工干预、返工次数、耗时、评分、失败原因、配置哈希） |
| **Stage 0~5 阶段编排骨架与模版** | `python run_pipeline.py run --dry-run` | **已开源 · 仅完成 Dry-Run 离线烟测**（用于接入用户自带 2D/3D 工具链，实机端到端付费链路待各自按需联调） |
| **重拓扑之后流程（自主 UV/烘焙、服装拟合与权重绑定、场景组装、引擎直出等）** | *(本版已刻意阉割移除)* | **未充分验证 · 暂不开源（属项目发起人知识产权与后续商业/定制演进部分）** |

---

## BYOK（用户自配 Key）与“有羊毛呢”新手提示

Hello 3DNeko 开源版严格采用 **100% BYOK（Bring Your Own Key）** 模式：本仓不托管、不代充、也不转售任何付费 AI 服务账号，所有 API 请求均由用户在本地直接配置并调用，`guards.py` 会持久化 `generation_id` 并在连续失败时自动熔断以保护用户余额。

> **小贴士（有羊毛呢）**：部分第三方 3D / 生图服务曾为新用户提供过约 **1 美元体验一个月 / 30 个积分** 的入门体验活动，非常适合独立开发者与学生低成本试跑首批模型；具体价格、积分额度与使用条款请以各第三方官网实时活动页为准，请勿批量注册小号以免触发风控。

---

## 许可边界与跨进程架构说明

本项目采用**双许可跨进程解耦架构**，保障社区自由使用与下游合规开发：

- **主编排器、CLI、验证架构、安全熔断与测试 (`run_pipeline.py`, `pipeline/`, `workflows/`, `templates/`, `tests/`)**：采用 **Apache License 2.0**（详见根目录 `LICENSE`）。主协调层**不直接导入 `bpy` / `bmesh`**。
- **Blender 无头执行脚本 (`blender_scripts/`)**：运行在 Blender 内部并直接调用 `bpy` / `bmesh`，遵循与 Blender 一致的 **GNU GPL-3.0-or-later**（详见 [blender_scripts/LICENSE-GPL-3.0-NOTICE.md](blender_scripts/LICENSE-GPL-3.0-NOTICE.md)），由外部主进程通过 `subprocess` 与 JSON/GLB 标准文件隔离调用。
- **零 AGPL 记忆组件绑定**：本仓不依赖也不包含任何 `AGPL` 协议外部记忆服务端；本地经验记录采用独立轻量化 JSONL 账本结构。
