# AI 3D v1.1：实施记录与研究入口

更新：2026-11-20。依据 [v1.1 方案](jojo-badminton-AI-3D-motion-reconstruction-upgrade-v1.1.md)。

当前交付是 **P0 研究工具、ONNX-X 与官方 MMPose-L 的真实连续窗口对照、显式相机假设研究，与默认关闭的 P1 基础**。两条路径已从 3 个采样帧扩展到同一授权视频的 3 窗、各 183 帧；另有稀疏帧的本机离线 XYZ 预览。连续结果暴露明显肘角与骨长波动，相机参数敏感性也未解决，尚不能用于产品测量或人体回放。生产模型适配器仍拒绝推理，Gate A/B/C 未通过。前端保持示意 3D 与原 2D 指标语义；P2 等待模型和产品价值验证。

## 源码审计与 ADR

实施前 HEAD 为 `53952202cebc246edcacd1f183e2a283246e7bef`，实际 Git tree 为 `618dc7e9adfa0988948c9de09ef612e8e10c3646`。原方案将前者称作 tree；本记录纠正名称，不修改用户设计稿。审计依据是本地源码，而非仅 README。

| 基线 | 核查结果与决策 |
| --- | --- |
| `worker/main.py`、`algorithms.py` | 使用 `Wholebody`、133 → 65、2D 追踪和窗口事件；保留既有推理算法，3D 使用单独 CLI |
| `packages/schema/src/index.ts` | 原为 0.1.0，保留旧数据读取；新增 0.2.0 非破坏扩展与独立 `pose3d` |
| `apps/gateway/src/app.ts` | 原为一个串行子进程队列、单一分析状态；新增独立 3D 持久化队列，3D 不覆盖 2D 成功记录 |
| `apps/web/src/pose3d.ts` | `schematic-2d` 带人为展示深度；继续标识为示意 |
| `apps/web/src/anatomyRig.ts` | `retarget` 读取 XY 后将 Z 置零；P2 未开启，不把新数据接到该函数 |
| `fixtures/` | 合成演示，只用于契约/回归测试，不构成模型准确率、性能或许可证据 |
| 本地模型与视频 | 补查忽略目录后，发现 `data/models/rtmw_balanced.onnx` 与 `yolox_m.onnx`（现有2D权重）、两段原始上传 MP4、对应代理及上传 smoke 样例。未发现3D权重、逐段授权/人工标注证据清单；未审阅视频内容，不能把这些文件自动计作POC授权样本。未扫描个人其他目录 |

**ADR-3D-001：失败时保留 2D。** 独立任务键、子进程、超时、取消、重试和产物路径；已成功 2D 可即时读取。结果按任务键分目录，避免不同模型/范围覆盖。

**ADR-3D-002：不开启未验证模型。** `POSE3D_ENABLED=0`；即使打开开关并设置权重路径，两条适配器仍返回 `model_unavailable`。提供本地权重路径不等于完成坐标、许可或精度审核。没有自动下载，没有虚构哈希，没有用演示骨架替代模型输出。

**ADR-3D-003：窗口参数不冒充实测。** `preSec` / `postSec` 为 `null`，自动窗口须经 POC 后配置。研究配置中的质量/关联/视角阈值只供契约开发，`validationStatus=unverified`；不得据其宣称可靠。未知视角阻断，正面/后视携带受限警告。

**ADR-3D-004：P2 等待证据。** 先完成 ≥3 段授权真实视频 A/B、坐标与左右核对、按视角评测和 5–8 位用户研究。未通过前不新增“AI 估计 3D”播放声明，不做完整前端 Rig 重构。

**后续研究进展（2026-11-16 至 17）**：用户已授权两段现有视频用于本机内部验证；准备工具选出其中一段可用 2D 输入。已下载并核验固定 RTMW3D-x ONNX，执行 BGR/RGB 各 3 帧，并在增加代理实际 PTS 核对后重跑 RGB 3 帧。已在真实输出上复现 Z 解码分母差异，并按 5 组显式相机假设恢复 XYZ；同一帧左肘角计算值跨度约 32°。研究数据不作为用户 3D 文件发布。模型来源与命令见 [模型实测记录](pose3d-model-evaluation.md)、[相机恢复研究](pose3d-camera-research.md)；准备流程见 [研究输入说明](pose3d-prepare.md)。这取代初始“无本地3D权重、没有真实调用”的状态，但不改变 Gate 未通过的结论。

**2026-11-18 本机研究预览**：`scripts/pose3d_research_preview.py` 已从最新审计报告生成 `data/pose3d-research/camera-preview-03.html`。离线文件内嵌 3 个采样帧，可切换 5 组相机假设、查看同帧 2D 和旋转身体 17 点 XYZ；缺失点不补齐，所有情景共用尺度。生成时重新核验授权/来源/PTS，并从原始数据重算核对全部研究数组。文件约 4.04 MB，无外部资源，JavaScript 语法检查通过。内置浏览器拒绝本地文件协议，因此尚未完成浏览器视觉验收；此工具只供 P0 人工研究，不是 P2 产品 Rig 或连续动作回放。

**2026-11-18 路径 B 真实运行**：在独立 `.venv-pose3d-b` 安装官方 MMPose 1.3.2，并从官方 MMCV 2.1.0 源码编译真实 CPU 算子，解决 lite 环境缺少 `_ext` 的导入失败。`scripts/pose3d_mmpose_probe.py` 对相同 60/169/327 帧实跑成功；补充实际解码器属性后重跑到 `mmpose-l-02`，两次全部原始数组数值一致。新运行记录时间、内存、权重/源码/config hash 和原始坐标空间，未修改 2D 环境。安装与复现见 [MMPose 环境记录](pose3d-mmpose-environment.md)。

**2026-11-19 连续窗口研究**：新增有上限的窗口执行器与评估器，顺序运行 A/B 的相同 183 帧，两路均成功保留全部原始输出，总耗时分别 13.3641 / 20.8695 秒。相同显式相机假设下，每路有 180 对窗口内相邻帧；部分肘角变化超过 100°，已保存未平滑曲线和待复核帧。三个窗口只来自一个视频，尚未人工确认完整挥拍。详细证据、复现步骤和解释见 [连续窗口研究](pose3d-window-research.md)，人工输入要求见 [素材与标注规范](pose3d-annotation-guide.md)。

## 模块所有权与交付边界

| 责任 | 实施文件 | 状态 / 依赖 |
| --- | --- | --- |
| Architect / QA | 本记录、模型与许可审计、P0 评估及原始探针脚本 | A/B 同帧原始对照已完成；正式验证集与完整场景验收待完成 |
| Schema | `packages/schema/` | 0.1 兼容、0.2、3D JSON/f32 校验与契约测试 |
| Backend | `apps/gateway/src/pose3d-jobs.ts`、`pose3d-windows.ts` | 独立状态、幂等/缓存、作用域/窗口/门控、取消/超时/重试、受控文件读取 |
| Perception | `worker/pose3d/`、`pose3d_main.py`、`scripts/pose3d_probe.py` | 基础模块可测试，研究探针实际可运行；生产适配器与完整逐帧流水线未接通 |
| Frontend | 原有示意回放、`Pose3DNotice.tsx` | P2 暂缓；新增独立子任务状态提示与轮询，明确合成/二维示意及肌肉图例，仍不播放 AI 数据 |
| 产品负责人 / 数据负责人 | 授权与数据冻结、价值研究与发布决策 | 尚无实测结果，不能由自动化测试替代 |

## 可运行入口

```bash
npm run check
.venv/bin/python -m unittest discover -s tests/worker -v
npm run test:pose3d-evaluation
python3 scripts/pose3d_evaluate.py validation/pose3d-manifest.json --output output/pose3d-report.json
python3 scripts/pose3d_compare.py validation/pose3d-manifest.json --output output/pose3d-ab-report.json
```

最后两条是**已有标准化产物的离线评估**，不下载模型、不调用真实推理。每次使用新的输出文件名；已有文件、符号链接或并发创建的输出会被拒绝，不覆盖输入或原报告。空 manifest 的预期结果是 `blocked`、`releaseReady: false`，退出码 `2`。有可评估产物时退出 `0` 只表示收集到证据，`gateDecision` 仍为 `not_assessed`。输入损坏、缺少授权、模型坐标证据缺失分别写入错误记录。工具测试使用合成数值，仅检验计算与拒绝边界。

独立 Worker 实际采用以下 CLI，与设计稿的示意参数不同：

```bash
.venv/bin/python worker/pose3d_main.py --request /absolute/path/request.json --output-dir /absolute/path/output
```

请求由 Gateway 创建，包含 `analysis`、`video`、`pose3d`、`config` 和本地 `inputPath` / `timestampsPath` / `keypointsPath`。生产 Worker 校验输入后仍返回显式失败；没有可供播放的三维产物。独立研究探针实际使用的主要依赖版本已写入 `worker/requirements-pose3d.txt`；该文件记录已观察环境，不代表已验证一个全新隔离环境或完成发布审核。npm 的 Python 测试命令优先使用 `PYTHON`，其次项目 `.venv`，最后系统 Python，以便 NumPy 测试使用正确环境。

网关 API：

| 方法 / 路径 | 行为 |
| --- | --- |
| `GET /api/analyses/:id/pose3d` | 无任务时返回未请求/功能关闭说明；有任务时返回来源、窗口、质量和错误 |
| `POST /api/analyses/:id/pose3d` | 默认 `scope=swing-windows`；`manual` 带 `windows`；`full` 须显式指定并受上限约束 |
| `POST /api/analyses/:id/pose3d/retry` | 只重试失败/质量拒绝子任务 |
| `POST /api/analyses/:id/pose3d/cancel` | 取消排队或运行中的 3D，终态使用 `failed` / `cancelled` 错误码 |
| `GET /api/analyses/:id` | 读旧记录，并按选择的 3D 子任务附加概要 |
| `GET /data/results/:id/pose3d/:jobId/pose3d.f32` | 仅成功且通过校验的对应任务产物；增加 jobId 隔离不同缓存结果 |

配置入口：`.env.example` 的 `POSE3D_ENABLED`、`POSE3D_TIMEOUT_MS`、`POSE3D_PYTHON`、`POSE3D_CONFIG`。独立配置位于 `worker/pose3d/config.json`。普通用户 API 不允许 `evaluationMode` 绕过视角门控。当前仍为本机单进程开发服务，无云端鉴权或多实例队列。

缺少权重时返回 HTTP 503 / `model_unavailable`，不伪造必需的 SHA-256 来源字段。未通过 POC 配置的自动窗口返回 `window_policy_unverified`。无 3D 子任务时 GET 返回 `{status:"disabled",pose3d:null,errorCode:"feature_disabled"或"not_requested"}`，这不是包含虚构模型信息的 `Pose3DResult`。运行配置、请求和 provenance JSON 保留在私有任务目录，公共数据路由仅发布校验过的成功 `pose3d.f32`。

## 第一阶段自动化结果（2026-11-16；不含后续研究工具新增测试）

| 命令 | 实际结果 |
| --- | --- |
| `npm run check` | TypeScript 与 Vite 构建成功，72 项测试全部通过 |
| `.venv/bin/python -m unittest discover -s tests/worker -v` | 58 项全部通过，覆盖原有 2D 与新增 3D 数值/拒绝边界 |
| `npm run test:pose3d-evaluation` | 11 项全部通过，已接入 CI |
| `npm ls @badminton/schema` | Web/metrics 均解析至工作区 0.2.0，无旧版本依赖残留 |
| `git diff --check` | 通过 |
| 两条离线评估 CLI（空 manifest） | 均退出 2，`blocked` / `releaseReady:false`，符合预期 |

共 141 项自动化测试。Vite 仍提示现有 Three.js 查看器包超过 500 kB；这不是构建失败。测试需监听本地 API 与 tsx IPC，受限沙箱下必须允许这些端口。旧合成 fixture 保持 schema 0.1.0；新 2D 输出为 0.2.0，pipeline 改为 `0.1.3-schema-v02`。未运行真实视频全集回归，因为尚无授权标注清单。

错误复现覆盖在 `apps/gateway/test/pose3d.test.ts` 与 `tests/worker/test_pose3d.py`：缺模型、无窗口、未知视角、崩溃、超时、取消、重启中断、文件截断、路径穿越、符号链接、评测结果混入用户输出、不合法坐标矩阵。它们均保留 2D 成功状态。运行中取消与终态落盘共用锁，避免取消被覆盖为成功；网关重新校验双髋与原点、逐点缺失模式、窗口外缺失、二进制长度、来源及实际权重 hash。通用静态服务前还统一解码路径，测试确认 `%70ose3d` 和编码分隔符不能绕过私有目录拦截。

`apps/gateway/test/pose3d-worker-fixture.py` 只生成标识明确的合成传输测试字节，不是可选生产后端。真实 CLI 的测试证实：即便提供一个本地权重文件，尚未验证的模型仍返回 `model_unavailable`。这不能视为模型实际推理成功。离线空数据报告保存在忽略目录 `output/pose3d-report.json` 与 `output/pose3d-ab-report.json`。

### 稀疏探针研究工具验证（2026-11-18）

`npm run test:pose3d-evaluation` **127 项通过**，`npm run test:worker` **58 项通过**。研究工具最终日志为 `output/pose3d-tests-final-2026-11-18.log`。新检查覆盖实际 PTS 对齐、显式相机恢复、离线预览、MMPose 原始输出、跨模型 RAW 对照、禁止覆盖证据，以及相同帧数/时间轴但不同视频 hash 的评估拒绝。准备工具兼容旧 2D 的有限 XY、零置信度点，bbox 仍按原阈值排除这些点。CI 增加预览 PNG 解码所需的 headless OpenCV，未改动原 2D 环境；本地检查不代表远程 CI 已运行。前端/网关本轮未变更，沿用第一阶段 72 项构建与测试结果。

真实同帧对照已保存为 `data/pose3d-research/runtime-probe-comparison-01.json`：3 帧共 51 个身体点，两模型图像 XY 平均距离 `2.3935 px`，正确解码并去髋中心的相对 Z 平均绝对差 `0.081425`（非米制）。该报告验证来源和坐标处理后记录模型分歧，不提供准确率、胜者或发布结论；详情见 [模型实测](pose3d-model-evaluation.md)。

### 连续窗口工具验证（2026-11-19）

`npm run test:pose3d-evaluation` **165 项通过**，最终日志为 `output/pose3d-tests-final-2026-11-19.log`，本轮新增 38 项。覆盖完整逐帧分母、一路零成功、写盘中断、流式读取、实际时间间隔、跨缺失/窗口禁止连接、原始窗口配置绑定与输出保护。A/B 183 帧实跑及 `window-comparison-01.json` 的证据复核均成功；生成的静态曲线已核对。前端、网关与生产 Worker 本轮未修改，沿用此前检查结果，不把旧测试记录算作本轮重跑。

## 坐标与关键点协议

- `cwb-65` 索引：COCO-WholeBody `[0..22] + [91..132]`，去掉脸部 68 点，保留身体/脚/双手；左右按人体解剖侧，禁止镜像后偷偷交换。
- 二进制固定 Little-Endian Float32，`frameCount × 65 × 4`，每点 `[x,y,z,confidence]`，逐帧严格对应已有采集时间戳；缺失只能是 `[NaN,NaN,NaN,0]`。
- 根为左右髋 11/12 的中点。统一空间 `camera-root-relative`，轴标签 `x-right-y-up-z-forward`，单位 `model-relative`。不能输出米制距离或把显示地面位置称为真实走位。
- 窗口外和被阻断窗口必须缺失；不跨窗口平滑，不无限延用上一帧。原始和平滑产物须分开留档。
- 轴标签本身不足以证明哪个 Z 符号朝向相机。评估运行证据必须记录 `depthNearerSign`（`1` 表示 Z 越大越近，`-1` 表示越小越近）及 `coordinateEvidence`；该约定须由已知姿态/相机和真实视频确定。
- 通用坐标模块仅接受带证据的正交/统一尺度矩阵；模型适配器需先恢复同尺度 XYZ。rtmlib 当前上游代码的原始 XY 与 Z 不同尺度，不能直接套符号翻转。详见 [模型审计](pose3d-model-evaluation.md)。

## P0 数据清单与评估格式

`validation/pose3d-manifest.json` 故意为空且未冻结。不要将模板或演示加入真实样本计数。每段真实片段先录入授权证据、保留期限、文件 hash、采集时钟、人工视角、数据划分。完整验证集目标为 15–30 段，覆盖左右手、机位、帧率、多人/交叉/遮挡和照明；POC 先取侧面、斜侧、转身/遮挡各一段。按球员/拍摄场次分组划分，避免同源片段泄漏到验证集。

以下为**格式示意，不是真实证据**。文件路径均相对所在 JSON；私有视频/授权材料不要提交到仓库，建议放置在已忽略的 `data/` 中：

```json
{
  "schemaVersion": "pose3d-evaluation-0.1.0",
  "frozen": false,
  "videos": [{
    "id": "authorized-side-001",
    "videoPath": "../data/validation/side.mp4",
    "videoSha256": "填写实际64位SHA256",
    "authorizationEvidence": "../data/validation/consent.md",
    "retentionUntil": "2027-01-01",
    "viewClass": "side",
    "split": "calibration",
    "frameCount": 120,
    "timestampsPath": "../data/validation/timestamps.f32",
    "manualHitFrames": [60],
    "ordinalDepth": [{"frame": 60, "a": 10, "b": 6, "closer": "a"}],
    "runs": [{"path": "../data/validation/run-a.json"}, {"path": "../data/validation/run-b.json"}]
  }]
}
```

每个 `run-*.json` 包含完整 `pose3d` 成功元数据，并额外记录：

```json
{
  "pose3d": "替换为实际成功的Pose3DResult对象",
  "timestampsPath": "timestamps.f32",
  "keypointsPath": "pose3d.f32",
  "provenance": {
    "inputVideoSha256": "与所属清单条目的videoSha256完全相同的原视频hash",
    "sourceRevision": "模型实现的固定commit",
    "dependencyVersions": {"python": "实测版本", "runtime": "实测版本"},
    "device": "实际CPU/GPU型号与后端",
    "configHash": "实际配置hash",
    "coordinateEvidence": "coordinate-verification.md",
    "depthNearerSign": -1
  },
  "manualReview": [{"frame": 60, "leftRightSwapped": false, "identitySwitch": false, "severeError": false}],
  "measurements": {"elapsedSec": 1.0, "windowSeconds": [1.0], "peakRamBytes": 1, "peakVramBytes": null, "coldStart": true}
}
```

示例的 hash、sign、时间、资源、frameCount、review 布尔值都是格式占位，必须由实测替换。`pose3d` 不接受字符串。运行记录必须通过 `provenance.inputVideoSha256` 绑定清单原视频；仅帧数/时间轴相同不能证明视频相同。保存原始/平滑版本时分别建立运行记录；单次 A/B 比较 manifest 对每个视频只能选一个 A 和一个 B。工具比较完全相同时间戳上的共同有效点，不能证明三维真值；不同 runtime 的相对尺度差异也须人工核对。

人工序数标注用 cwb 索引；`closer=a/b` 表示该关节离相机更近，不确定或无法判别的帧不标为二者之一。记录标注者、复核者和争议帧在私有证据中。可逐条设置 `viewClass` 处理转身窗口。缺失预测单独进入 coverage，不算作正确，也不会从报告中消失；序数准确率只对有预测的标注计算。左右交换/ID switch 统计只覆盖人工检查帧，不等于全视频无切换。

正式评估工具统计：按视角序数准确率与覆盖、按视角左右交换/ID switch/严重错误、标注击球帧窗口覆盖、窗口内骨长相对漂移、提供的实测资源记录，以及 A/B 数值与关节对深度符号一致性。独立 RAW 探针额外提供与原 2D 的图像偏差信号及预处理比较。**尚未实现**足滑、完整验证集的独立重投影偏差、视角误放行混淆矩阵、漏/多窗口匹配、原2D事件回归联合报告、GT MPJPE/PA-MPJPE 和模型 A/B 自动推理调度。无 GT 时明确 `not_measurable`，不能由现有工具的退出码代替 Gate B。

## 成本与产品价值记录模板

**同源连续研究窗口成本已实测，完整任务与部署成本尚未测量。** 两路 183 帧的逐窗/总耗时与进程峰值 RSS 已整理至 [本机资源实测](pose3d-cost-report.md)。仍需补足人工确认的完整挥拍、独立场景、固定资源配置的冷/热启动与重复运行，以及 10 秒/120fps、60 秒片段、RAM/VRAM、失败重试。实际账单估算须纳入部署单价、冷启动、队列、存储和重试。当前没有部署单价或定价对照，金额结论仍为 `not_measurable`。

**P0-U 尚未开始。** POC 的三段真实视频准备视频+2D、示意3D、AI估计3D三组离线材料；招募 5–8 位业余球友，匿名/随机次序呈现，记录身体朝向/肘位置/阶段判断正确率、预先冻结的信任评分和复用意愿。开始前冻结门槛与方案 hash；提议门槛为 AI 的任务正确率不低于示意、多数参与者的信任/复用偏好倾向 AI。报告人数/分母和原始匿名记录，不声称统计显著。未通过则不进入 P2；部分通过限定到证据支持的机位。

## Gate 与剩余任务

| Gate | 当前结论 | 缺少证据 |
| --- | --- | --- |
| A / POC | 未通过（A/B 均已完成同一视频的 3 个连续研究窗口） | ≥3 段授权场景、人工确认的挥拍范围及坐标/左右/深度核对、选型与完整成本记录 |
| B / Beta | 未通过 | 冻结验证集、所有必需按视角指标、视角策略校准、许可放行、成本和产品价值结果 |
| C / Verified | `not_measurable`（可选） | 校准多视角或可信同域3D真值及评测协议 |

下一阶段按顺序执行：补足授权场景与人工标注/相机证据 → 扩展完整窗口 A/B 与成本记录、完成许可结论 → 冻结窗口/门控策略和 runtime → 实现正式逐帧适配器 → 完成价值研究后决定 P2。现有两段视频中只有一段具备可用 2D 输入，不能用不同采样帧充作三段独立场景。商业发布受 [许可审计](pose3d-license-review.md) 约束。自动化通过只证明所覆盖的软件契约，不代表以上 gate 通过。
