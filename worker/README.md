# Python 感知 Worker

当前实现是 P0a 的可运行算法基线，**尚未经过真实验证集校准**。它不把演示骨架当成真实视频分析结果，不输出好坏判断。模型未配置、视频无法解码、目标不明或没有可信挥拍时，任务明确失败或质检拒绝。

## 安装与启动

需要 Python 3.10+、FFmpeg（含 ffprobe）。在仓库根目录：

```sh
python3 -m venv .venv
.venv/bin/pip install -r worker/requirements.txt
```

Gateway 默认优先使用项目的 `.venv/bin/python`；仅需要其他解释器时设置 `PYTHON=/绝对路径/python`。创建虚拟环境前请确认 `python3 --version` 为 3.10 或更新版本；macOS 系统自带的 Python 可能较旧，可改用已安装的 `python3.11 -m venv .venv`。配置一组兼容的本地 ONNX 模型：

```sh
export RTMLIB_DET_MODEL=/absolute/path/yolox_m.onnx
export RTMLIB_POSE_MODEL=/absolute/path/rtmw_x_256x192.onnx
export RTMLIB_DET_SIZE=640,640
export RTMLIB_POSE_SIZE=192,256
export POSE_DEVICE=cpu
```

模型输入尺寸是 **width,height**，必须与权重匹配。姿态模型必须输出 COCO-WholeBody 的 133 点，使用 MMPose 顺序，不能开启 OpenPose 转换。代码将 body17 + feet6 + 索引 91:133 的 hands42 转换为 cwb-65。默认使用 ONNX Runtime CPU；GPU 需要另装相应的 ONNX Runtime 后端及驱动。

也可显式设置 `ALLOW_MODEL_DOWNLOAD=1`，由 RTMLib 下载其 balanced 默认模型；默认不会下载。网络与下载耗时不计入任何已验证的性能承诺。模型代码及权重应分别记录来源和许可后再部署。

API 依据：[RTMLib 官方仓库](https://github.com/Tau-J/rtmlib)、[Wholebody 实现](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/wholebody.py)。本仓库不包含模型权重。

```sh
.venv/bin/python worker/main.py \
  --input /path/to/original.mp4 \
  --output-dir data/analyses/example \
  --analysis-id example --video-id video-example \
  --swing-type smash --target-point 0.45,0.55
```

`--target-point` 为视频画面中的归一化点，可省略；`--capture-fps 120` 仅在明确知道真实采集帧率时使用，不能根据文件播放帧率猜测。输出到 stdout 的最后且唯一一行是 JSON：`{video,analysis,error?,errorCode?}`。推理库日志发往 stderr。结构化失败也返回退出码 0，以便调用方读取具体状态；进程崩溃/超时应由 Gateway 判定失败。

产物：`result.json`、`proxy.mp4`、`timestamps.f32`、`keypoints.f32`，有可用声音时还包含 `audio.wav`。Float32 文件显式使用小端序。只有完整成功才写入骨架引用；原视频由 Gateway 保留。

## 数据与处理约定

算法版本统一读取仓库根目录的 `pipeline-version.json`；修改模型、阈值或时序处理时需递增 `pipelineVersion`，Gateway 与 Worker 使用同一个版本来创建和核对任务。更新后用下方 `regression.py` 在同一验证集上重跑。

- 摄取读取逐帧 `best_effort_timestamp_time`；时间从第一帧归零。显式采集 FPS 仅支持等间隔源，按采集时间重建时轴；可变重映射拒绝。代理视频同步转换为采集时轴，逐帧编号保持一致。
- 普通视频的 PTS 被视为采集时轴。已导出的慢动作文件若删除了采集信息，仅靠 PTS **无法证明真实采集速度**。已知慢动作标签会要求确认采集 FPS；不支持分段慢动作恢复。不要将该启发式当成完整的 iOS/Android 慢动作元数据解析器。
- 按时间顺序对代理视频的每一帧推理一次，全程使用同一个目标跟踪器，避免抽帧造成关键点周期性消失。质量统计覆盖全部帧；未检测到目标或置信度不足的点仍保留为缺失，超过 50ms 的缺失不插值。指标/可用率应在击球窗口评估。
- 自动选择明显最大的目标；多人尺寸接近要求点选。后续按位置连续性关联，交叉或目标丢失不切换到其他人。这是保守几何跟踪基线，不具备外观重识别。
- One Euro 滤波、短间隙插值、腕部速度峰、音频瞬态门控均使用真实时间戳。离线滤波对前向、后向结果取平均，再保留 50% 原观测以降低快速挥拍的相位滞后和幅度衰减；置信度不足 0.5 的点先设为缺失，避免拖动后续可信点。滤波不会跨越缺失或超过 50ms 的时间间隙；短间隙插值仍会降低置信度。这些参数仍需真实标注集校准。音频只增加相邻视觉候选的置信度，不把声波到达时刻作为击球帧。重映射音轨不做不可靠恢复，退回纯视觉。
- 左右手由腕速度差估计，差异不足返回 `unknown`；动作类型来自用户输入。没有腕峰则拒绝，不虚构击球。
- 视角分类是肩宽/躯干比例与面部可见度的保守启发式，无法区分背面和遮挡时为 `unknown`。光线只检测偏暗，未做逆光分类。所有结果包含 `view_classifier_uncalibrated` 提示。
- 阶段边界用腕速下降阈值估计，缺失时不填；速度阈值、滤波滞后、音频窗口和视角阈值必须用真实视频校准。
- 当前本地实现限制 60 秒、4K、15–480fps、20000 帧。全帧推理的耗时随视频帧数增加，耗时目标尚未测量；模型漏检、遮挡或置信度不足仍可能导致短促挥拍漏检。

## 测试和真实评测

```sh
.venv/bin/python -m unittest discover -s tests/worker -v
.venv/bin/python scripts/evaluate.py /path/to/validation.json --output report.json
.venv/bin/python scripts/regression.py /path/to/validation.json --output-dir output/regression --baseline baseline.json
```

验证清单示例（路径相对清单；真实标注和视频由项目提供）：

```json
{
  "cases": [{
    "id": "side-120-smash-01",
    "video": "videos/side-120-smash-01.mp4",
    "result": "results/side-120-smash-01/result.json",
    "swingType": "smash",
    "targetPoint": [0.45, 0.55],
    "expected": {
      "status": "succeeded",
      "impactFrames": [182, 537],
      "handedness": "right",
      "view": "side"
    }
  }]
}
```

对拒绝样本标注 `expected.status="rejected_quality"` 及 `expected.rejectCode`，否则不能把拒绝算作任务成功。击球匹配是一对一匹配，漏检计入召回分母，误检单列并给精确率。关键点可用率在标注击球前后 100ms 统计躯干与持拍臂；视角 `unknown` 计错。10 秒/120fps 样本才参与 2 分钟运行时目标，缺少适用样本为 `incomplete`。数值回归不允许缺数据被当成通过。合成 fixture 会被评测拒绝，不能证明产品精度。`regression.py` 每次重跑整个清单，输出版本间指标差；检测退化或未通过全部验收目标时退出码为 1。

10–15 段覆盖各档帧率、视角、光照和多人的授权真实视频，以及人工逐帧标注，目前仍需项目方提供。结果与阈值必须在指定机器和同一数据版本上比较。

## 独立 AI 3D 研究 Worker

`pose3d_main.py` 与原 `main.py` 分开运行，3D 失败不会修改二维分析状态或产物。当前交付包含窗口、视角门控、关键点映射、坐标协议、目标关联、可选时序清理与二进制工具；**RTMLib 与 MMPose 适配器均保持阻断，真实模型推理尚未集成**。现有示意 3D 仍是示意来源。

这样处理的原因是 P0 的完整 A/B、坐标准确度与许可验证尚未通过。独立研究工具已经使用授权视频及核验过 hash 的本地权重，但 rtmlib `RTMPose3d` 的原始 XY 为裁剪坐标、Z 使用独立尺度，不能直接送入 Rig。研究中已完成显式相机假设下的反投影；这不代表相机已标定或估计准确。生产 Worker 不自动下载权重，MMPose 研究使用独立环境，不把 torch/mmcv/mmpose 引入原 2D 环境。

默认配置位于 `worker/pose3d/config.json`。`validationStatus` 为 `unverified`，`weightsPath` 为 `null`，自动挥拍窗口的 `preSec` / `postSec` 为 `null`；这些值须由 P0 得出后显式配置。现有视角、关联与质量数值是未校准的实验参数。滤波默认关闭，以便保留原始速度峰。`ALLOW_MODEL_DOWNLOAD=1` 只影响原二维 Worker，不会开启 3D。

网关默认 `POSE3D_ENABLED=0`。内部联调时可用 `POSE3D_PYTHON` 指定独立解释器、`POSE3D_CONFIG` 指向独立配置。单独执行已生成的请求文件：

```sh
.venv/bin/python worker/pose3d_main.py \
  --request /absolute/path/to/pose3d/request.json \
  --output-dir /absolute/path/to/pose3d/output
```

请求 JSON 包含：

| 字段 | 约定 |
| --- | --- |
| `analysis` / `video` | 完整的已成功二维 Analysis 和 VideoAsset |
| `timestampsPath` / `keypointsPath` | 原二维 Little-Endian Float32 文件的本地绝对路径 |
| `inputPath` | 使用原采集时钟、与骨架帧号一致的代理视频 |
| `pose3d` | 网关生成的完整 queued/running 元数据，含合并窗口、门控和真实权重 hash |
| `config` | 版本化的完整嵌套配置；不得把示例值当作已验证参数 |

stdout 最后一行是 `{ "pose3d": <终态元数据>, "error": <可选说明> }`。有效请求下，当前适配器返回 `failed / model_unavailable`；无窗口为 `rejected_quality / no_swing_windows`，全部窗口被门控为 `rejected_quality / view_blocked`。结构化终态退出码为 0，供网关读取；无法解析请求等无法返回元数据的情况退出码为 1。当前不会产生成功 `pose3d.f32` 或伪造来源文件。

可独立运行基础模块测试：

```sh
.venv/bin/python -m unittest discover -s tests/worker -p 'test_pose3d.py' -v
```

这些是合成数值契约测试，验证帧时钟、缺失值、窗口/视角边界、左右索引、显式坐标变换、交叉目标拒绝和失败隔离；不证明模型精度或用户价值。真实模型通过 P0 后，才可接入逐帧推理、保存原始/滤波版本和 provenance，并继续前端真实 XYZ 重定向。

### 仅供本机研究的 RAW 探针

`scripts/pose3d_probe.py` 可以实际调用本地 RTMW3D-x ONNX，直接使用同帧 2D bbox，输出原始四元组 NPZ、图像对照和执行报告。它不调用上述生产 provider，不生成产品 `.f32`，最多处理10帧，不自动下载模型。须提供绑定代理文件 SHA-256 的本机验证授权、已有采集时间戳、同帧骨架或明确 bbox。支持 `--color-order bgr|rgb`；二者只用于预处理对照，未确定生产设置。

已使用版本记录在 `requirements-pose3d.txt`。后续环境安装请使用独立 `.venv-pose3d-*`，避免修改原 2D 环境。研究探针本次实际在既有环境只读加载模型；尚未验证全新隔离环境的安装。

运行命令、实际3帧×2色序结果及后续解码/比较工具见 [模型实测记录](../docs/pose3d-model-evaluation.md)。本地 rtmlib 0.0.16 的首个返回值是混合尺度，而且其 Z 解码分母与官方配置存在差异；不能直接接入人体 Rig。

`scripts/pose3d_camera_research.py` 使用显式内参和根深度，在重验授权及原始产物后恢复研究用 XYZ，保存双髋去中心前后数据与相机参数敏感性报告。没有默认内参，不输出产品 `.f32`。详见 [相机恢复研究](../docs/pose3d-camera-research.md)。研究报告每次使用新输出路径，工具会拒绝覆盖已有证据。

`scripts/pose3d_mmpose_probe.py` 已在独立 `.venv-pose3d-b` 对相同三帧运行官方 RTMW3D-L，保存官方解码值、默认相机坐标与图像 XY。该环境需要实际 MMCV CPU 算子，`mmcv-lite` 不足以导入完整 API；构建与主要依赖版本见 [环境记录](../docs/pose3d-mmpose-environment.md) 及 `requirements-pose3d-mmpose.txt`。不得安装到原 2D 环境。两条研究脚本都不代表生产 provider 已启用。

`scripts/pose3d_window_probe.py` 在各自环境中运行有上限的连续研究窗口：最多 3 窗、每窗最多 3 秒且 90 帧、合计最多 240 帧。它顺序解码代理、每个任务只加载一次模型；缺目标、推理或写盘失败保留逐帧原因并继续，窗口外不推理。`scripts/pose3d_window_evaluate.py` 核对双方授权/来源/实际 PTS/窗口配置/原始数组，在相同显式相机假设下计算变化量，不跨缺失帧或窗口计算相邻变化。新目录/文件不得覆盖已有证据；原有限帧探针的 10 帧上限未放宽。实际 183 帧运行结果与复现命令见 [连续窗口研究](../docs/pose3d-window-research.md)。
