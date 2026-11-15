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
