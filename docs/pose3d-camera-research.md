# 显式相机参数下的 3D 恢复研究

更新：2026-11-18。已从一段本机授权视频的 RGB 探针产物中，对 **3 个既有时间轴采样帧 × 5 组显式参数假设**完成相机 XYZ 反投影。最新 `camera-rgb-audited-03` 报告补充完整来源链与独立代理 PTS 复核，15 份数组与保留的 `camera-rgb-assumptions-02` 数值相同。产物仍是研究数据，`productCompatible=false`、`releaseReady=false`、`cameraCoordinatesValidated=false`。它验证了恢复链路可执行，并显示角度对相机假设明显敏感；没有完成实际相机标定或动作精度验收。

## 输入、权限与复现

本次实际输入与结果均位于被 Git 忽略的本机 `data/` 目录：

| 内容 | 路径 |
| --- | --- |
| 已授权 RGB 原始探针报告与逐帧 NPZ | `data/pose3d-research/probes/rtmw3d-rgb-clock-02/` |
| 显式相机假设 | `data/pose3d-research/camera-assumptions.json` |
| 最新恢复报告及 15 个研究 NPZ | `data/pose3d-research/camera-rgb-audited-03/` |

视频授权范围是 `local-internal-3d-validation`，`externalUploadAllowed=false`。视频、原始数组和恢复数组留在本机；本报告只记录实现、参数及研究统计。脚本不上传视频、不下载模型，也不再次执行模型推理。

在仓库根目录运行；输出目录必须为空，以保留本次已有证据：

```bash
.venv/bin/python scripts/pose3d_camera_research.py \
  data/pose3d-research/probes/rtmw3d-rgb-clock-02/report.json \
  --camera-config data/pose3d-research/camera-assumptions.json \
  --output-dir output/pose3d-camera-research-new

.venv/bin/python -m unittest discover -s tests/pose3d -p test_camera_research.py -v
```

生成离线研究预览；`--output` 文件必须不存在，脚本以原子方式创建且拒绝覆盖：

```bash
.venv/bin/python scripts/pose3d_research_preview.py \
  --camera-report data/pose3d-research/camera-rgb-audited-03/camera-research.json \
  --probe-report data/pose3d-research/probes/rtmw3d-rgb-clock-02/report.json \
  --output output/pose3d-camera-research-new.html

.venv/bin/python -m unittest discover -s tests/pose3d -p test_research_preview.py -v
```

预览 HTML 内嵌同帧图像，无网络资源或连续播放；可切换采样帧/参数情景，并拖动旋转使用真实 XYZ 的身体 17 点骨架，缺失点不补齐。所有情景共用显示尺度。生成时重新验证授权、来源与产物 hash、代理 PTS，并重算 XYZ 核对保存的数组。HTML 含采样图，仍只供本机研究。旧报告若缺少配置来源路径，需显式加 `--camera-config data/pose3d-research/camera-assumptions.json`。

最新真实研究报告已成功生成包含 3 帧、5 情景的预览文件；自动化与文件校验已通过，**尚未完成浏览器视觉验收**。当前浏览器工具不允许访问本地文件协议，因此没有以实际浏览器操作验证拖动和切换的视觉表现。

相机配置的 `schemaVersion` 必须是 `pose3d-camera-research-config-0.1.0`。每个 `scenarios[]` 元素必须显式包含 `id/source/evidence/imageWidth/imageHeight/fx/fy/cx/cy/rootDepth`。`source` 只接受 `assumed-research` 或 `calibrated`；后者也不会自动取得生产可用状态，脚本并未独立审验标定证据。

## 算法与数据边界

实现见 [研究脚本](../scripts/pose3d_camera_research.py)，源码依据见 [坐标审核](pose3d-coordinate-audit.md)。

1. 复用解码审计的验证：探针 schema 和研究标记、授权证据与输入 hash、时间戳 hash 和帧数、原始 NPZ hash/路径/形状/有限值、同帧时间、SimCC 位置和已声明的上游解码公式。
2. 检查 ONNX 的 X/Y/Z 输出长度为 `576/768/576`、分桶倍率为 `2`，得到深度输入长度 `288`、正确分母 `144`。从保存的 `keypoints_simcc` 独立重解码 Z，避免沿用本地 RTMLib 的分母 `192`。
3. 通过 ffprobe 独立读取代理全部逐帧 PTS，与提供的时间戳逐帧比较；读取实际画幅，要求与配置尺寸一致。图像 XY 使用同帧保存的 `keypoints_2d`；相机内参与根深度完全来自指定配置。
4. 反投影得到三分量 XYZ，再减去双髋中心。任何已解码点出现非正深度、双髋缺失或非有限结果，都会拒绝该次恢复。所有情景验证完成后才写入研究数组。

公式中 `q_z` 是已经除以分桶倍率的 SimCC 深度位置，`u/v` 是原图像素：

```text
z_relative = (q_z / 144 - 1) * 2.1744869
Z = z_relative + rootDepth
X = (u - cx) / fx * Z
Y = (v - cy) / fy * Z
P_root = (P_leftHip + P_rightHip) / 2
P_relative = P - P_root
```

原相机轴是 X 向右、Y 向下、正 Z 远离相机。展示数组用 `diag(1,-1,-1)` 改为 X 向右、Y 向上、正 Z 朝向相机。该换轴不改变姿态角度。根关节使用 WholeBody 索引 `11/12`；尚未把研究产物转换成产品 `cwb-65`。

输出 `camera-research.json` 和每情景每帧的 `*.camera-research.npz`。NPZ 保存 `camera_xyz`、`camera_root_xyz`、`camera_root_relative_xyz`、`display_root_relative_xyz`、图像 XY、原始分数、有效点 mask、相对 Z 和采集帧/时间。缺失点保留 NaN；JSON 使用空值和原因说明。**不产生产品 `pose3d.f32`，不写入分析记录，不改变现有 2D 指标。**

## 本次五组显式假设

全部场景均为 `source='assumed-research'`；画幅 `480 × 854`、主点 `(cx,cy)=(240,427)`。焦距数值采用像素单位。根深度与模型深度目标使用同一数值尺度，**不标注为实测米数**。

| 情景 | fx = fy | rootDepth | 与参考情景的变化 |
| --- | ---: | ---: | --- |
| `reference-prior` | 1145 | 5.14388 | 参考先验 |
| `focal-800` | 800 | 5.14388 | 焦距假设 |
| `focal-1600` | 1600 | 5.14388 | 焦距假设 |
| `root-depth-3` | 1145 | 3 | 根深度假设 |
| `root-depth-8` | 1145 | 8 | 根深度假设 |

参考焦距约 1145、根深度 5.14388 来自官方示例/codec 先验，主点取图像中心；它们均未针对这段视频标定。参数变化范围也不是该相机的不确定性分布。各情景仅用于观察选择这些参数会如何改变结果，不能据此选出“真实参数”。

## 真实素材中的角度敏感性

本次恢复报告状态为 `camera_research_recovered`。采样帧为 `60/169/327`，对应既有时间轴的 `2.0000/5.6333/10.9000` 秒；每个情景每帧均有 133 个已解码点。这里只根据原始分数大于零计算，不等于 133 个点都准确或通过解剖质量门控。

最新报告的 `sourceEvidence` 记录原始探针、相机配置、代理视频和时间戳的路径/hash。`clockEvidence` 表明 448 帧代理 PTS 与既有时间戳的最大误差约为 `3.97 × 10⁻⁷` 秒，小于 `0.0001` 秒容差。该结果仅确认**代理与既有时间文件对齐**；`originalCaptureTimingVerified=false`、`sourceEditingHistoryVerified=false`，不能据此证明原始采集帧率或原片未剪辑。

下表是**同一帧、同一模型输出**在五组参数下的计算结果，单位为度；区间是参数情景的最小值到最大值，不是置信区间，也不是与真值的误差。保留两位小数仅便于核对报告。

| 采集帧 | 参考左肘 | 五组左肘范围 | 左肘跨度 | 参考右肘 | 五组右肘范围 | 右肘跨度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 130.05 | 114.99–147.05 | 32.06 | 157.66 | 152.18–164.65 | 12.48 |
| 169 | 87.73 | 84.98–89.10 | 4.12 | 131.77 | 126.95–140.92 | 13.97 |
| 327 | 148.90 | 148.90–154.08 | 5.18 | 153.36 | 152.06–158.31 | 6.25 |

例如第 60 帧左肘：保持根深度不变，焦距假设从 800 到 1600 时为 `118.56° → 140.56°`；保持焦距 1145 不变，根深度假设从 3 到 8 时为 `147.05° → 114.99°`。这说明**去髋中心不会消除未知内参和根深度对角度的影响**。它不能用于判断球员动作好坏。

参考情景中，三个稀疏采样帧的右前臂长度为模型相对值 `0.1422–0.3087`，`(最大值−最小值)/平均值≈73.16%`。这是待调查的一致性信号；不能解释成人体前臂真实变长，也不能由三个帧推断连续动作的稳定性。模型深度、二维定位及参数假设都可能影响这个结果。

全部 15 份恢复的最大反投影→投影往返残差为约 `2.84 × 10⁻¹⁴` 像素。该检查验证实现中的算术自洽：恢复和投影使用同一组参数，错误的相机假设同样可能给出接近零的残差。**往返自洽不验证深度、角度或人体尺度准确。**

## 验证与后续边界

相机研究模块的 18 项自动化测试通过，覆盖旋转姿态下的三分量保留、角度/骨长不变性、投影往返、根中心、轴方向、缺失关节、非正深度、坏参数、维度/哈希/帧时间和授权校验，以及独立代理 PTS 对照、拒绝覆盖与并发占用。离线预览另有 12 项测试通过，覆盖完整来源链、XYZ 重算核对、统一尺度、HTML 注入防护、原子发布及并发写入保护。这些测试验证代码契约；真实视频精度仍须另行评估。

本次只有一段本机视频的三个采样帧，没有独立 3D 真值、相机标定或左右/近远的人工标注。后续已完成相同帧的 MMPose-L 推理及原始数值对照，见 [模型实测记录](pose3d-model-evaluation.md)；完整挥拍窗口、跨视角和遮挡评估仍未完成。当前结果不输出实际身高、米制走位/速度、肌肉受力或训练优劣结论；不替换产品原有五项 2D 指标。生产 3D provider 继续关闭，研究成功不构成 Gate 通过。

## 本次证据指纹

以下指纹来自上述实际报告和配置，便于本机复核；各研究 NPZ 和输入授权的具体 hash 继续保存在 JSON 中。

| 证据 | SHA-256 |
| --- | --- |
| `camera-rgb-audited-03/camera-research.json` | `e63902bd28c9253eee9a27a16d80ab450a2b7cc369a12471a21ec16ffd62f41c` |
| 相机假设配置 | `412e1c184f51d8047c89fba152875850f21266fd12a66f67836c21dfbf59774f` |
| RGB 探针报告 | `667df55fc80d6c6b26a663da524398840a9cee511da2506b97dba0ed07ff500c` |
| ONNX 权重 | `4a289c0e99d47eb595e99679d9d4a2d1def1b4241f9adcbafba44b9ff585ebcd` |
