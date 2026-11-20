# P0-M：模型接口审计与评估状态

更新：2026-11-20。**路径 A（ONNX-X）与路径 B（官方 MMPose-L）均已完成同一视频 3 个连续研究窗口、各 183 帧的真实推理，尚未通过完整模型 A/B 验收或选择生产 runtime。** 两路已在相同显式相机假设下对照连续变化；路径 A 另有多组相机参数敏感性研究。当前两个生产适配器仍拒绝推理；研究执行成功不等于产品已集成可靠的 3D 姿态。

## 路径 A：rtmlib / ONNX

官方当前源码提供高层 `Wholebody3d`，其返回四元组 `keypoints, scores, keypoints_simcc, keypoints_2d`；底层类名为 `RTMPose3d`。高层默认检测器是 YOLOX，姿态模型配置为 RTMW3D-x、输入宽高 `(288,384)`。默认权重 URL 指向第三方 Soykaf 的 Hugging Face 仓库：官方代码引用不是官方签名、权重许可或转换一致性的证明。[Wholebody3d 源码](https://raw.githubusercontent.com/Tau-J/rtmlib/main/rtmlib/tools/solution/wholebody3d.py)、[官方演示](https://raw.githubusercontent.com/Tau-J/rtmlib/main/wholebody3d_demo.py)。

底层 `RTMPose3d(image, bboxes=...)` 可传 XYXY 框。但其 `postprocess` 中，第一输出 XY 保留 SimCC 解码后的裁剪尺度，Z 单独通过 `z_range` 变换；另一个 `keypoints_2d` 才经裁剪的尺度/中心还原图像坐标。因此第一输出不能直接当作同单位三维笛卡尔坐标，更不能仅翻转 Y/Z 后计算关节角。[RTMPose3d 源码](https://raw.githubusercontent.com/Tau-J/rtmlib/main/rtmlib/tools/pose_estimation/rtmpose3d.py)。

初次审计是上游 `main` 的源码检查。后续已核实本机 `rtmlib==0.0.16` 并实际运行下述固定模型；研究探针直接使用同帧 2D bbox，不运行新检测器。完整坐标恢复、左右与深度真值、导出一致性、空检测行为仍未完成验证。未发现可据以确认 L 档 ONNX 的本地资料，不宣称不存在。

## 路径 A 的本机实测（RAW，不进入产品）

用户明确授权项目中的两段既有视频用于本机内部 3D 研发验证，不上传。授权记录绑定原片和代理视频的 SHA-256，保存在 `data/pose3d-research/authorization.json`。准备工具选出一段最新成功的 2D 分析：448 帧、30 fps、480×854；另一段未通过 2D 质检，未生成可用探针输入。同一视频的旧分析不重复计数。现有 2D 视角预测为 oblique / 0.55，低于门控阈值，**用户展示仍为 blocked**。

模型为 rtmlib 源码引用的 Soykaf RTMW3D-x ONNX，固定仓库 revision `f5f9d237c9e4f327f0cd02e00a98444168800ac4`。原站直连超时后从镜像取回同一文件；字节数 `369330857`，SHA-256 `4a289c0e99d47eb595e99679d9d4a2d1def1b4241f9adcbafba44b9ff585ebcd`，均匹配原站记录。权重仅保存在忽略目录 `data/models/pose3d-research/`，未加入产品分发。[固定文件与原站 hash](https://huggingface.co/Soykaf/RTMW3D-x/blob/f5f9d237c9e4f327f0cd02e00a98444168800ac4/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx)

环境：Darwin 25.6.0 / arm64、Python 3.11、本机既有 rtmlib 0.0.16、ONNX Runtime 1.32.0、NumPy 2.4.6、OpenCV 4.14.0.94，实际 `CPUExecutionProvider`。此次没有升级原 2D 依赖；`worker/requirements-pose3d.txt` 记录已使用的主要版本供独立环境复现，尚未重新安装验证一个全新环境。

同一代理的采集帧 60、169、327 分别用 BGR 与 RGB 输入运行，共 6 次真实模型调用。两组都返回完整四元组；模型实际输出轴长度为 `[1,133,576] / [1,133,768] / [1,133,576]`。下表的偏差是与已有 2D 预测的平均图像距离，**不是与真值比较，更不能证明 Z 正确**。

| 采集帧 | BGR 耗时（秒） | RGB 耗时（秒） | BGR / 原 2D 偏差（像素） | RGB / 原 2D 偏差（像素） |
| --- | ---: | ---: | ---: | ---: |
| 60 | 0.2773 | 0.2513 | 5.3896 | 3.6641 |
| 169 | 0.0721 | 0.0732 | 6.0333 | 3.6199 |
| 327 | 0.0578 | 0.0580 | 8.0399 | 7.9403 |

每帧两组均有 17 个身体共同点达到探针 0.4 阈值。计时包含预处理、ONNX 与后处理；不是实时产品承诺。两进程峰值 RSS 约 1.159 GB / 1.155 GB，包含整个进程，不能当模型单独内存。首帧较慢，样本仅三帧，未测稳定吞吐、GPU、完整窗口或每次分析成本。RGB 在这些帧上更接近现有 2D，但这不足以选择生产预处理或证明准确度更高。

严格重验同模型/同输入/同帧/同bbox后的比较报告位于 `data/pose3d-research/probes/rtmw3d-preprocessing-comparison-01.json`。全部 51/51 个身体点共同可比较，汇总图像偏差为 BGR `6.48757 px` / RGB `5.07478 px`。报告重新读取原始数组计算，未直接信任已有均值；`winner:null`、`releaseReady:false`。原始分数不含 Z 置信度，SimCC Z 差值也不等于深度误差。

实际 SimCC 进一步复现本地解码差异：本机库的 Z 分母为 192，官方 codec 对该实际 Z 输出长度应为 144。独立解码审计从保存的 `keypoints_simcc` 重算相对 Z，不生成产品 `pose3d.f32`。后续新增相机研究脚本，以显式参数完成针孔反投影、双髋去中心和显示轴变换；结果只在所提供参数成立时有几何意义。详见 [坐标审核及公式](pose3d-coordinate-audit.md) 与 [相机恢复研究](pose3d-camera-research.md)。真实相机内参、根深度、色序和 ONNX 与原 PyTorch 的等价性仍需验证。

本地证据目录：`data/pose3d-research/probes/rtmw3d-bgr-01/` 和 `rtmw3d-rgb-01/`。各有 `report.json`、逐帧 `raw.npz` / `overlay.png`、`decode-audit/audit.json`。保存图像仅用于本机核查。每个报告记录模型/输入/授权/原始产物 hash、依赖、采集时间和 bbox。

2026-11-17 增加实际代理 PTS 检查后，使用相同 RGB 设置重跑 3 帧，报告为 `data/pose3d-research/probes/rtmw3d-rgb-clock-02/report.json`。448 帧的代理 PTS 与既有时间戳最大偏差 `3.9691e-7 s`，通过 `1e-4 s` 容差。只检查帧数不足以排除时间轴缩放/偏移，探针现已拒绝这些输入；此核对仍**不能证明原片没有剪辑或实际采集帧率已知**。

在该探针上运行 5 组明确标为 `assumed-research` 的相机参数，最新报告位于 `data/pose3d-research/camera-rgb-audited-03/camera-research.json`。3 帧×5 情景均生成研究 XYZ，独立复算与初版 15 份数组一致。第 60 帧左肘计算角度随焦距/根深度假设从 `114.99°` 到 `147.05°`，跨度 `32.06°`；这支持继续要求参数证据，不能把某一个假设下的角度提供为用户测量。最大投影往返误差 `2.84e-14 px` 只验证公式算术，不衡量真实姿态精度。

复现使用新空输出目录，避免覆盖证据：

```bash
python3 scripts/pose3d_prepare.py --data-dir data \
  --authorization-evidence data/pose3d-research/authorization.json \
  --output <new-inputs-draft.json>
# 以下路径取 inputs-draft.json 对应 input 的实际字段；帧为既有2D候选，未人工确认。
.venv/bin/python scripts/pose3d_probe.py \
  --input <proxy.mp4> --input-kind capture-proxy --timestamps <timestamps.f32> \
  --keypoints2d <keypoints.f32> --pose-model data/models/pose3d-research/rtmw3d-x.onnx \
  --frames 60,169,327 --color-order bgr \
  --authorization-evidence data/pose3d-research/authorization.json --output-dir <new-bgr-run>
# 同参数改 --color-order rgb，写入另一个新目录。
.venv/bin/python scripts/pose3d_decode_audit.py <new-bgr-run>/report.json --output-dir <new-z-audit>
.venv/bin/python scripts/pose3d_probe_compare.py <new-bgr-run>/report.json <new-rgb-run>/report.json --output <comparison.json>
```

两种色序比较只是**同一模型的预处理诊断**，不是文档要求的 rtmlib-X / MMPose-L 模型 A/B。授权仅覆盖本机验证；缺少三段场景覆盖、人工标注、校准参数和产品价值研究，Gate A/B 仍未通过。

此前 RAW 阶段软件验证：研究工具 61 项、Worker 58 项全部通过；本轮增加时钟、相机恢复和输出保护测试，最新数量见 [实施记录](pose3d-implementation.md)。测试中的合成夹具不计入真实模型调用或准确率证据。

## 路径 B：MMPose / PyTorch

官方 RTMPose3D 项目列出了 L/X 配置与 checkpoint；其 H3WB 表中数值分别为 0.056/0.057。它们是该项目基准结果，不能解释成本项目羽毛球视频精度，也不能据此完成选型。[官方项目与示例](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d)。

已下载官方 L checkpoint 至 `data/models/pose3d-research/rtmw3d-l.pth`：`230771611` 字节，SHA-256 为 `794dbc78b04a43d81781f8ab0eba5b24f3dd5d71aaf6ae253940424159fb81ed`；MD5 与下载响应头一致。来源为 [OpenMMLab 固定 checkpoint](https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth)。这只验证来源与完整性，不构成许可或精度通过。

官方 MMPose v1.3.2 归档已保存到 `data/pose3d-research/sources/`，`16608638` 字节，SHA-256 为 `fcf1e7ac3f5bbb6f3f98849bb579d63184cb3c7ba944ee1a894dc52084d901b3`。独立 `.venv-pose3d-b` 的实际组合为 Python 3.11、torch 2.2.2、torchvision 0.17.2、NumPy 1.26.4、mmengine 0.10.5、MMCV 2.1.0、mmdet 3.3.0、MMPose 1.3.2。最初的 mmcv-lite 缺少 `mmcv._ext`，不能导入官方完整 API；已从官方 v2.1.0 源码在本机编译 77 个 CPU 对象，生成并安装原生轮子，卸载 lite，实际 API 与原生 NMS 调用均通过。构建轮子 SHA-256 为 `ff965c128e42b5682e7b620aa4c878621691f43f4d8a2dfe25022718f3936486`。未使用虚假算子替代，也未改动原 2D 环境。完整步骤见 [MMPose 环境记录](pose3d-mmpose-environment.md)。

### 路径 B 的本机实测

`scripts/pose3d_mmpose_probe.py` 实际调用官方 `init_model` / `inference_topdown`，直接传同帧 2D bbox，不运行额外检测器。输入为 BGR，官方预处理执行 RGB 交换；保持配置 `flip_test=false`，显式关闭 backbone 的远程预训练初始化，使用完整本地 checkpoint。以只读复制观察官方 `head.decode` 的返回值，保存相机反投影前的 `decoded_keypoints`、`keypoints_simcc`、分数，以及官方最终相机坐标和图像 XY；没有替换模型计算值。

首次成功报告为 `data/pose3d-research/probes/mmpose-l-01/report.json`。下表同样只与既有 2D 预测比较，不是准确率：

| 采样帧 | MMPose-L 推理秒数 | 共同 body 点 | 与原 2D 平均图像偏差（像素） |
| --- | ---: | ---: | ---: |
| 60 | 0.1087 | 17 | 4.6916 |
| 169 | 0.0727 | 17 | 4.4139 |
| 327 | 0.0702 | 17 | 7.9676 |

首次运行记录的模型加载阶段约 `14.096 s`，包含库导入和初始化，并触发首次字体缓存；探针总计 `15.050 s`，峰值进程 RSS `1018609664` 字节。推理时使用 4 个 PyTorch CPU 线程。不能用该加载时间与 A 的 ONNX 会话加载时间直接得出冷启动性能排序；完整窗口和公平资源配置仍未测量。

随后将实际 `model.head.decoder` 属性加入来源记录，在新目录 `mmpose-l-02/` 重跑相同 3 帧；原 `01` 保留。新报告确认 `codecInputSize=[288,384,288]`、`simccSplitRatio=2`、`zRange=2.1744869`、`useDark=false`、`rootIndices=[11,12]`、Z 分母 `144`。这是从实际对象读取的配置；不是事后向旧报告补写。报告 SHA-256 为 `1836c749f38e64519cdc0d8283b547c74b543c4cc6fbb599e8b3f65dce3798de`。

官方 `mode='3d'` 输出使用默认焦距、主点 `np.array(ori_shape)/2` 与根深度先验 `5.14388`。该主点按 `(H,W)` 顺序，与研究恢复中显式 `(W/2,H/2)` 假设不同。保留其原始结果供核查，不将默认相机 XYZ 当成视频已标定的物理坐标，也不直接拿它与 A 的混合坐标计算三维距离。

独立只读复核确认：B 的 `keypoints_simcc` 已是 argmax/2；按分母 144 重算 Z 与保存值误差不超过 `1.26e-7`。三帧官方相机 XYZ 与其实际默认主点 `[427,240]`、焦距和根深度公式重算一致。两次 B 运行的全部原始数组数值一致。这里只确认数据和公式自洽，没有独立三维真值。

### 同帧 A/B 原始数值对照

`scripts/pose3d_runtime_probe_compare.py` 已实际比较 A 的 `rtmw3d-rgb-clock-02` 与 B 的 `mmpose-l-02`，报告保存在 `data/pose3d-research/runtime-probe-comparison-01.json`。工具重新核对授权、代理/时间戳/2D/权重/原始 NPZ hash、B 配置与源码、实际 PTS、逐帧 bbox；先将 A 的 Z 修正为官方分母 144，再分别减去双髋平均 Z。不直接比较不同假设下的相机 XYZ。

| 对照量 | 真实 3 帧结果 | 含义 |
| --- | ---: | --- |
| 共同已解码身体点 | 51 / 51 | 原始 score > 0，未做跨模型置信度校准 |
| 两模型图像 XY 平均距离 | 2.3935 px | 两组预测的差异，不是真值误差 |
| 去髋中心相对 Z 平均绝对差 | 0.081425 | 模型相对深度量，不是米 |
| 相对根的深度符号相同 | 39 / 50 | 另有 1 个平局；相互一致不代表正确 |
| 左右对应关节对的近远符号相同 | 11 / 14 | 另有 4 个平局；不是左右交换率 |
| 与原 2D 预测平均图像距离 | A 5.0748 px / B 5.6911 px | 同一组 51 点；原 2D 不是人工真值 |

```bash
.venv/bin/python scripts/pose3d_runtime_probe_compare.py \
  data/pose3d-research/probes/rtmw3d-rgb-clock-02/report.json \
  data/pose3d-research/probes/mmpose-l-02/report.json \
  --output <new-runtime-comparison.json>
```

报告固定 `accuracyMeasured=false`、`metricDepthMeasured=false`、`cameraXYZCompared=false`、`releaseReady=false`。同帧对照同时改变模型规格 X/L、运行时与实现，不是同权重导出等价性测试；目前不足以选定胜者或标注预测哪一侧正确。

## A/B 证据与决策

2026-11-19 已完成后续连续窗口实跑：同一授权视频三段各 61 帧，A/B 各 183 帧，处理失败均为 0。工具保留失败帧分母，在同一显式相机假设下统计骨长与相邻角度变化；部分相邻肘角变化超过 100°，不能从稀疏帧均值或处理成功率推断可连续播放。原始报告、全部数值、曲线与命令见 [连续窗口研究](pose3d-window-research.md)。这仍只是一段视频，未完成人工左右、身份与近远核对。

| 项目 | A | B |
| --- | --- | --- |
| 官方接口调查 | 已查高/低层源码并实跑 | 已实跑官方初始化、bbox 推理和解码/相机返回值 |
| 本地固定依赖、权重 SHA-256 | RAW 探针实际版本与权重 hash 已记录；未验证独立新环境 | 独立环境、官方源码/权重/CPU wheel hash 已记录并实跑 |
| 授权侧面/斜侧/遮挡各一段 | 仅一段具备2D输入；未满足三类场景与人工标注 | 同一段的 3 个连续窗口；场景数量仍不足 |
| 逐帧坐标/左右/目标身份核对 | 183 帧有显式相机假设恢复，人工方向/身份尚未验证 | 同 183 帧官方原始输出及相同相机假设恢复，人工方向/身份尚未验证 |
| 按视角序数准确率/左右交换 | `not_measurable` | `not_measurable` |
| 总/每窗口耗时、RAM/VRAM | 3 个连续研究窗口与进程 RSS 已测；完整任务/GPU/部署成本未测 | 同 3 窗 CPU 耗时与进程 RSS 已测；完整任务/GPU/部署成本未测 |
| 商业权重与数据集许可 | 不确定 | 不确定 |
| 集成结论 | 不启用 | 不启用 |

执行工具与人工标注格式见 [实施记录](pose3d-implementation.md) 与 [标注规范](pose3d-annotation-guide.md)。`pose3d_compare.py` 比较已有标准化、同视频 hash/同帧的 A/B 产物，不执行模型，也不把相互一致当成准确。有限帧原始推理已由 `pose3d_window_probe.py` 扩展到连续研究窗口；标准化端到端流水线仍未接通。两条路径在完整验证中均不合格才进入 MotionBERT 对照，需单独的 H36M 17 点映射和坐标/许可证据。

最重要的复核实验：同一已知姿态使用不同 bbox 缩放/平移，验证恢复出的相对骨长和深度是否一致；左右手主动前后伸展核验 z 符号；相同采集帧与2D叠加核验顺序；交叉/遮挡时核验缺失而非换人。保存原始四元组及全部配置，不能只保存“看起来合理”的动画。
