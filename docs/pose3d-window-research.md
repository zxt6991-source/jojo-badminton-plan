# 连续窗口 A/B 本机研究

更新：2026-11-20；实测于 2026-11-19。两条真实模型路径已各完成同一授权视频的 **183 帧**连续采样，均无处理失败；在同一显式相机假设下完成逐帧对照和窗口内变化统计。结果仍有明显肘角跳变与骨长波动，不能根据执行成功或均值相近认定三维可靠。生产 `POSE3D_ENABLED=0` 与适配器阻断保持不变。

## 输入与采样范围

原视频 id 为 `6326ca19-b2b8-404b-8b35-1fde7f2636c1`，使用已有分析 `d862698fd0ceb0b3141a463b6ccd945e` 的 480×854、448 帧代理、采集时间戳及同帧 2D bbox。用户已授权本机内部研发验证；本轮没有外传视频。另一段授权视频的 2D 输入被质量门控拒绝，未用于本轮。

窗口配置为 `data/pose3d-research/windows-spec-01.json`，所有窗口保持 `humanVerified:false`。围绕已有 2D 事件候选取前后约一秒，只是研发采样；尚未确认覆盖完整引拍、击球和随挥，也不用于设置生产 `preSec/postSec`。

| 窗口 | 零起始帧范围（含端点） | 代理时间范围 | 帧数 |
| --- | --- | --- | ---: |
| candidate-060 | 30–90 | 1.000–3.000 秒 | 61 |
| candidate-169 | 139–199 | 4.633333206–6.633333206 秒 | 61 |
| candidate-327 | 297–357 | 9.899999619–11.899999619 秒 | 61 |

计算使用配置中的完整浮点值和实际 `timestamps.f32`，表格只用于阅读。实际代理 PTS 已独立复核；这不能证明原片采集帧率或是否变速。三个窗口来自同一视频，只计一个来源，不能满足 Gate A 的三段独立场景要求。

## 实现与证据保护

- `scripts/pose3d_window_spec.py` 验证最多 3 窗、每窗最多 3 秒且 90 帧、总计最多 240 帧；时间范围含端点，拒绝重叠、乱序、越界及被冒充为人工确认的研究选窗。
- `scripts/pose3d_window_probe.py` 在各 runtime 原有环境中只加载一次模型，顺序读取全部代理帧，选中帧才推理。不保留整个视频的图像，不借用上一帧填补缺失。没有改动原有限帧探针的 10 帧上限。
- bbox 不可用、推理异常或原始数组写入失败均保留具体帧与原因，后续帧继续；未处理帧也保留在分母。窗口外不推理，原始 NPZ 和报告置于私有研究目录。新输出目录独占创建，已有目录不覆盖。
- `scripts/pose3d_window_evaluate.py` 重验授权、视频/时间戳/2D/权重/原始 NPZ hash、实际 PTS、两路裁剪框和窗口原始配置；有成功输出时核验对应解码器。逐帧失败及缺失角度输出明确原因和 `null`；只在同窗连续帧间计算变化，不跨缺失帧或窗口边界。

这套入口产生研究证据，不生成产品 `pose3d.f32`，不启用生产适配器。

## 实测结果

顺序运行 A（RTMLib / RTMW3D-X ONNX），然后 B（官方 MMPose / RTMW3D-L PyTorch），避免两模型互相争用 CPU。两路均复用既有 2D 框，不运行额外检测器。A 使用 CPUExecutionProvider；B 使用独立环境和 4 个 PyTorch 线程。

| 实测项 | A | B |
| --- | ---: | ---: |
| 成功 / 选择帧数 | 183 / 183 | 183 / 183 |
| 处理失败帧 | 0 | 0 |
| 条件相机恢复可计算帧 | 183 | 183 |
| 窗口内相邻帧对 | 180 | 180 |
| 本次任务总耗时 | 13.3641 秒 | 20.8695 秒 |
| 累计模型调用耗时 | 11.4636 秒 | 13.3486 秒 |
| 进程峰值 RSS | 1,188,904,960 字节 | 997,621,760 字节 |

“成功”只表示原始数据通过结构与来源核验，使用 `score > 0` 表示存在解码点；这些分数尚未校准为深度置信度。峰值内存是进程生命周期 RSS；任务总耗时含校验、加载、完整解码和中间落盘，排除最终报告写入。两个模型规格、环境和线程配置不同，单次数字不是公平性能基准。逐窗与计时细节见 [资源实测](pose3d-cost-report.md)。

A 的 Z 统一按实际官方 codec 分母 144 重算；两路深度差先去双髋中心。两者 XYZ 均使用显式 `reference-prior`：`fx=fy=1145`、主点 `(240,427)`、根深度先验 `5.14388`。这些是研究假设，未标定；不使用 B 官方默认主点下的 XYZ 与 A 直接比较。长度保持模型相对单位，角度是该假设下的计算值，非已验证的人体测量。

| 窗口 | A/B 图像 XY 平均距离（px） | 去根相对 Z 平均绝对差（非米） | 左肘相邻角度最大变化 A / B |
| --- | ---: | ---: | ---: |
| candidate-060 | 3.0365 | 0.119713 | 112.23° / 135.96° |
| candidate-169 | 3.4933 | 0.124303 | 102.99° / 146.68° |
| candidate-327 | 1.7626 | 0.143372 | 75.43° / 92.31° |

每窗有 61×17=1,037 个共同身体点，总计 3,111 点，图像 XY 平均距离为 `2.7641 px`，去根相对 Z 平均绝对差为 `0.129129`。这些均为预测间分歧，不是准确率或真实三维误差。

需要优先结合原片复核的例子：B 第 156→157 帧左肘计算值变化 `146.68°`；A 第 356→357 帧右肘变化 `117.62°`。第二窗左前臂长度的标准差/均值分别为 A `38.38%`、B `62.43%`。这些现象说明单看处理成功率或稀疏帧不足以评估连续播放。波动原因可能涉及预测、遮挡、裁剪和相机假设；尚无独立标注，不能据此判断球员动作或直接指定模型胜者。

已生成 [双肘原始曲线](../data/pose3d-research/window-elbow-curves-01.png)，三个窗口分别绘制，时间轴采用实际代理 PTS。没有加滤波、裁去异常或把模型结果转成人工标签。相机参数敏感性见 [相机恢复研究](pose3d-camera-research.md)。

## 产物与复现

私有证据均在被 Git 忽略的 `data/pose3d-research/`，不由公共数据路由发布：

| 产物 | SHA-256 |
| --- | --- |
| `windows-spec-01.json` | `4eb856fc79dca46019b3a1fd62c9352f8e1dff987f4372ae225a3be9935d2f74` |
| `windows-rtmlib-01/report.json` | `2f45a8cdcaab5315b98cab068740f9f498fbadc64ccae2779480fcf541d53ce0` |
| `windows-mmpose-01/report.json` | `83457da2210bb3f801078291f308d6bbf9b6564baf1818d29a5dca808fd87474` |
| `window-comparison-01.json` | `a7951d79553fd9d6801efe96256ebaaf1d50aa08354a9d4780dd80ebae26392d` |

下面命令在项目根目录执行，并使用尚不存在的 `02` 输出路径；已存在时须换新编号，不能删除证据后重跑。同一次对照应顺序运行 A/B。模型和依赖均已在本机，本入口不下载。

```bash
.venv/bin/python scripts/pose3d_window_probe.py \
  --runtime rtmlib-onnx \
  --input data/results/d862698fd0ceb0b3141a463b6ccd945e/proxy.mp4 \
  --timestamps data/results/d862698fd0ceb0b3141a463b6ccd945e/timestamps.f32 \
  --keypoints2d data/results/d862698fd0ceb0b3141a463b6ccd945e/keypoints.f32 \
  --authorization-evidence data/pose3d-research/authorization.json \
  --window-spec data/pose3d-research/windows-spec-01.json \
  --pose-model data/models/pose3d-research/rtmw3d-x.onnx \
  --color-order rgb --device cpu \
  --output-dir data/pose3d-research/windows-rtmlib-02

.venv-pose3d-b/bin/python scripts/pose3d_window_probe.py \
  --runtime mmpose \
  --input data/results/d862698fd0ceb0b3141a463b6ccd945e/proxy.mp4 \
  --timestamps data/results/d862698fd0ceb0b3141a463b6ccd945e/timestamps.f32 \
  --keypoints2d data/results/d862698fd0ceb0b3141a463b6ccd945e/keypoints.f32 \
  --authorization-evidence data/pose3d-research/authorization.json \
  --window-spec data/pose3d-research/windows-spec-01.json \
  --source-root data/pose3d-research/sources/mmpose-1.3.2 \
  --config data/pose3d-research/sources/mmpose-1.3.2/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py \
  --checkpoint data/models/pose3d-research/rtmw3d-l.pth --threads 4 \
  --output-dir data/pose3d-research/windows-mmpose-02

.venv/bin/python scripts/pose3d_window_evaluate.py \
  data/pose3d-research/windows-rtmlib-02/report.json \
  data/pose3d-research/windows-mmpose-02/report.json \
  --camera-config data/pose3d-research/camera-assumptions.json \
  --scenario reference-prior \
  --output data/pose3d-research/window-comparison-02.json
```

## 验证与剩余证据

`npm run test:pose3d-evaluation` **165 项通过**，日志为 `output/pose3d-tests-final-2026-11-19.log`。其中新增窗口规范/执行器 29 项、评估器 9 项，覆盖不等间隔时间、183 帧完整分母、一路零成功、缺髋/缺角度、禁止跨缺失连接、写盘中断、来源篡改、窗口配置绑定、流式读取及输出保护。测试中的合成数值只用于验证实现。

本轮仍缺 ≥3 段独立授权场景及真人左右/朝向/身份/近远核对，正式清单保持为空。下一步按 [素材与人工标注规范](pose3d-annotation-guide.md) 补证据，结合重复成本实测与许可初审再决定 runtime。Gate A 不要求动捕真值，人工序数评估可基于可辨认的图像关系开展。完整验证集、产品价值与 Gate B/C 的要求见 [实施记录](pose3d-implementation.md)。
