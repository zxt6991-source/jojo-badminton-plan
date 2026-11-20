# RTMW3D 坐标与接口审核

初次审核：2026-11-16；研究进展更新：2026-11-17。范围：本地 `rtmlib==0.0.16` 与 OpenMMLab 官方源码。本文记录源码契约及人工 SimCC 实验；后续已在实际 ONNX 输出中复现解码差异，并完成显式相机假设下的研究恢复，见 [本机报告说明](pose3d-camera-research.md)。**不代表三维精度验证、模型 A/B 通过或生产适配完成**，生产模型适配器仍保持阻断。

## 可落地结论

1. `RTMPose3d` 首个返回值的 XY 是裁剪输入像素，Z 已单独解码；不能直接作为同尺度 XYZ。
2. 官方同尺度相机坐标恢复公式存在，但需要相机内参和根深度。未知参数只能显式采用研究假设，不能宣称标定米制。
3. 本地 RTMLib 的 Z 解码分母与官方 L/X 配置不一致，已用人工数值复现。研究探针应保留原始 SimCC 并独立按实际 Z 轴长度解码。
4. 官方 demo 主动采用 `mode='vis'`，显示的是 SimCC 空间。不能将 demo 的视觉效果当成相机 XYZ 转换验证。

## 1. 本地 API 与返回值

本地文件：`.venv/lib/python3.11/site-packages/rtmlib/tools/pose_estimation/rtmpose3d.py`。

```python
RTMPose3d(onnx_model, model_input_size=(288, 384),
          backend='onnxruntime', device='cpu', z_range=None)
model(image, bboxes=[[x1, y1, x2, y2]])
# -> keypoints, scores, keypoints_simcc, keypoints_2d
```

| 返回值 | 形状 | 本地实现中的含义 |
| --- | --- | --- |
| `keypoints` | `[N,133,3]` | 裁剪输入 XY 像素 + 单独反归一化的相对 Z；不是统一尺度 XYZ |
| `scores` | `[N,133]` | X/Y 峰值的最小值；不含 Z 峰值，不能称为深度置信度 |
| `keypoints_simcc` | `[N,133,3]` | 三轴 argmax 除以 `simcc_split_ratio=2`，Z 仍为编码坐标 |
| `keypoints_2d` | `[N,133,2]` | 逆裁剪后的原图像素 XY |

`Wholebody3d(...).__call__(image)` 先调用 YOLOX，再把所有检测框交给 `RTMPose3d`，返回上述四项。目标球员已知时直接调用 `RTMPose3d` 并提供 bbox，可避免默认整图框或重新盲选目标。`bboxes=[]` 会退回整图框。

裁剪处理依次为 bbox 中心/尺寸、`padding=1.25`、按 `288/384` 扩展到目标宽高比、零旋转仿射变换。必须保留**宽高比修正之后**的 `center/scale`。本地 `preprocess` 和 `BaseTool.inference` 未执行 BGR/RGB 交换；官方模型配置启用 `bgr_to_rgb=True`。当前 ONNX 导出是否已吸收颜色转换，需要导出元数据或与原版逐帧比对确认，不能凭 demo 使用 OpenCV 推断。

## 2. 官方解码与本地差异

官方 `SimCC3DLabel.decode()` 将三轴峰值除以分桶倍率，再单独恢复深度；`encode()` 的训练目标是相机 Z 减去双髋平均 Z。缺少 3D 标签时使用的根深度先验为 `5.14388`；默认 `z_range=2.1744869`。这些是模型/训练配置常数，不是上传视频中的已知相机参数。[官方 codec](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose3d/rtmpose3d/simcc_3d_label.py#L86-L107)

记 argmax 为 `(a_x,a_y,a_z)`、分桶倍率 `r`、深度输入长度 `D`：

```text
u_crop = a_x / r
v_crop = a_y / r
q_z = a_z / r
z_relative = (q_z / (D / 2) - 1) * z_range
```

官方 L 与 X 的 codec 均采用 `(W,H,D)=(288,384,288)`、`r=2`、`root_index=(11,12)`，所以该配置 Z 分母是 `144`。X 文件名是 `rtmw3d-x_8xb32_cocktail14-384x288.py`，不是 `8xb64`。[官方 X 配置](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose3d/configs/rtmw3d-x_8xb32_cocktail14-384x288.py#L44-L52)、[官方 L 配置](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py#L44-L52)

本地 `RTMPose3d.postprocess()` 却使用二维 `model_input_size[-1]/2`，即 `384/2=192`。在本地未修改的库中调用该方法，向三轴传入人工峰值，结果如下：

| 项目 | 数值 |
| --- | --- |
| 人工 SimCC 输出形状 | `[1,133,576]`、`[1,133,768]`、`[1,133,576]` |
| 三轴峰值索引 | `(288,384,288)` |
| `keypoints_simcc` | `(144,192,144)` |
| 本地 `keypoints` | `(144,192,-0.5436217189)` |
| 同峰值按官方 `D=288` 解码 | `z_relative=0` |

因此对同一 `q_z`，本地 Z 等于 `0.75 * official_z - 0.25 * z_range`。去髋中心只消掉常量项，仍留下深度轴 `0.75` 倍缩放；这不是一个可用统一缩放消除的差异。若把该混合坐标直接送入现有正交坐标矩阵，会改变肢体夹角。

**探针处理建议**：先验证模型实际输出轴长度。若与官方配置一致，从 `keypoints_simcc[...,2]` 重新计算 Z，或从原始 `simcc_z` 峰值解码；记录所用 `D/r/z_range`。不要只修改 `z_range`，也不要把二维 `model_input_size` 改成三维元组，因为预处理按二维尺寸解包。

人工探针仅测试公式，不是模型推理。其最小复现如下：

```python
import numpy as np
from rtmlib import RTMPose3d

m = RTMPose3d.__new__(RTMPose3d)  # 不载入权重，只执行数值后处理
m.model_input_size, m.z_range = (288, 384), 2.1744869
outputs = [np.zeros((1, 133, size), np.float32) for size in (576, 768, 576)]
for output, peak in zip(outputs, (288, 384, 288)):
    output[:, :, peak] = .9
k, scores, q, uv = m.postprocess(outputs, np.array([144.,192.]), np.array([288.,384.]))
print(k[0,0], q[0,0], (q[0,0,2]/144-1)*m.z_range)
```

## 3. 同尺度相机坐标恢复

官方入口是 `TopdownPoseEstimator3D.add_pred_to_datasample()`，必须使用 `mode='3d'`。它先恢复原图像素，再以相机内参和绝对根深度做反投影；`mode='vis'` 则返回另一套 `keypoints_simcc`。[官方 estimator](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose3d/rtmpose3d/pose_estimator.py#L40-L82)

设 `S=(s_x,s_y)` 为扩展裁剪尺寸、`C=(b_x,b_y)` 为裁剪中心、`f=(f_x,f_y)` 与 `c=(c_x,c_y)` 为原图内参：

```text
u = u_crop / W * s_x + b_x - s_x / 2
v = v_crop / H * s_y + b_y - s_y / 2
Z = z_relative + root_z
X = (u - c_x) / f_x * Z
Y = (v - c_y) / f_y * Z
P_root = (P_leftHip + P_rightHip) / 2
P_relative = P - P_root
```

这里 `(X,Y,Z)` 共享深度目标的单位。其反投影依赖 `u*Z/v*Z`，**不能用一个固定 3×3 矩阵将混合的 crop-XY/decoded-Z 直接恢复**。先做反投影，再用正交矩阵改变轴方向。

官方 fallback 焦距是 `(1145.04940459,1143.78109572)`，主点代码为 `np.array(data_sample.ori_shape)/2`。不要把它们当作手机相机标定值，也不要默默照抄主点：`ori_shape` 常为 `(H,W)`，而像素坐标是 `(u,v)`，非方形图的顺序需要单独验证。研究中如采用图像中心，应显式声明 `(W_image/2,H_image/2)` 是自设假设，并记录与官方 fallback 的差别。[官方 fallback](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose3d/rtmpose3d/pose_estimator.py#L56-L71)

即使最终去髋中心，`f` 与 `root_z` 仍影响 XY 与 Z 的相对尺度；不能称“去中心后不需要相机参数”。未知内参/根深度只能输出明确假设下的研究结果，保持 `unit='model-relative'`，不提供实际身高、距离或绝对速度。

## 4. 轴方向、近远与 demo 陷阱

从上述针孔反投影可推导：原相机 X 随图像向右增大，Y 随图像向下增大，正深度 Z 沿镜头朝场景方向增大。相对 Z 为正表示比根关节更远，为负表示更近。这是**对官方投影公式的解释**；仍应通过有已知前后关系的真实画面验证模型预测是否遵守该顺序。

项目字符串 `x-right-y-up-z-forward` 本身没有明确“forward”是朝相机还是远离相机。若项目决定正 Z 指向观看者/相机，则采用 `diag(1,-1,-1)`；若正 Z 指向场景深处则采用 `diag(1,-1,1)`。必须在 provenance 中用文字写清正 Z 的近远含义，再固定矩阵和测试，不能只凭肉眼选负号。当前审核没有替产品作该命名决策。

官方 demo 实际位于 `projects/rtmpose3d/demo/body3d_img2pose_demo.py`；项目根目录同名 URL 返回 404。它设置 `mode='vis'`，随后对 SimCC 三轴取负、换序，并可将最低点平移到地面。这是绘图操作，不是恢复真实相机坐标或全局地面轨迹。[官方 demo 的 mode](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose3d/demo/body3d_img2pose_demo.py#L268)、[绘图换轴](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose3d/demo/body3d_img2pose_demo.py#L195-L213)

H3WB 加载器确实使用 `camera_3d`，并在传入模型前除以 `1000`；这说明其训练标签经过单位缩放，**不证明未知手机视频恢复出的数值是经过标定的米**。[官方 H3WB 数据读取](https://github.com/open-mmlab/mmpose/blob/main/mmpose/datasets/datasets/wholebody3d/h3wb_dataset.py#L121-L135)、[标签缩放](https://github.com/open-mmlab/mmpose/blob/main/mmpose/datasets/datasets/wholebody3d/h3wb_dataset.py#L177-L212)

## 5. 研究探针需要记录的证据

- 原始三轴输出名称与 shape、每轴峰值、保留的 SimCC 坐标；确认当前 ONNX 与官方配置的 `D=288` 一致。
- ONNX hash、来源、RTMLib/ORT 版本及实际执行 provider；不能用文件名确认导出等价性。
- 目标 bbox、padding、仿射后的 `center/scale`、原图尺寸、输入色序和归一化配置。
- 解码倍率、Z 范围、根深度来源，以及内参是标定值、训练先验还是显式研究假设。
- 双髋去中心前后数据、正 Z 的近远说明、最终轴矩阵。
- 原始置信度仅由 XY 峰值产生这一限制；不能将相同值解释为经过校准的深度可靠性。
- 同帧 A/B、已知近远顺序与左右侧核对；单帧成功不能替代三段授权羽毛球视频验收。

目前尚未完成：当前下载 ONNX 的导出配置/色序等价性、MMPose 同权重实跑对照、相机参数有效性、深度序数人工核对。**这些缺口未关闭前，生产 provider 维持拒绝发布；研究输出不能进入用户结果。**

## 6. 本地源码指纹

以下 SHA-256 是本次实际读取的 `rtmlib==0.0.16` 源码；路径均相对 `.venv/lib/python3.11/site-packages/rtmlib/`。官方链接目前指向 `main`，继续研究时应再固定上游 commit，避免分支变动改变复现语义。

| 文件 | SHA-256 |
| --- | --- |
| `tools/pose_estimation/rtmpose3d.py` | `94b143e4760173b056ef85dd95baaca89ca084756df0ecca88aa5735dc4daabf` |
| `tools/solution/wholebody3d.py` | `bc8d5dc1e4402861480d8dfbb01b1375c7cc2dae2d0d307b78dee9c699f3c0f1` |
| `tools/pose_estimation/pre_processings.py` | `2bd84ad2300c3bb43a831e18aff30588c6f8e570d41d492826cf540cfe8c0fcb` |
| `tools/pose_estimation/post_processings.py` | `6c372b34c0021a808c28cbc49206b1908e4cf9c820e859fde18c22a5362bcf05` |
