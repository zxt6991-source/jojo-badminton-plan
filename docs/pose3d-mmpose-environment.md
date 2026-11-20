# MMPose-L 本机研究环境记录

记录日期：2026-11-18。此环境仅用于已授权视频的本机研究，不替换原 `.venv` 或二维 Worker。**完整 MMCV CPU wheel、官方 MMPose API 与真实 `mmcv.ops.nms` 数值调用已验证；RTMW3D-L 已成功处理同一授权视频的 60、169、327 三帧，保存全部原始模型输出。** 这证明本机运行可行，不代表三维精度、完整 A/B POC 或发布验收通过。

## 已实际安装的环境

主机为 macOS arm64，Python 3.11.14，独立环境为 `.venv-pose3d-b/`。

| 包 | 已安装版本 | 说明 |
| --- | --- | --- |
| numpy | 1.26.4 | 与 PyTorch 2.2.2 组合使用 |
| torch / torchvision | 2.2.2 / 0.17.2 | macOS arm64 wheel |
| mmengine | 0.10.5 | 已安装 |
| mmcv | 2.1.0 | 官方源码编译的真实 CPU 算子 wheel；先前不足以导入 API 的 mmcv-lite 已卸载 |
| mmdet | 3.3.0 | 官方模块注册依赖；探针不运行检测器 |
| mmpose | 1.3.2 | 本地官方 release 源码构建并非 editable 安装成功 |
| opencv-python | 4.10.0.84 | 独立环境版本，不修改二维环境 |
| scipy / matplotlib | 1.13.1 / 3.9.2 | 已安装 |
| chumpy / xtcocotools | 0.70 / 1.14.3 | 安装成功不等于所有训练功能可用 |
| Cython / wheel | 3.0.11 / 0.44.0 | 已安装构建工具 |
| setuptools | 80.9.0 | 保留当前已验证版本；MMEngine 导入依赖其提供的 `pkg_resources` |
| json_tricks / munkres | 3.17.3 / 1.1.4 | 已安装 |

这些是实际安装记录，**不是已通过三维模型验收的版本组合**。主要依赖版本记录在 `worker/requirements-pose3d-mmpose.txt`，完整 `pip freeze` 已保存于 `data/pose3d-research/mmpose-environment-freeze.txt`，本机 MMCV 构建记录保存于 `data/pose3d-research/wheels/mmcv-build.json`。

## 固定来源与权重

官方 MMPose 源码来自 [v1.3.2 archive](https://codeload.github.com/open-mmlab/mmpose/tar.gz/refs/tags/v1.3.2)，存放于 `data/pose3d-research/sources/mmpose-v1.3.2.tar.gz`，16,608,638 字节，SHA-256：

```text
fcf1e7ac3f5bbb6f3f98849bb579d63184cb3c7ba944ee1a894dc52084d901b3
```

解压目录为 `data/pose3d-research/sources/mmpose-1.3.2/`。该版本包含 `projects/rtmpose3d`。非 editable 构建产生的 MMPose wheel SHA-256 为 `38726376b419565880b65d77aa3a6125f667f43681abe208fe747dd2e76f8198`。

官方 MMCV 源码来自 [v2.1.0 archive](https://codeload.github.com/open-mmlab/mmcv/tar.gz/refs/tags/v2.1.0)，存放于 `data/pose3d-research/sources/mmcv-v2.1.0.tar.gz`，7,617,305 字节，SHA-256：

```text
4e9d28b42179404aff9ce4749e2cc5a3ad45fe76e3a0621608bcc8e144e917a1
```

RTMW3D-L 权重为官方模型表引用的 `rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth`，本地路径 `data/models/pose3d-research/rtmw3d-l.pth`，230,771,611 字节，SHA-256：

```text
794dbc78b04a43d81781f8ab0eba5b24f3dd5d71aaf6ae253940424159fb81ed
```

完整权重 hash 是本地下载校验记录，前缀与官方文件名一致；不把它解释为许可审计通过。来源元数据保存在对应的本地 `source.json` / `rtmw3d-l-source.json`。

## 安装步骤与实际失败

创建隔离环境并安装基础依赖：

```sh
.venv/bin/python -m venv .venv-pose3d-b
.venv-pose3d-b/bin/python -m pip install \
  numpy==1.26.4 torch==2.2.2 torchvision==0.17.2 \
  mmengine==0.10.5 mmcv-lite==2.1.0 mmdet==3.3.0 \
  opencv-python==4.10.0.84 scipy==1.13.1 matplotlib==3.9.2 \
  setuptools==80.9.0 wheel==0.44.0 json_tricks==3.17.3 munkres==1.1.4
.venv-pose3d-b/bin/python -m pip install --no-build-isolation \
  chumpy==0.70 xtcocotools==1.14.3 Cython==3.0.11
.venv-pose3d-b/bin/python -m pip install --no-deps --no-build-isolation --use-pep517 \
  data/pose3d-research/sources/mmpose-1.3.2
```

实际遇到的失败：

- 一次性安装 `mmpose==1.3.2` 时，旧 `chumpy` 的隔离构建元数据阶段报 `ModuleNotFoundError: No module named 'pip'`。拆分依赖并使用 `--no-build-isolation` 后，chumpy 构建安装成功。
- editable 安装触发 `setup.py develop` 内部再次运行 pip、请求新的构建依赖，受到沙箱 DNS 限制。改为上面的非 editable、显式 PEP 517 本地构建后安装成功。
- 官方 `from mmpose.apis import init_model, inference_topdown` 实际失败：`mmpose.models.heads.EDPoseHead` 无条件导入 `mmcv.ops.MultiScaleDeformableAttention`，进一步报 `ModuleNotFoundError: No module named 'mmcv._ext'`。因此此版本的官方完整 API 不能直接使用纯 `mmcv-lite` 环境，即使 RTMW3D 本身不运行 EDPose。

没有使用伪造的 `mmcv._ext`、占位算子、删改模型数学过程或替换模型输出来绕过此失败。

## 完整 MMCV CPU 构建（已成功）

本机已找到 Apple Clang 和 macOS SDK。实际启动的命令：

```sh
MMCV_WITH_OPS=1 MAX_JOBS=4 .venv-pose3d-b/bin/python -m pip wheel \
  --no-deps --no-build-isolation --use-pep517 \
  data/pose3d-research/sources/mmcv-2.1.0 \
  --wheel-dir data/pose3d-research/wheels
```

启动时 `torch.cuda.is_available()` 与 `torch.backends.mps.is_available()` 均为 `False`，构建走实际 CPU 算子分支。77 个对象编译完成，产出并安装 `mmcv-2.1.0-cp311-cp311-macosx_15_0_arm64.whl`，1,228,463 字节，存放于 `data/pose3d-research/wheels/`，SHA-256：

```text
ff965c128e42b5682e7b620aa4c878621691f43f4d8a2dfe25022718f3936486
```

独立环境中 `mmcv-lite` 已卸载，完整 wheel 已安装。官方 `mmpose.apis` 导入及实际 `mmcv.ops.nms` 数值调用已成功，不是通过占位模块掩盖缺失算子。构建命令未加 `-v`，工具保留了实际执行输出，没有单独的完整逐条编译日志文件。上面的 wheel hash 是本机产物；不同编译工具链不保证生成相同 hash。

```sh
.venv-pose3d-b/bin/python -m pip uninstall -y mmcv-lite
.venv-pose3d-b/bin/python -m pip install --no-deps \
  data/pose3d-research/wheels/mmcv-2.1.0-cp311-cp311-macosx_15_0_arm64.whl
```

真实 L 探针已写入 `data/pose3d-research/probes/mmpose-l-01/`，`report.json` 状态为 `raw_evidence_collected`。三帧分别耗时约 0.108715、0.072730、0.070172 秒；模型载入约 14.09565 秒（包含导入及首次缓存等开销）、总运行约 15.05037 秒、进程峰值 RSS 为 1,018,609,664 字节。单次三帧运行不构成性能基准，也不能把原始 score 当作深度置信度。

同样三帧已再次实跑并保存于 `data/pose3d-research/probes/mmpose-l-02/`；该报告补充从实际 `model.head.decoder` 读取的 `sourceGeometry`，用于核对 codec 输入大小、SimCC 比例、Z 范围、DARK 设置、根节点索引和深度解码分母。原 `mmpose-l-01` 报告保持原样。

`setuptools==80.9.0` 已通过实际环境元数据再次核对。不要在此环境中无记录地升级 setuptools；MMEngine 0.10.5 仍使用 `pkg_resources`，安装更高版本可能破坏已验证的导入路径。

## 已实现的真实模型调用探针

`scripts/pose3d_mmpose_probe.py` 使用官方 `init_model` 和 `inference_topdown`，直接传入原二维骨架推导的同帧 bbox，不加载检测器。它显式清除 backbone 的远程预训练初始化，只接受已有本地配置、完整 `.pth` 权重及绑定视频 hash 的本机授权。

```sh
.venv-pose3d-b/bin/python scripts/pose3d_mmpose_probe.py \
  --input data/results/d862698fd0ceb0b3141a463b6ccd945e/proxy.mp4 \
  --input-kind capture-proxy \
  --timestamps data/results/d862698fd0ceb0b3141a463b6ccd945e/timestamps.f32 \
  --keypoints2d data/results/d862698fd0ceb0b3141a463b6ccd945e/keypoints.f32 \
  --frames 60,169,327 \
  --source-root data/pose3d-research/sources/mmpose-1.3.2 \
  --config data/pose3d-research/sources/mmpose-1.3.2/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py \
  --checkpoint data/models/pose3d-research/rtmw3d-l.pth \
  --authorization-evidence data/pose3d-research/authorization.json \
  --threads 4 \
  --output-dir data/pose3d-research/probes/mmpose-l-cpu-01
```

输出目录须为空。探针核对 ffprobe 真实逐帧 PTS，保存同帧 RAW NPZ、二维对照图、来源、耗时和内存信息。它只读复制官方 `head.decode` 的结果，保留反投影前的解码值，同时另存官方最终相机坐标；不改变模型输出。官方焦距、主点和 root depth 是未标定假设，不属于已验证的实际相机几何。输出不包含产品 `pose3d.f32`。

12 项合成契约测试已通过，涵盖 RAW 数据空间分离、单目标数量、缺失模型、授权不符、采集时间不符、目标框缺失、依赖失败记录和输出保护：

```sh
.venv/bin/python -m unittest discover -s tests/pose3d -p test_mmpose_probe.py -v
```

这些测试不运行真实权重，不是 MMPose-L 精度或部署验收结论。真实 L/X 比较还同时改变模型规格与运行时，不是相同模型的导出等价性验证。
