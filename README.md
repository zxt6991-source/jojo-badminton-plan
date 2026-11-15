# 羽毛球动作分析 · P0a

基于《羽毛球动作分析｜P0a 实施方案与核心数据 Schema》v0.2 的本地开发工程。包括 Web 上传与结果页、持久化任务网关、共享 Schema 与指标包、Python 姿态分析 Worker，以及评测脚本。

**当前交付是可运行的开发基线。** 内置演示为合成骨架，始终标识为演示。真实上传走 Python 推理，未配置模型时明确失败，绝不替换成演示结果。尚未拥有文档要求的 10–15 段授权、人工标注验证视频，因此不宣称达成准确率或 GPU 耗时验收目标。

默认结果页提供「身体协同」：在「3D 动作回放」中，象牙色骨骼与珊瑚色肌肉直接跟随姿态做出挥拍动作。可播放、慢放、逐帧，旋转视角看身体各部位的运动，并使用同一时间轴切回视频/二维骨架核对。骨骼和肌肉使用真实解剖网格，通过运行时骨架绑定驱动；动作来自当前片段的二维关键点。原有 5 项测量仍在「基础指标」。**尚未完成真实三维重建**，不估算受力、肌肉激活或发力效率。详见 [3D 查看器说明](docs/3d-viewer.md) 与 [身体协同说明](docs/biomechanics-notes.md)。

## 启动 Web 与 API

需要 Node.js 22+。

```bash
npm ci
npm run dev
```

打开 http://localhost:5174。网关监听 http://127.0.0.1:8787；开发服务器代理 API 和数据请求。默认只监听本机。

```bash
npm run build
npm start
```

构建后由网关同时提供网页，打开 http://127.0.0.1:8787。

## 启用真实视频分析

需要 Python 3.10+、ffmpeg/ffprobe、RTMLib 及其 ONNX 权重。完整说明见 [worker/README.md](worker/README.md)。

```bash
python3 -m venv .venv
.venv/bin/pip install -r worker/requirements.txt
cp .env.example .env
```

网关默认使用项目的 `.venv/bin/python`；未创建虚拟环境时才使用 `python3`。显式设置的 `PYTHON` 始终优先，如已有旧 `.env` 中写着 `PYTHON=python3`，请移除该行或指向已安装依赖的环境。在 `.env` 配置模型：

```dotenv
RTMLIB_DET_MODEL=/absolute/path/to/detection.onnx
RTMLIB_POSE_MODEL=/absolute/path/to/wholebody.onnx
POSE_DEVICE=cpu
```

或在开发环境显式设置 `ALLOW_MODEL_DOWNLOAD=1` 使用 RTMLib 默认模型。权重不随仓库分发；商业发布前仍须核对所选权重许可。重启 `npm run dev` 后上传 60 秒以内、500 MB 以内的原始视频。支持 MP4/MOV/M4V/WebM；浏览器无法预览的编码仍可能由 FFmpeg 摄取。多人画面可在预览中点选目标球员；建议拍摄全身、光线充足的侧面或斜侧视频。

慢动作视频可填写**已确认的真实采集帧率**。此输入仅用于均匀慢放；变速剪辑需提供原片。没有元数据的均匀慢放无法仅凭视频可靠识别，不能把文件播放帧率当作已验证的采集帧率。代理视频会归一化到采集时钟，骨架、指标和播放器共用该时间轴。

## 交付内容

| 目录 | 内容 |
| --- | --- |
| `apps/web` | 上传续传、动作选择、目标点选、任务状态、叠加骨架、逐帧/慢放、事件时间轴、指标与抑制原因、肘角曲线、人工标帧 |
| `apps/gateway` | 上传分片与完成校验、文件持久化、串行任务队列、任务幂等、重启恢复、显式重试、结果校验、TS 指标计算 |
| `packages/schema` | PDF 中的 v0.1 类型、运行时校验、cwb-65 索引、二进制校验 |
| `packages/metrics` | 5 个指标、采集时间计算、画幅比例修正、置信度与帧率/视角抑制 |
| `worker` | 摄取/转码、RTMLib 逐帧人体姿态、连续目标跟踪、平滑与短缺失插值、质量报告、音视频融合检测 |
| `fixtures` | 720 帧 / 120fps / 3 次挥拍的合成样例及生成器 |
| `scripts/evaluate.py` | 击球定位、误检漏检、关键点可用率、视角、耗时和成功率评测 |
| `scripts/regression.py` | 在标注集上重跑 Worker，并对比已有基线 |

所有指标为 2D 投影测量；不判断动作好坏，不输出教练话术。已提供解剖人体的 3D 动作回放；真实三维重建、肌肉负荷、App、账户、支付和分享尚未实现。解剖 GLB 随项目本地分发，来自 Fit Mit With / Z-Anatomy / BodyParts3D，署名与 CC BY-SA 许可证保存在 `apps/web/public/models/anatomy/`，构建时一并复制。

## 验证

```bash
npm run check
.venv/bin/python -m unittest discover -s tests/worker -v
```

自动化检查覆盖数据结构与二进制损坏、真实时间戳、指标几何与抑制、分片续传/任务幂等、Worker 数值算法、目标跟踪及事件评测。CI 对每次提交运行构建与这些测试。

真实视频回归需要准备标注清单，格式见 [worker/README.md](worker/README.md)：

```bash
.venv/bin/python scripts/regression.py validation/manifest.json --output-dir output/regression
.venv/bin/python scripts/regression.py validation/manifest.json --output-dir output/next --baseline output/regression/report.json
```

评测数据缺失或达不到目标时不会产生虚假的通过结论。更改根目录 `pipeline-version.json` 中的 `pipelineVersion` 后应运行全集回归；实际视频集及 GPU 环境尚未接入 CI。

## API

| 请求 | 用途 |
| --- | --- |
| `GET /api/health` | 网关健康状态 |
| `GET /api/demo` | 明确标识的演示结果 |
| `POST /api/uploads` | 以文件指纹创建或恢复上传会话 |
| `GET /api/uploads/:id` | 获取已上传分片 |
| `PUT /api/uploads/:id/chunks/:index` | 上传原始二进制分片 |
| `POST /api/uploads/:id/complete` | 校验所有分片并生成视频资产 |
| `POST /api/analyses` | 创建幂等任务，传入 videoId / swingType / targetPoint? / captureFps? |
| `GET /api/analyses` | 分析历史 |
| `GET /api/analyses/:id` | 状态和结果 |
| `POST /api/analyses/:id/retry` | 重试失败任务 |
| `POST /api/analyses/:id/annotations` | 保存人工击球帧，传入 `{ "frame": 120 }` |

`data/` 保存原视频、分片、JSON 状态、代理视频和骨架文件，重启保留。网关为单进程本地开发队列，不适合多个进程同时访问同一数据目录。上线时需替换为共享任务队列、数据库和对象存储，并接入访问控制和数据生命周期管理。

## 仍需真实数据验证

视角、持拍侧、朝向及击球检测目前是保守的基线算法，未经过教练标注校准；遮挡或不确定朝向会抑制指标。姿态识别覆盖每一帧，CPU 不承诺实时性能。文档中的 ≤3 帧 / ≥90%、视角 ≥85%、两分钟 GPU 耗时等目标，必须在授权验证集和目标硬件上实测。
