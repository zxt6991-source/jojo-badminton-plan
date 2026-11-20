# 本机 3D 研究输入准备

`scripts/pose3d_prepare.py` 从已成功的 2D 分析准备原始模型探针输入。它不运行模型、不上传视频、不生成人工标注，也不改变产品视角门控。

```bash
python3 scripts/pose3d_prepare.py \
  --data-dir data \
  --authorization-evidence data/pose3d-research/authorization.json \
  --output data/pose3d-research/inputs-draft-new.json

.venv/bin/python -m unittest discover -s tests/pose3d -v
```

必须显式提供非空本地 JSON 授权记录。Markdown 证据也可用，但须包含恰好一个 `json` 代码块，内容采用相同结构。脚本不代写授权；授权记录须由用户的实际指示生成，不能由脚本示例推断同意。

每次运行须使用新的输出文件名；已有文件、符号链接和并发创建的同名输出都会被拒绝，失败报告也不会覆盖原始证据。

授权结构为 `schemaVersion: pose3d-local-authorization-0.1.0`、`scope: local-internal-3d-validation`、`externalUploadAllowed: false`、非空 `userStatement`、有效日期 `authorizedAt`，以及 `items`。若显式提供 `retentionUntil`，已经过期的授权在读取视频前拒绝；未提供或为 `null` 时不编造保留期限。每项含 `videoId`、绝对 `originalPath` / `proxyPath` 和各自 `originalSha256` / `proxySha256`。准备时严格核对路径和实际内容哈希；修改视频或改用另一代理后须重新核对授权范围。未提供的保留期限、用途扩展或人工标签不会自动补全。

脚本选择每段原视频的最新成功 2D 分析。旧成功记录列为 `older_successful_analysis`；失败或质量拒绝记录列为 `analysis_not_succeeded`。最新产物损坏时报告错误，不静默换用旧结果。原片、代理、时间戳、2D骨架均记录实际 SHA-256。路径必须属于对应任务/视频目录，拒绝符号链接和穿越；输出不能覆盖来源目录。

代理经 `ffprobe` 逐帧核对解码数量、宽高和既有时间戳，误差上限为 0.0001 秒；不会根据容器 FPS 重建时间轴。这只证明代理帧与分析时间轴一致，不证明原片未经剪辑或实际拍摄帧率已独立核实。骨架检查 cwb-65 的 `frameCount × 65 × 3` 小端浮点布局、置信度、归一化坐标和缺失编码。候选击球帧必须与既有时间戳一致。

输出是 `pose3d-research-inputs-0.1.0` 的 **draft**，不是正式评估 manifest：

- `inputs[].modelInputPath` 指向与 2D 同时钟的代理，原片与代理路径分别保留。
- `inputs[].frames[]` 包含每个采集帧的 `tSec`、图像归一化/像素 XYXY bbox、可见 body17 点数。bbox 由已有 2D 点包围框得到，阈值 0.4、至少 5 点、不加 padding；不足时明确 `usable:false` 和空 bbox，不无限沿用上一帧。
- `inputs[].candidateSwingFrames[]` 是既有 2D 事件候选及同帧 bbox，`humanVerified:false`。这些事件和 bbox 均不等于人工确认的击球或身份真值。
- `predictedView` 保留原 2D 预测及置信度，明确来源为启发式。`currentViewGate` 按当前版本化配置记录门控结果；即使允许本地探针，`userDisplayAllowed` 始终为 `false`。
- 不填写人工 `viewClass`、左右交换、序数深度、保留期限或发布通过结论。`gateDecision:not_assessed`、`releaseReady:false`、`inferenceExecuted:false` 固定表示本工具的边界。

CLI 有至少一个可用输入时退出 `0`；全部不可用或授权无效时退出 `2`。前者不表示 Gate A/B 通过。测试里的代理解码器和视频字节为合成契约夹具，不纳入模型表现。
