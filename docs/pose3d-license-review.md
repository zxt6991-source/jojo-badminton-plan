# P0-L：3D 发布许可审计

更新：2026-11-17。**当前发布结论：不放行 AI 3D 商业发布。** 以下是证据台账，不是对模型权重或衍生数据商用资格的自动判断；未知项保持未知。代码许可不能替代权重、训练数据与视频授权。

| 组件 | 已查证据 | 当前结论 / 剩余工作 |
| --- | --- | --- |
| rtmlib 代码 | 官方 LICENSE 为 Apache-2.0 | 代码许可已识别；发布时须锁定版本、保留要求的版权/许可/NOTICE，并审查依赖 |
| MMPose 代码 | 官方 LICENSE 为 Apache-2.0；已下载官方 v1.3.2 源码归档并记录 SHA-256 | 研究源码已固定；实际依赖组合与分发义务仍须核对 |
| MotionBERT 代码 | 官方 LICENSE 为 Apache-2.0 | 对照代码许可已识别；尚未选用 |
| RTMW3D ONNX 权重 | rtmlib 引用的 Soykaf HF 模型卡标 Apache-2.0；已取固定 revision，实际369330857字节与原站 SHA-256匹配，见模型实测记录 | **商业放行仍不确定**：发布者声明不替代上游权重/训练数据义务核查；转换等价性未验证，当前仅本机研究 |
| RTMW3D MMPose 权重 | 已从 OpenMMLab 官方入口下载 L checkpoint，230771611 字节，完整 SHA-256 见模型记录 | **不确定**：下载及完整性核对不等于已核实商用条款；当前仅本机研究 |
| MotionBERT 权重 | 未加载或核实 | **不确定**；不能继承代码 Apache 结论 |
| Human3.6M / H3WB | H3WB 官方称是 Human3.6M 的扩展；Human3.6M EULA 限定免费授权为学术用途，并要求未覆盖用途先取得许可 | **不确定**：所选模型实际训练清单、衍生标注与权重的适用义务须逐一确认；不能据 H3WB 代码 MIT 推导数据/权重可商用 |
| 解剖 GLB 与 Rig | 本仓库 `ATTRIBUTION.txt`、`source.json`、`MOTION-NOTICE.txt` 声明 CC BY-SA-4.0 及上游来源 | 保留已有署名与相同方式共享声明；App/Web 对模型和模型专用 Rig/蒙皮适配的分发方案尚未完成发布复核 |
| 验证视频 | 用户已明确授权现有两段视频用于本机内部验证，不上传；原片/代理hash绑定记录保存在私有data目录，正式manifest仍为空 | 仅该内部验证范围已获授权；未扩展到发布/训练集分发，未提供的保留期限与人工标注不伪造 |

代码许可来源：[rtmlib LICENSE](https://raw.githubusercontent.com/Tau-J/rtmlib/main/LICENSE)、[MMPose LICENSE](https://raw.githubusercontent.com/open-mmlab/mmpose/main/LICENSE)、[MotionBERT LICENSE](https://raw.githubusercontent.com/Walter0807/MotionBERT/main/LICENSE)。

模型/数据来源：[rtmlib Wholebody3d 配置](https://raw.githubusercontent.com/Tau-J/rtmlib/main/rtmlib/tools/solution/wholebody3d.py)、[MMPose RTMPose3D](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d)、[H3WB 官方仓库](https://github.com/wholebody3d/wholebody3d)、[Human3.6M EULA](https://vision.imar.ro/human3.6m/eula.php)。未对来源进行全文转载。

解剖材料的仓库证据为 `apps/web/public/models/anatomy/` 内的署名与许可文件；原始 GLB 的记录 hash 为 `d80bebc4045069b660ca8e7afe599d76d15ccc7d96410b3727c600ef01d24e54`，此次未更换模型。既有适配说明明确不代表测量三维重建或肌肉力学模拟。

发布前每项补齐：组件/权重精确版本、来源固定 commit/URL、实际 SHA-256、许可证快照位置、可商用/不可商用/不确定结论、适用用途、署名和分发义务、审核人和日期。涉及未授权用途时取得权利人书面结论。任何必需项为不可或不确定，对应组件不进入商业发布；当前只保留默认关闭的研究基础。VideoPose3D / GVHMR 未选用、未打包，本次没有重新核实其许可，不沿用设计稿数字作为已完成审计。
