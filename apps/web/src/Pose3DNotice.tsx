import {useEffect, useState} from 'react';
import type {Pose3DResult} from '@badminton/schema';
import {api} from './api';

const reasons: Record<string, string> = {
  model_unavailable: '未配置可用的 3D 模型。',
  no_swing_windows: '未识别到击球片段，可先手动标帧。',
  view_blocked: '当前视角无法可靠估计 3D。',
  cancelled: '3D 处理已取消。',
  timeout: '3D 处理超时。',
  interrupted: '3D 处理被中断。',
};

/** Status only: AI rendering remains gated until the P0 model/user study passes. */
export default function Pose3DNotice({analysisId, initial}: {analysisId: string; initial?: Pose3DResult}) {
  const [result, setResult] = useState(initial);
  useEffect(() => {
    setResult(initial);
    if (!initial || !['queued', 'running'].includes(initial.status)) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const response = await api<{pose3d: Pose3DResult | null}>(`/api/analyses/${analysisId}/pose3d`);
        if (disposed) return;
        setResult(response.pose3d ?? undefined);
        if (response.pose3d && ['queued', 'running'].includes(response.pose3d.status)) timer = setTimeout(poll, 1200);
      } catch {if (!disposed) timer = setTimeout(poll, 3000);}
    };
    void poll();
    return () => {disposed = true; clearTimeout(timer);};
  }, [analysisId, initial]);
  if (!result) return null;
  const message = result.status === 'queued' ? 'AI 3D 正在排队。' : result.status === 'running' ? 'AI 3D 正在处理。'
    : result.status === 'succeeded' ? '3D 研究结果已生成，AI 姿态回放尚未开放。'
    : reasons[result.errorCode ?? ''] ?? (result.status === 'disabled' ? 'AI 3D 尚未启用。' : '当前片段的 3D 估计不可用。');
  return <p className="body3d-note" role="status">{message}视频、二维指标与示意回放仍可使用。{result.quality.warnings.includes('view_limited') ? ' 当前机位下 3D 深度可靠性较低。' : ''}</p>;
}
