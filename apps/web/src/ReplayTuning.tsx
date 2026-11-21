import {useEffect, useRef, useState} from 'react';
import {isReplayProfile, type ReplayProfile} from '@badminton/schema';
import {api, json} from './api';
import Icon from './Icon';

const labels: Record<ReplayProfile, string> = {
  balanced: '均衡',
  responsive: '保留快速动作',
  steady: '优先稳定',
};

interface Props {
  analysisId: string;
  profile: ReplayProfile;
  onChange: (profile: ReplayProfile) => void;
}

export default function ReplayTuning({analysisId, profile, onChange}: Props) {
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [pending, setPending] = useState(false);
  const [suggestion, setSuggestion] = useState<ReplayProfile | null>(null);
  const [error, setError] = useState('');
  const request = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api<{configured: boolean}>('/api/replay-profile/status', {signal: controller.signal})
      .then(status => {if (!controller.signal.aborted) setConfigured(status.configured);})
      // If the status check fails, the user can still try the optimization request.
      .catch(() => {});
    return () => {controller.abort(); request.current?.abort();};
  }, [analysisId]);

  async function optimize() {
    if (request.current) return;
    if (suggestion) {onChange(suggestion); return;}
    const controller = new AbortController();
    request.current = controller;
    setPending(true);
    setError('');
    try {
      const result = await api<{profile?: unknown}>(
        `/api/analyses/${encodeURIComponent(analysisId)}/replay-profile`,
        {...json({}), signal: controller.signal},
      );
      if (controller.signal.aborted) return;
      if (!isReplayProfile(result?.profile)) throw new Error('模型返回了无法应用的回放档位，请重试。');
      setSuggestion(result.profile);
      onChange(result.profile);
    } catch (cause) {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'AI 选档失败，请重试。');
    } finally {
      if (request.current === controller) {
        request.current = null;
        if (!controller.signal.aborted) setPending(false);
      }
    }
  }

  function select(next: ReplayProfile) {
    request.current?.abort();
    request.current = null;
    setPending(false);
    setError('');
    onChange(next);
  }

  let message = 'AI 根据本片段的姿态质量摘要选择平滑强度。';
  if (configured === false) message = '启用 AI 选档需在网关 .env 中设置 DEEPSEEK_API_KEY 并重启。';
  if (pending) message = '正在分析姿态质量，当前档位可以继续播放。';
  if (suggestion) message = `AI 为本片段推荐「${labels[suggestion]}」。`;
  if (error) message = error;
  const applied = suggestion !== null && profile === suggestion;

  return <section className="replay-tuning" aria-label="回放优化">
    <label>回放平滑
      <select value={profile} onChange={e => {if (isReplayProfile(e.target.value)) select(e.target.value);}}>
        {Object.entries(labels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
      </select>
    </label>
    <button type="button" onClick={() => void optimize()} disabled={configured === false || pending || applied}>
      <Icon name={applied ? 'check' : 'bolt'} size={14}/>
      {pending ? 'AI 选档中…' : applied ? '已应用 AI 建议' : suggestion ? '应用 AI 建议' : 'AI 优化回放'}
    </button>
    <small role="status">{message}</small>
  </section>;
}
