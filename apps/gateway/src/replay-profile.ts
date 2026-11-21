import {CWB_65 as K, isReplayProfile, validateSkeletonData, type ReplayProfile, type VideoAsset} from '@badminton/schema';

export interface ReplayInput {
  video: VideoAsset;
  timestamps: Float32Array;
  keypoints: Float32Array;
}

export interface ReplayAdvisorOptions {
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  fetch?: typeof globalThis.fetch;
  timeoutMs?: number;
}

const fail = (status: number, errorCode: string, message: string) => Object.assign(new Error(message), {status, errorCode});
const rounded = (value: number) => Math.round(value * 1000) / 1000;

/** Only these clip-level statistics can leave the gateway. All distances are aspect-corrected. */
export function summarizeReplay({video, timestamps, keypoints}: ReplayInput) {
  try {
    validateSkeletonData(timestamps, keypoints, video.frameCount);
  } catch {
    throw fail(422, 'invalid_replay_data', '回放骨架数据不完整');
  }
  const count = Math.min(600, timestamps.length - 2);
  if (count < 1) throw fail(422, 'insufficient_replay_data', '连续姿态帧不足，无法分析回放');
  const aspect = video.width / video.height, stride = K.count * K.stride;
  const index = (frame: number, joint: number) => frame * stride + joint * K.stride;
  const visible = (i: number) => keypoints[i + 2] >= 0.5;
  const wristSpeeds: number[] = [];
  let missing = 0, gaps = 0, usableTriplets = 0, reversals = 0;
  const jointCount = K.rightAnkle - K.leftShoulder + 1;
  for (let sample = 0; sample < count; sample++) {
    const frame = count === 1 ? 1 : 1 + Math.round(sample * (timestamps.length - 3) / (count - 1));
    const before = timestamps[frame] - timestamps[frame - 1], after = timestamps[frame + 1] - timestamps[frame];
    const hasGap = Math.max(before, after) > Math.max(0.09, 2.5 / video.captureFps);
    if (hasGap) gaps++;
    const [ls, rs, lh, rh] = [K.leftShoulder, K.rightShoulder, K.leftHip, K.rightHip].map(j => index(frame, j));
    const torso = [ls, rs, lh, rh].every(visible) ? Math.hypot(
      (keypoints[ls] + keypoints[rs] - keypoints[lh] - keypoints[rh]) * aspect / 2,
      (keypoints[ls + 1] + keypoints[rs + 1] - keypoints[lh + 1] - keypoints[rh + 1]) / 2,
    ) : 0;
    for (let joint: number = K.leftShoulder; joint <= K.rightAnkle; joint++) {
      const i = index(frame, joint);
      if (!visible(i)) { missing++; continue; }
      if (hasGap || torso < 0.025 || !visible(i - stride) || !visible(i + stride)) continue;
      const dx1 = (keypoints[i] - keypoints[i - stride]) * aspect, dy1 = keypoints[i + 1] - keypoints[i - stride + 1];
      const dx2 = (keypoints[i + stride] - keypoints[i]) * aspect, dy2 = keypoints[i + stride + 1] - keypoints[i + 1];
      usableTriplets++;
      // Count a sharp direction reversal only when both displacements are appreciable.
      if (dx1 * dx2 + dy1 * dy2 < 0 && Math.min(Math.hypot(dx1, dy1), Math.hypot(dx2, dy2)) > torso * 0.04) reversals++;
      if (joint === K.leftWrist || joint === K.rightWrist) {
        wristSpeeds.push(Math.hypot(dx1 + dx2, dy1 + dy2) / ((before + after) * torso));
      }
    }
  }
  const observations = count * jointCount;
  if (!usableTriplets) throw fail(422, 'insufficient_replay_data', '连续可信的关节样本不足，无法分析回放');
  wristSpeeds.sort((a, b) => a - b);
  return {
    captureFps: rounded(video.captureFps),
    missingJointFraction: rounded(missing / observations),
    timestampGapFraction: rounded(gaps / count),
    usableTripletFraction: rounded(usableTriplets / observations),
    sharpReversalFraction: rounded(reversals / usableTriplets),
    wristSpeedP95TorsoLengthsPerSec: wristSpeeds.length ? rounded(wristSpeeds[Math.floor((wristSpeeds.length - 1) * 0.95)]) : null,
  };
}

const instructions = [
  'Choose a display smoothing profile for this badminton replay from the supplied aggregate tracking statistics.',
  'steady has the strongest temporal filter and can help when many well-observed joints reverse abruptly between adjacent samples.',
  'responsive has the lightest filter and best preserves rapid wrist motion when tracking is reliable.',
  'Choose balanced when evidence is mixed or sparse. Timestamp gaps and missing joints limit the observations.',
  'Return a JSON object with exactly one field, as in {"profile":"balanced"}. The other allowed values are "steady" and "responsive".',
].join(' ');

function parseReply(payload: {choices?: {finish_reason?: string; message?: {content?: unknown}}[]}): ReplayProfile {
  const choice = payload?.choices?.[0], content = choice?.message?.content;
  if (choice?.finish_reason === 'stop' && typeof content === 'string' && content.length <= 256) {
    try {
      const value: unknown = JSON.parse(content);
      if (value && typeof value === 'object' && Object.keys(value).length === 1 && 'profile' in value && isReplayProfile(value.profile)) return value.profile;
    } catch { /* Blank, prose and truncated completions all use the same fallback. */ }
  }
  throw fail(502, 'invalid_replay_profile', 'AI 返回的回放参数无效，请重试');
}

export function createReplayAdvisor(options: ReplayAdvisorOptions = {}) {
  const key = (options.apiKey ?? process.env.DEEPSEEK_API_KEY)?.trim();
  const model = (options.model ?? process.env.DEEPSEEK_MODEL)?.trim() || 'deepseek-flash';
  let endpoint: URL | undefined;
  try {
    const base = new URL(options.baseUrl ?? process.env.DEEPSEEK_BASE_URL ?? 'https://api.deepseek.com');
    const local = ['localhost', '127.0.0.1', '[::1]'].includes(base.hostname);
    if ((base.protocol === 'https:' || (local && base.protocol === 'http:')) && !base.username && !base.password && !base.search && !base.hash) {
      base.pathname = base.pathname.replace(/\/+$/, '') + '/chat/completions';
      endpoint = base;
    }
  } catch { /* A bad optional configuration must not interrupt normal replay. */ }

  async function request(summary: ReturnType<typeof summarizeReplay>): Promise<ReplayProfile> {
    const signal = AbortSignal.timeout(options.timeoutMs ?? 15_000);
    let response: Response;
    let payload;
    try {
      response = await (options.fetch ?? globalThis.fetch)(endpoint!, {
        method: 'POST', redirect: 'error', signal,
        headers: {'Content-Type': 'application/json', Authorization: `Bearer ${key}`},
        body: JSON.stringify({
          model, messages: [{role: 'system', content: instructions}, {role: 'user', content: JSON.stringify(summary)}],
          response_format: {type: 'json_object'}, max_tokens: 256, stream: false,
          ...(endpoint!.hostname === 'api.deepseek.com' && {thinking: {type: 'disabled'}}),
        }),
      });
      if (response.ok) payload = await response.json();
      else await response.body?.cancel();
    } catch {
      if (signal.aborted) throw fail(504, 'replay_ai_timeout', 'AI 调优请求超时，请重试');
      throw fail(502, 'replay_ai_unavailable', '无法读取 AI 服务响应，请检查服务地址或稍后重试');
    }
    if (!response.ok) {
      const message = response.status === 401 || response.status === 403 ? 'AI 服务拒绝认证，请检查网关密钥'
        : response.status === 429 ? 'AI 服务繁忙，请稍后重试' : 'AI 服务请求失败，请检查模型配置或稍后重试';
      throw fail(502, 'replay_ai_unavailable', message);
    }
    return parseReply(payload);
  }

  // A succeeded analysis has immutable source points. Manual labels do not affect this summary.
  const results = new Map<string, ReplayProfile>();
  const pending = new Map<string, Promise<ReplayProfile>>();
  return {
    configured: Boolean(key && endpoint),
    async advise(id: string, load: () => Promise<ReplayInput>) {
      if (!key || !endpoint) throw fail(503, 'replay_ai_unconfigured', '请在网关配置 AI 服务密钥和有效的服务地址');
      const cached = results.get(id);
      if (cached) return cached;
      const active = pending.get(id);
      if (active) return active;
      const job = load().then(summarizeReplay).then(request).then(profile => {
        if (results.size >= 128) results.delete(results.keys().next().value!);
        results.set(id, profile);
        return profile;
      }).finally(() => pending.delete(id));
      pending.set(id, job);
      return job;
    },
  };
}
