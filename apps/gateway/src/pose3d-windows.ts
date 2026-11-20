import type {Analysis, Pose3DScope, Pose3DWindow, ViewClass} from '@badminton/schema';

export interface WindowPolicy {
  preSec: number | null;
  postSec: number | null;
  minViewConfidence: number;
  maxFullDurationSec: number;
  maxFrames: number;
  maxFullFrames: number;
  maxWindows: number;
}
export const poseError = (errorCode: string, message: string, status = 400) =>
  Object.assign(new Error(message), {errorCode, status});

/** Uses acquisition timestamps, never an assumed FPS. Thresholds are research policy. */
export function selectPose3DWindows(analysis: Analysis, timestamps: Float32Array,
  scope: Pose3DScope, requested: unknown, policy: WindowPolicy): Pose3DWindow[] {
  const count = analysis.skeleton!.frameCount;
  if (count > policy.maxFrames) throw poseError('scope_limit_exceeded', '3D 帧数超出上限', 422);
  if (timestamps.length !== count || !timestamps.every((t, i) => Number.isFinite(t) && t >= 0 && (!i || t > timestamps[i - 1])))
    throw poseError('invalid_timestamps', '采集时间轴损坏', 422);
  let ranges: {startFrame: number; endFrame: number}[];
  if (scope === 'manual') {
    if (!Array.isArray(requested) || !requested.length || requested.length > policy.maxWindows)
      throw poseError('invalid_windows', `手动处理需要 1–${policy.maxWindows} 个帧窗口`);
    ranges = requested.map(w => {
      if (!w || !Number.isSafeInteger(w.startFrame) || !Number.isSafeInteger(w.endFrame) ||
        w.startFrame < 0 || w.endFrame < w.startFrame || w.endFrame >= count)
        throw poseError('invalid_windows', '窗口帧范围无效或越界');
      return {startFrame: w.startFrame, endFrame: w.endFrame};
    });
  } else if (scope === 'full') {
    const deltas = Array.from(timestamps.slice(1), (time, i) => time - timestamps[i]).sort((a, b) => a - b);
    const middle = Math.floor(deltas.length / 2);
    const tailDuration = !deltas.length ? 0 : deltas.length % 2 ? deltas[middle] : (deltas[middle - 1] + deltas[middle]) / 2;
    if (count > policy.maxFullFrames || timestamps[count - 1] + tailDuration > policy.maxFullDurationSec)
      throw poseError('scope_limit_exceeded', '整段 3D 超过时长或帧数上限', 422);
    ranges = [{startFrame: 0, endFrame: count - 1}];
  } else {
    if (!analysis.swings.length) throw poseError('no_swing_windows', '未识别到击球片段，请先手动标帧', 422);
    if (policy.preSec === null || policy.postSec === null)
      throw poseError('window_policy_unverified', '挥拍窗口时长尚未完成 POC 配置', 503);
    ranges = analysis.swings.map(s => {
      const time = timestamps[s.impact.frame];
      const startTime = time - policy.preSec!, endTime = time + policy.postSec!;
      let startFrame = 0, endFrame = count - 1;
      while (startFrame < count - 1 && timestamps[startFrame] < startTime) startFrame++;
      while (endFrame > 0 && timestamps[endFrame] > endTime) endFrame--;
      return {startFrame, endFrame};
    });
  }
  ranges.sort((a, b) => a.startFrame - b.startFrame || a.endFrame - b.endFrame);
  const merged: typeof ranges = [];
  for (const range of ranges) {
    const previous = merged.at(-1);
    if (previous && range.startFrame <= previous.endFrame + 1) previous.endFrame = Math.max(previous.endFrame, range.endFrame);
    else merged.push({...range});
  }
  if (merged.reduce((sum, w) => sum + w.endFrame - w.startFrame + 1, 0) > policy.maxFrames)
    throw poseError('scope_limit_exceeded', '3D 窗口超过帧数上限', 422);
  if (merged.length > policy.maxWindows) throw poseError('scope_limit_exceeded', '3D 窗口数量超出上限', 422);
  const view = analysis.quality?.view;
  const viewClass: ViewClass = !view || (analysis.quality?.viewConfidence ?? 0) < policy.minViewConfidence ? 'unknown' : view === 'back' ? 'rear' : view;
  const viewGate = viewClass === 'unknown' ? 'blocked' : ['side', 'oblique'].includes(viewClass) ? 'pass' : 'limited';
  return merged.map(w => ({...w, origin: scope === 'manual' ? 'manual-annotation' : scope === 'full' ? 'full-clip' : 'auto-swing-event', viewClass, viewGate, usableFrameRatio: 0}));
}
