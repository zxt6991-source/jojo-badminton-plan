"""Pure numerical stages. All derivatives use capture timestamps, never frame/fps."""
from __future__ import annotations
import math
import numpy as np

CWB_INDICES = list(range(23)) + list(range(91, 133))
SMOOTHING = {"minCutoff": 1.7, "beta": 2.0, "derivativeCutoff": 1.0, "maxGapSec": 0.05,
             "minConfidence": 0.5, "bidirectional": 1, "observationWeight": 0.5}


def capture_timestamps(playback, override_fps=None, declared_fps=None):
    pts = np.array(playback, dtype=np.float64, copy=True)
    if len(pts) < 2 or not np.all(np.isfinite(pts)) or np.any(np.diff(pts) <= 0):
        raise ValueError("invalid_timestamps")
    pts -= pts[0]
    # Explicit capture FPS only defines a uniform clock when the source is CFR.
    steps = np.diff(pts)
    fps = 1 / float(np.median(steps))
    # ffprobe prints microsecond precision; avoid treating 59.9988 as sub-60.
    if abs(fps - round(fps)) < .01:
        fps = float(round(fps))
    if override_fps is not None:
        if not 15 <= override_fps <= 480:
            raise ValueError("invalid_capture_fps")
        if np.max(np.abs(steps - np.median(steps))) > np.median(steps) * 0.15:
            raise ValueError("variable_remap_unsupported")
        pts = np.arange(len(pts), dtype=np.float64) / override_fps
        fps = override_fps
    return pts, fps


def to_cwb65(points, scores, width, height):
    if np.asarray(points).shape != (133, 2) or np.asarray(scores).shape != (133,):
        raise ValueError("pose_model_must_output_133_keypoints")
    out = np.zeros((65, 3), dtype=np.float32)
    out[:, :2] = np.asarray(points)[CWB_INDICES] / [width, height]
    out[:, 2] = np.clip(np.asarray(scores)[CWB_INDICES], 0, 1)
    invalid = (~np.isfinite(out).all(axis=1) | (out[:, 2] < .15)
               | (out[:, :2] < 0).any(axis=1) | (out[:, :2] > 1).any(axis=1))
    out[invalid, :2] = np.nan
    out[invalid, 2] = 0
    return out


def smooth_pose(pose, times, max_gap=.05):
    """Offline, confidence-gated One Euro smoothing without a causal phase delay.

    Average independent forward/backward filters, then retain half the measured
    position so fast swings are not flattened by the symmetric filter. Neither
    pass crosses a missing observation or a long timestamp gap. This does not
    repair model misdetections or estimate depth.
    """
    out = np.array(pose, dtype=np.float64, copy=True)
    times = np.asarray(times, dtype=np.float64)
    if (out.ndim != 3 or out.shape[2] != 3 or times.shape != (len(out),)
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('invalid_pose_timestamps')
    valid = (np.isfinite(out).all(axis=2) & (out[:, :, 2] >= SMOOTHING['minConfidence'])
             & (out[:, :, 2] <= 1))
    out[~valid, :2] = np.nan
    out[~valid, 2] = 0
    for joint in range(out.shape[1]):
        good = np.flatnonzero(valid[:, joint])
        for a, b in zip(good[:-1], good[1:]):
            if b > a + 1 and times[b] - times[a] <= max_gap:
                for f in range(a + 1, b):
                    ratio = (times[f] - times[a]) / (times[b] - times[a])
                    out[f, joint, :2] = (1 - ratio) * out[a, joint, :2] + ratio * out[b, joint, :2]
                    out[f, joint, 2] = min(out[a, joint, 2], out[b, joint, 2]) * .8
        observations = out[:, joint, :2].copy()

        def filter_direction(order):
            filtered = observations.copy()
            last, previous_raw, derivative, previous_time = None, None, np.zeros(2), None
            for frame in order:
                t, value = times[frame], observations[frame]
                if not np.isfinite(value).all():
                    last, previous_raw, previous_time = None, None, None
                    continue
                dt = abs(t - previous_time) if previous_time is not None else None
                if last is not None and dt <= max_gap:
                    alpha_d = 1 / (1 + 1 / (2 * math.pi * SMOOTHING['derivativeCutoff'] * dt))
                    derivative = alpha_d * ((value - previous_raw) / dt) + (1 - alpha_d) * derivative
                    cutoff = SMOOTHING['minCutoff'] + SMOOTHING['beta'] * np.linalg.norm(derivative)
                    alpha = 1 / (1 + 1 / (2 * math.pi * cutoff * dt))
                    filtered[frame] = alpha * value + (1 - alpha) * last
                else:
                    derivative = np.zeros(2)
                last, previous_raw, previous_time = filtered[frame].copy(), value, t
            return filtered

        symmetric = (filter_direction(range(len(times))) + filter_direction(range(len(times)-1, -1, -1))) / 2
        weight = SMOOTHING['observationWeight']
        out[:, joint, :2] = weight * observations + (1 - weight) * symmetric
    return out.astype('<f4')


def wrist_speed(pose, times, wrist, aspect=1.):
    xy = pose[:, wrist, :2] * [aspect, 1.]
    out = np.full(len(times), np.nan)
    valid = np.isfinite(xy).all(axis=1) & (pose[:, wrist, 2] >= .4)
    for i in range(1, len(times) - 1):
        if valid[i-1:i+2].all():
            out[i] = np.linalg.norm(xy[i+1] - xy[i-1]) / (times[i+1] - times[i-1])
    return out


def peak_frames(speed, times, separation=.35):
    finite = speed[np.isfinite(speed)]
    if len(finite) < 3:
        return []
    threshold = max(.35, float(np.median(finite) + 1.5 * np.median(np.abs(finite - np.median(finite)))))
    candidates = [i for i in range(1, len(speed)-1)
                  if np.isfinite(speed[i-1:i+2]).all() and speed[i] > threshold
                  and speed[i] > speed[i-1] and speed[i] >= speed[i+1]]
    selected = []
    for i in sorted(candidates, key=lambda k: speed[k], reverse=True):
        if all(abs(times[i]-times[j]) >= separation for j in selected):
            selected.append(i)
    return sorted(selected)


def audio_onsets(samples, sample_rate):
    samples = np.asarray(samples, dtype=float)
    hop = max(1, int(sample_rate * .005))
    n = len(samples) // hop
    if n < 3:
        return []
    rms = np.sqrt(np.mean(samples[:n*hop].reshape(n, hop) ** 2, axis=1))
    flux = np.maximum(0, np.diff(rms, prepend=rms[0]))
    threshold = max(.015, float(np.median(flux) + 6 * np.median(np.abs(flux - np.median(flux)))))
    candidates = [i for i in range(1, n-1) if flux[i] > threshold and flux[i] > flux[i-1] and flux[i] >= flux[i+1]]
    selected = []
    for i in sorted(candidates, key=lambda k: flux[k], reverse=True):
        if all(abs(i-j)*hop/sample_rate >= .2 for j in selected):
            selected.append(i)
    return sorted(i * hop / sample_rate for i in selected)


def detect_swings(pose, times, onsets, swing_type, aspect=1.):
    speeds = {hand: wrist_speed(pose, times, wrist, aspect) for hand, wrist in [('left', 9), ('right', 10)]}
    strength = {hand: float(np.nanpercentile(s, 95)) if np.isfinite(s).any() else 0 for hand, s in speeds.items()}
    strongest = max(strength, key=strength.get)
    other = 'left' if strongest == 'right' else 'right'
    handedness = strongest if strength[strongest] > 1.15 * strength[other] else 'unknown'
    speed = speeds[strongest]
    events = []
    for index in peak_frames(speed, times):
        # Audio can lag visual impact due to propagation. It gates confidence only;
        # it never moves the event onto the audio onset frame.
        supported = any(-.04 <= onset - times[index] <= .10 for onset in onsets)
        wrist = 9 if strongest == 'left' else 10
        window = np.abs(times - times[index]) <= .1
        confidence = float(np.mean(pose[window, wrist, 2])) * (.85 if supported else .62)
        if confidence < .3:
            continue
        threshold = speed[index] * .25
        start, end = index, index
        while start > 0 and times[index] - times[start-1] <= .8 and np.isfinite(speed[start-1]) and speed[start-1] > threshold:
            start -= 1
        while end < len(times)-1 and times[end+1]-times[index] <= .8 and np.isfinite(speed[end+1]) and speed[end+1] > threshold:
            end += 1
        phases = {"impactFrame": index}
        # Missing speed or clip boundaries do not manufacture stage boundaries.
        if start > 0 and np.isfinite(speed[start-1]) and speed[start-1] <= threshold:
            phases['backswingStartFrame'] = start
        if end < len(times)-1 and np.isfinite(speed[end+1]) and speed[end+1] <= threshold:
            phases['followThroughEndFrame'] = end
        events.append({"id": f"swing-{len(events)+1}", "type": swing_type,
                       "typeConfidence": 1., "handedness": handedness,
                       "impact": {"frame": index, "tSec": float(times[index]), "confidence": min(1., confidence),
                                  "evidence": ['wrist-peak', 'audio'] if supported else ['wrist-peak']},
                       "phases": phases})
    return events


def classify_view(pose, aspect):
    """Conservative geometric baseline; not a trained/certified view classifier."""
    good = (pose[:, [5, 6, 11, 12], 2] >= .5).all(axis=1)
    if good.sum() < 3:
        return 'unknown', 0.
    body = pose[good]
    shoulders = (body[:, 5, :2] + body[:, 6, :2]) / 2
    hips = (body[:, 11, :2] + body[:, 12, :2]) / 2
    torso = np.linalg.norm((shoulders-hips) * [aspect, 1], axis=1)
    ratio = np.median(np.abs(body[:, 5, 0] - body[:, 6, 0]) * aspect / np.maximum(torso, 1e-5))
    face = np.mean(body[:, :5, 2])
    if ratio < .28:
        return 'side', .65
    if ratio < .55:
        return 'oblique', .55
    if face > .7:
        return 'front', .6
    # Low face confidence alone cannot distinguish a back view from occlusion.
    return 'unknown', .3


def quality_report(pose, fps, luminance, multi_person, aspect, sample_indices=None):
    sampled = pose if sample_indices is None else pose[sample_indices]
    visible = float(np.mean(sampled[:, :17, 2] >= .5)) if len(sampled) else 0.
    view, view_confidence = classify_view(sampled, aspect)
    issues = [{"code": "view_classifier_uncalibrated", "severity": "warn"}]
    if visible < .5:
        issues.append({"code": "body_not_visible", "severity": "block"})
    lighting = 'dark' if luminance < 40 else 'ok'
    if lighting == 'dark':
        issues.append({"code": "dark_video", "severity": "warn"})
    if fps < 59:
        issues.append({"code": "low_fps", "severity": "warn"})
    if multi_person:
        issues.append({"code": "multiple_people", "severity": "info"})
    if view == 'unknown':
        issues.append({"code": "unknown_view", "severity": "warn"})
    return {"fpsTier": 'high' if fps >= 119 else ('mid' if fps >= 59 else 'low'),
            "view": view, "viewConfidence": view_confidence, "bodyVisibleRatio": visible,
            "lighting": lighting, "multiPerson": bool(multi_person), "issues": issues,
            "usable": visible >= .5}
