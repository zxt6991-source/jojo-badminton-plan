#!/usr/bin/env python3
"""Evaluate a labeled manifest. Synthetic examples are never counted as validation."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import struct


def match_events(reference, predicted):
    """Order-preserving minimum-error alignment with a cost for misses/extra events.

    A 3-frame window defines a hit. Maximizing hits first prevents a distant event
    from stealing a valid match. Return (matched triples, missed count, extras).
    """
    truth, guesses = sorted(reference), sorted(predicted)
    memo = {}
    def solve(i, j):
        if i == len(truth) or j == len(guesses):
            return (0, 0, [])
        if (i, j) in memo:
            return memo[i, j]
        choices = [solve(i+1, j), solve(i, j+1)]
        error = abs(truth[i]-guesses[j])
        if error <= 3:
            n, cost, pairs = solve(i+1, j+1)
            choices.append((n+1, cost+error, [(truth[i], guesses[j], error)] + pairs))
        memo[i, j] = max(choices, key=lambda r: (r[0], -r[1]))
        return memo[i, j]
    count, _, pairs = solve(0, 0)
    return pairs, len(truth)-count, len(guesses)-count


def floats(path):
    raw = Path(path).read_bytes()
    if len(raw) % 4:
        raise ValueError('Binary file is not Float32 aligned')
    return struct.unpack('<' + 'f'*(len(raw)//4), raw)


def keypoint_availability(case, result, result_dir):
    analysis = result['analysis']
    ref = analysis.get('skeleton')
    expected = case['expected']
    if not expected.get('impactFrames'):
        return (0, 0)
    if not ref:
        # Failed inference still contributes unavailable expected-window frames.
        fps = (result.get('video') or {}).get('captureFps', case.get('captureFps', 30))
        half_window = round(.1 * fps)
        frames = {f for impact in expected['impactFrames'] for f in range(max(0, impact-half_window), impact+half_window+1)}
        return (0, len(frames))
    # Manifest may override gateway URL keys with local artifacts.
    timestamps = floats(result_dir / case.get('timestamps', ref['timestampsKey']))
    points = floats(result_dir / case.get('keypoints', ref['keypointsKey']))
    if len(points) != len(timestamps)*65*3:
        raise ValueError('Skeleton length mismatch')
    joints = [5, 6, 11, 12] + ([8, 10] if expected.get('handedness', 'right') == 'right' else [7, 9])
    windows = set()
    for impact in expected['impactFrames']:
        if not 0 <= impact < len(timestamps):
            raise ValueError('Reference impact frame outside skeleton')
        t = timestamps[impact]
        windows.update(i for i, timestamp in enumerate(timestamps) if abs(timestamp-t) <= .1)
    good = sum(all(points[(f*65+j)*3+2] >= .5 and all(math.isfinite(v) for v in points[(f*65+j)*3:(f*65+j)*3+3]) for j in joints) for f in windows)
    return good, len(windows)


def evaluate(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    cases = manifest.get('cases', [])
    if not cases:
        raise ValueError('Validation manifest must contain at least one labeled real case')
    if manifest.get('synthetic') or any(c.get('synthetic') for c in cases):
        raise ValueError('Synthetic fixtures cannot substantiate validation acceptance')
    rows, total_ref, hits, extras, good_frames, total_frames = [], 0, 0, 0, 0, 0
    views, correct_views, acceptable_tasks, qualified_runtimes = 0, 0, 0, []
    versions = set()
    for case in cases:
        result_path = (path.parent / case['result']).resolve()
        result = json.loads(result_path.read_text())
        analysis, expected = result['analysis'], case['expected']
        versions.add(analysis['pipelineVersion'])
        reference = expected.get('impactFrames', [])
        predicted = [s['impact']['frame'] for s in analysis.get('swings', [])]
        pairs, misses, extra = match_events(reference, predicted)
        total_ref += len(reference)
        hits += len(pairs)
        extras += extra
        available, count = keypoint_availability(case, result, result_path.parent)
        good_frames += available
        total_frames += count
        if 'view' in expected and expected.get('status', 'succeeded') == 'succeeded':
            views += 1
            correct_views += analysis.get('quality', {}).get('view') == expected['view']
        acceptable = analysis['status'] == expected.get('status', 'succeeded')
        if analysis['status'] == 'rejected_quality':
            # Rejecting a valid video is not success. Reason must be annotated.
            reasons = {i['code'] for i in analysis.get('quality', {}).get('issues', [])}
            acceptable = acceptable and bool(expected.get('rejectCode')) and expected['rejectCode'] in reasons
        acceptable_tasks += int(acceptable)
        runtime = analysis.get('stageTimingsMs', {}).get('total')
        video = result.get('video') or {}
        if 9 <= video.get('durationSec', 0) <= 11 and 119 <= video.get('captureFps', 0) <= 121 and runtime is not None:
            qualified_runtimes.append(runtime)
        rows.append({'id': case['id'], 'status': analysis['status'], 'impactMatches': pairs,
                     'missedImpacts': misses, 'extraImpacts': extra, 'taskAccepted': bool(acceptable),
                     'runtimeMs': runtime})
    impact_rate = hits/total_ref if total_ref else None
    availability = good_frames/total_frames if total_frames else None
    view_accuracy = correct_views/views if views else None
    task_rate = acceptable_tasks/len(cases)
    metrics = {'impactWithin3Frames': impact_rate, 'impactPrecisionWithin3Frames': hits/(hits+extras) if hits+extras else None,
               'keypointAvailability': availability, 'viewAccuracy': view_accuracy, 'taskSuccessRate': task_rate,
               'qualifiedRuntimeMaxMs': max(qualified_runtimes) if qualified_runtimes else None}
    checks = {'minimumRealCases': len(cases) >= 10, 'impact': None if impact_rate is None else impact_rate >= .9,
              'keypoints': None if availability is None else availability >= .9,
              'view': None if view_accuracy is None else view_accuracy >= .85,
              'tasks': task_rate >= .95,
              'runtime': None if not qualified_runtimes else max(qualified_runtimes) <= 120000}
    return {'pipelineVersions': sorted(versions), 'caseCount': len(cases), 'metrics': metrics, 'checks': checks,
            'acceptance': 'pass' if all(v is True for v in checks.values()) else ('incomplete' if any(v is None for v in checks.values()) else 'fail'),
            'limitations': ['Initial targets require representative real videos and specified hardware; no population-level accuracy claim.'],
            'cases': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = evaluate(args.manifest)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
