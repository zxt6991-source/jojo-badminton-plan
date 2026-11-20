"""Bounded contiguous research selections; never human annotations or POC gates.

load_window_spec(path, timestamps) -> normalized list of windows.
validate_window_report_selection(report, timestamps) -> the same normalized list.
Both raise pose3d_probe.ProbeError with a machine-readable code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import numpy as np

from pose3d_probe import ProbeError, require, MAX_CLIP_FRAMES

WINDOW_SPEC_SCHEMA = 'pose3d-research-windows-0.1.0'
WINDOW_REPORT_SCHEMA = 'pose3d-window-probe-0.1.0'
MAX_WINDOWS = 3
MAX_WINDOW_FRAMES = 90
MAX_WINDOW_SECONDS = 3.0
MAX_TOTAL_FRAMES = 240
WINDOW_ID_PATTERN = r'[a-z0-9][a-z0-9_-]{0,47}'
SELECTION_FIELDS = ('id', 'startTimeSec', 'endTimeSec', 'anchorFrame', 'selectionReason', 'humanVerified')


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_timestamps(timestamps):
    try:
        values = np.asarray(timestamps)
        require(values.ndim == 1 and 1 <= len(values) <= MAX_CLIP_FRAMES
                and values.dtype.kind in 'if', 'invalid_window_timestamps')
        values = values.astype(np.float64)
        require(np.isfinite(values).all() and values[0] >= 0
                and np.all(np.diff(values) > 0), 'invalid_window_timestamps')
    except (ValueError, TypeError) as error:
        if isinstance(error, ProbeError):
            raise
        raise ProbeError('invalid_window_timestamps', str(error)) from error
    return values


def normalize_windows(windows, timestamps):
    """Inclusive time bounds; no margins, resampling, sorting or implicit repair."""
    times = validate_timestamps(timestamps)
    require(isinstance(windows, list) and 1 <= len(windows) <= MAX_WINDOWS, 'invalid_window_count')
    normalized, ids, previous_time, previous_frame, count = [], set(), None, -1, 0
    for item in windows:
        require(isinstance(item, dict) and all(key in item for key in SELECTION_FIELDS), 'window_selection_fields_required')
        name = item['id']
        require(isinstance(name, str) and re.fullmatch(WINDOW_ID_PATTERN, name) is not None, 'invalid_window_id')
        require(name not in ids, 'duplicate_window_id')
        ids.add(name)
        require(nonempty(item['selectionReason']) and item['humanVerified'] is False, 'unverified_research_selection_required')
        start, end = item['startTimeSec'], item['endTimeSec']
        require(finite_number(start) and finite_number(end) and times[0] <= start <= end <= times[-1], 'window_time_out_of_bounds')
        require(end - start <= MAX_WINDOW_SECONDS, 'window_duration_limit')
        require(previous_time is None or start > previous_time, 'overlapping_or_unsorted_windows')
        start_frame = int(np.searchsorted(times, start, side='left'))
        end_frame = int(np.searchsorted(times, end, side='right')) - 1
        require(start_frame <= end_frame, 'empty_window')
        require(start_frame > previous_frame, 'overlapping_or_unsorted_windows')
        frame_count = end_frame - start_frame + 1
        require(frame_count <= MAX_WINDOW_FRAMES, 'window_frame_limit')
        count += frame_count
        require(count <= MAX_TOTAL_FRAMES, 'total_window_frame_limit')
        anchor = item['anchorFrame']
        require(anchor is None or (type(anchor) is int and start_frame <= anchor <= end_frame), 'anchor_outside_window')
        normalized.append({**{key: item[key] for key in SELECTION_FIELDS},
                           'startFrame': start_frame, 'endFrame': end_frame,
                           'frameIndices': list(range(start_frame, end_frame + 1))})
        previous_time, previous_frame = end, end_frame
    return normalized


def load_window_spec(path, timestamps):
    require(isinstance(path, (str, Path)) and '://' not in str(path), 'local_window_spec_required')
    path = Path(path).expanduser().resolve()
    require(path.is_file() and 0 < path.stat().st_size <= 1_000_000, 'window_spec_unavailable')
    try:
        spec = json.loads(path.read_text())
    except (ValueError, UnicodeError) as error:
        raise ProbeError('invalid_window_spec_json', str(error)) from error
    require(isinstance(spec, dict) and spec.get('schemaVersion') == WINDOW_SPEC_SCHEMA, 'unsupported_window_spec')
    require(nonempty(spec.get('purpose')), 'window_purpose_required')
    return normalize_windows(spec.get('windows'), timestamps)


def validate_window_report_selection(report, timestamps):
    """Verify every selected frame is represented once, including failed frames.

    Artifact hashes and authorization remain the runner/evaluator's responsibility.
    A valid selection report does not imply successful inference or accuracy.
    """
    require(isinstance(report, dict) and report.get('schemaVersion') == WINDOW_REPORT_SCHEMA, 'unsupported_window_report')
    require(nonempty(report.get('purpose')), 'window_purpose_required')
    require(report.get('status') in ('raw_evidence_collected', 'raw_evidence_collected_with_failures', 'failed'),
            'incomplete_window_report')
    times = validate_timestamps(timestamps)
    normalized = normalize_windows(report.get('windows'), times)
    successes, failures = 0, 0
    for item, expected in zip(report['windows'], normalized):
        require(type(item.get('startFrame')) is int and type(item.get('endFrame')) is int
                and item['startFrame'] == expected['startFrame'] and item['endFrame'] == expected['endFrame'],
                'window_frame_range_mismatch')
        indices = item.get('frameIndices')
        require(isinstance(indices, list) and all(type(i) is int for i in indices)
                and indices == expected['frameIndices'], 'window_frame_indices_mismatch')
        frames = item.get('frames')
        require(isinstance(frames, list) and len(frames) == len(indices), 'incomplete_window_frame_coverage')
        for value, index in zip(frames, indices):
            require(isinstance(value, dict) and type(value.get('frame')) is int and value['frame'] == index,
                    'window_frame_order_mismatch')
            require(finite_number(value.get('captureTimeSec')) and value['captureTimeSec'] == float(times[index]),
                    'window_frame_time_mismatch')
            require(value.get('status') in ('succeeded', 'failed'), 'invalid_window_frame_status')
            if value['status'] == 'succeeded':
                require('errorCode' not in value, 'successful_frame_has_error')
                successes += 1
            else:
                require(nonempty(value.get('errorCode')), 'failed_frame_error_required')
                failures += 1
    status = report['status']
    require(status != 'raw_evidence_collected' or failures == 0, 'window_report_status_mismatch')
    require(status != 'raw_evidence_collected_with_failures' or (successes > 0 and failures > 0),
            'window_report_status_mismatch')
    return normalized
