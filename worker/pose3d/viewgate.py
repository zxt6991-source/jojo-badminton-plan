"""Uncalibrated initial view policy, explicitly versioned in config.json."""
import math


def classify_gate(view, confidence, policy):
    threshold = policy['minConfidence']
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('invalid_view_policy')
    view = 'rear' if view == 'back' else view
    if view not in ('side', 'oblique', 'front', 'rear', 'unknown'):
        view = 'unknown'
    if (not isinstance(confidence, (int, float)) or not math.isfinite(confidence)
            or not threshold <= confidence <= 1 or view == 'unknown'):
        return 'unknown', 'blocked'
    return view, 'pass' if view in ('side', 'oblique') else 'limited'


def gate_windows(windows, view, confidence, policy):
    view_class, gate = classify_gate(view, confidence, policy)
    return [{**window, 'viewClass': view_class, 'viewGate': gate,
             'usableFrameRatio': 0.0} for window in windows]
