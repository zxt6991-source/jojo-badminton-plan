"""Selection contract tests; synthetic timing is not a real-video POC."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from pose3d_window_spec import (WINDOW_SPEC_SCHEMA, WINDOW_REPORT_SCHEMA, load_window_spec,
                                normalize_windows, validate_window_report_selection)
from pose3d_probe import ProbeError, MAX_PROBE_FRAMES


def selection(name='swing-1', start=.1, end=.3, anchor=2):
    return {'id': name, 'startTimeSec': start, 'endTimeSec': end, 'anchorFrame': anchor,
            'selectionReason': 'Explicit unverified research selection', 'humanVerified': False}


class WindowSpecTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'windows.json'
        self.times = np.array([0, .1, .2, .3, .7, 1.1, 1.5], dtype=np.float64)
        self.spec = {'schemaVersion': WINDOW_SPEC_SCHEMA, 'purpose': 'Research timing validation',
                     'windows': [selection()]}

    def load(self, times=None):
        self.path.write_text(json.dumps(self.spec))
        return load_window_spec(self.path, self.times if times is None else times)

    def assert_code(self, code, callback):
        with self.assertRaises(ProbeError) as error:
            callback()
        self.assertEqual(error.exception.code, code)

    def report(self):
        windows = self.load()
        for window in windows:
            window['frames'] = [{'frame': frame, 'captureTimeSec': float(self.times[frame]), 'status': 'succeeded'}
                                for frame in window['frameIndices']]
        return {'schemaVersion': WINDOW_REPORT_SCHEMA, 'purpose': self.spec['purpose'],
                'status': 'raw_evidence_collected', 'windows': windows}

    def test_inclusive_irregular_timestamps_keep_requested_seconds_and_anchor(self):
        windows = self.load()
        self.assertEqual(windows[0]['frameIndices'], [1, 2, 3])
        self.assertEqual((windows[0]['startFrame'], windows[0]['endFrame']), (1, 3))
        self.spec['windows'] = [selection(start=.15, end=.69, anchor=None)]
        window = self.load()[0]
        self.assertEqual(window['frameIndices'], [2, 3])
        self.assertEqual(window['startTimeSec'], .15)
        self.assertEqual(window['endTimeSec'], .69)
        self.assertIsNone(window['anchorFrame'])

    def test_exact_float32_boundaries_are_not_rounded_or_resampled(self):
        times = np.arange(448, dtype=np.float32) / 30
        self.spec['windows'] = [selection(f'w-{anchor}', float(times[anchor-30]), float(times[anchor+30]), anchor)
                                for anchor in (60, 169, 327)]
        windows = self.load(times)
        self.assertEqual([len(w['frameIndices']) for w in windows], [61, 61, 61])
        self.assertEqual(sum(len(w['frameIndices']) for w in windows), 183)
        self.assertEqual(MAX_PROBE_FRAMES, 10, 'existing sparse probe limit must remain unchanged')

    def test_zero_duration_exact_sample_is_a_single_frame_but_empty_interval_rejected(self):
        self.spec['windows'] = [selection(start=.2, end=.2, anchor=2)]
        self.assertEqual(self.load()[0]['frameIndices'], [2])
        self.spec['windows'] = [selection(start=.4, end=.6, anchor=None)]
        self.assert_code('empty_window', self.load)

    def test_timestamps_require_finite_strictly_increasing_numeric_vector(self):
        for times in ([], [0, 0, 1], [0, .2, .1], [0, np.nan, 1], [0, np.inf], [-.1, 0, 1],
                      [[0, 1]], [False, True], ['0', '1']):
            with self.subTest(times=times):
                self.assert_code('invalid_window_timestamps', lambda: self.load(times))

    def test_time_bounds_and_anchor_cannot_be_clamped_or_guessed(self):
        for change, code in [({'startTimeSec': -.1}, 'window_time_out_of_bounds'),
                              ({'endTimeSec': 2}, 'window_time_out_of_bounds'),
                              ({'startTimeSec': .4}, 'window_time_out_of_bounds'),
                              ({'endTimeSec': float('nan')}, 'window_time_out_of_bounds'),
                              ({'startTimeSec': False}, 'window_time_out_of_bounds'),
                              ({'anchorFrame': 0}, 'anchor_outside_window'),
                              ({'anchorFrame': 2.0}, 'anchor_outside_window'),
                              ({'anchorFrame': True}, 'anchor_outside_window')]:
            self.spec['windows'] = [{**selection(), **change}]
            self.assert_code(code, self.load)

    def test_order_overlap_and_duplicate_ids_rejected_without_merging(self):
        cases = [([selection(), selection('swing-2', .3, .7, 4)], 'overlapping_or_unsorted_windows'),
                 ([selection('late', .7, 1.1, 4), selection()], 'overlapping_or_unsorted_windows'),
                 ([selection(), selection('swing-1', .7, 1.1, 4)], 'duplicate_window_id')]
        for windows, code in cases:
            self.spec['windows'] = windows
            self.assert_code(code, self.load)

    def test_safe_ids_unverified_selection_and_purpose_are_required(self):
        for name in ('../escape', 'a/b', 'UPPER', '', 'a' * 49):
            self.spec['windows'] = [selection(name)]
            self.assert_code('invalid_window_id', self.load)
        for verified in (True, 0, None):
            self.spec['windows'] = [{**selection(), 'humanVerified': verified}]
            self.assert_code('unverified_research_selection_required', self.load)
        self.spec['windows'] = [selection()]
        self.spec['purpose'] = ' '
        self.assert_code('window_purpose_required', self.load)
        self.spec['purpose'] = 'Research'
        self.spec['windows'][0]['selectionReason'] = ''
        self.assert_code('unverified_research_selection_required', self.load)

    def test_each_window_duration_and_frame_count_caps_are_independent(self):
        self.spec['windows'] = [selection(start=0, end=3.1, anchor=None)]
        self.assert_code('window_duration_limit', lambda: self.load(np.arange(5, dtype=float)))
        times = np.arange(400, dtype=float) / 100
        self.spec['windows'] = [selection(start=0, end=.90, anchor=45)]
        self.assert_code('window_frame_limit', lambda: self.load(times))
        self.spec['windows'][0]['endTimeSec'] = .89
        self.assertEqual(len(self.load(times)[0]['frameIndices']), 90)

    def test_total_frame_budget_and_window_count_caps(self):
        times = np.arange(500, dtype=float) / 100
        self.spec['windows'] = [selection(f'w{i}', i, i+.89, None) for i in range(3)]
        self.assert_code('total_window_frame_limit', lambda: self.load(times))
        self.spec['windows'] = [selection(f'w{i}', i, i+.79, None) for i in range(3)]
        self.assertEqual(sum(len(w['frameIndices']) for w in self.load(times)), 240)
        for windows in ([], [selection() for _ in range(4)]):
            self.spec['windows'] = windows
            self.assert_code('invalid_window_count', self.load)

    def test_spec_file_schema_and_required_fields(self):
        self.spec['schemaVersion'] = 'wrong'
        self.assert_code('unsupported_window_spec', self.load)
        self.spec['schemaVersion'] = WINDOW_SPEC_SCHEMA
        del self.spec['windows'][0]['anchorFrame']
        self.assert_code('window_selection_fields_required', self.load)
        self.assert_code('local_window_spec_required', lambda: load_window_spec('https://example.com/spec.json', self.times))

    def test_complete_success_and_explicit_failed_frames_preserve_same_selection(self):
        report = self.report()
        expected = self.load()
        self.assertEqual(validate_window_report_selection(report, self.times), expected)
        report['windows'][0]['frames'][1].update(status='failed', errorCode='inference_failed')
        report['status'] = 'raw_evidence_collected_with_failures'
        self.assertEqual(validate_window_report_selection(report, self.times), expected)
        for frame in report['windows'][0]['frames']:
            frame.update(status='failed', errorCode='frame_not_processed')
        report['status'] = 'failed'
        self.assertEqual(validate_window_report_selection(report, self.times), expected)

    def test_missing_duplicate_out_of_order_and_wrong_time_frames_never_pass(self):
        original = self.report()
        for mutate, code in [
            (lambda r: r['windows'][0]['frames'].pop(), 'incomplete_window_frame_coverage'),
            (lambda r: r['windows'][0]['frames'][1].update(frame=1), 'window_frame_order_mismatch'),
            (lambda r: r['windows'][0]['frames'].reverse(), 'window_frame_order_mismatch'),
            (lambda r: r['windows'][0]['frames'][0].update(captureTimeSec=.11), 'window_frame_time_mismatch'),
            (lambda r: r['windows'][0].update(startFrame=0), 'window_frame_range_mismatch'),
            (lambda r: r['windows'][0].update(frameIndices=[1, 3]), 'window_frame_indices_mismatch'),
        ]:
            report = copy.deepcopy(original)
            mutate(report)
            self.assert_code(code, lambda: validate_window_report_selection(report, self.times))

    def test_frame_error_and_terminal_status_contract(self):
        original = self.report()
        for mutate, code in [
            (lambda r: r.update(status='running'), 'incomplete_window_report'),
            (lambda r: r.update(status='raw_evidence_collected_with_failures'), 'window_report_status_mismatch'),
            (lambda r: r['windows'][0]['frames'][0].update(status='failed'), 'failed_frame_error_required'),
            (lambda r: r['windows'][0]['frames'][0].update(errorCode=None), 'successful_frame_has_error'),
            (lambda r: r['windows'][0]['frames'][0].update(status='skipped'), 'invalid_window_frame_status'),
            (lambda r: r['windows'][0]['frames'][0].update(status='failed', errorCode='blocked'), 'window_report_status_mismatch'),
        ]:
            report = copy.deepcopy(original)
            mutate(report)
            self.assert_code(code, lambda: validate_window_report_selection(report, self.times))


if __name__ == '__main__':
    unittest.main()
