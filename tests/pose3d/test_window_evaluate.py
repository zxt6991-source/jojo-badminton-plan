"""Synthetic window evidence and metric checks; never model accuracy evidence."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_runtime_probe_compare as fixtures
import pose3d_window_evaluate as evaluation
from pose3d_camera_research import CONFIG_SCHEMA, recover_camera, pose_metrics
from pose3d_probe import ProbeError, digest, validate_proxy_clock
from pose3d_window_spec import load_window_spec


class WindowEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RuntimeComparisonTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root.resolve()
        self.a, self.b = self.fixture.a, self.fixture.b
        self.paths = [self.fixture.a_path, self.fixture.b_path]
        self.times = np.cumsum(np.resize([.03, .04, .05], 240)).astype('<f4')
        self.times -= self.times[0]
        self.times.tofile(self.fixture.fixture.times)
        np.full((240, 65, 3), [.01, .01, .9], dtype='<f4').tofile(self.fixture.points)
        self.clock = {**self.fixture.clock, 'frameCount': len(self.times)}
        for runtime, report in zip(('rtmlib-onnx', 'mmpose'), (self.a, self.b)):
            report.update(schemaVersion='pose3d-window-probe-0.1.0', runtime=runtime,
                          purpose='synthetic contiguous research windows', elapsedSec=2.,
                          modelLoadSec=.2, decodeSec=.1)
            report['input'].update(frameCount=len(self.times),
                timestampsSha256=digest(self.fixture.fixture.times),
                keypoints2dSha256=digest(self.fixture.points), proxyClockValidation=self.clock)
        self.camera = {'id': 'explicit', 'source': 'assumed-research',
            'evidence': 'Synthetic test camera, no calibration claim',
            'imageWidth': 100, 'imageHeight': 100, 'fx': 150., 'fy': 160.,
            'cx': 48., 'cy': 52., 'rootDepth': 5.}
        self.config_path, self.spec_path = self.root / 'camera.json', self.root / 'windows.json'
        self.config_path.write_text(json.dumps({'schemaVersion': CONFIG_SCHEMA, 'scenarios': [self.camera]}))
        self.build_windows([(0, 4), (8, 12), (20, 24)])

    def build_windows(self, bounds):
        spec = {'schemaVersion': 'pose3d-research-windows-0.1.0', 'purpose': 'Synthetic research only',
                'windows': [{'id': f'window-{i}', 'startTimeSec': float(self.times[start]),
                             'endTimeSec': float(self.times[end]), 'anchorFrame': start,
                             'selectionReason': 'Synthetic selection without human annotation', 'humanVerified': False}
                            for i, (start, end) in enumerate(bounds)]}
        self.spec_path.write_text(json.dumps(spec))
        windows = load_window_spec(self.spec_path, self.times)
        self.arrays = {'a': {}, 'b': {}}
        for label, report, path in zip(('a', 'b'), (self.a, self.b), self.paths):
            report.update(windows=[], windowSpecPath=str(self.spec_path), windowSpecSha256=digest(self.spec_path))
            template = report['frames'][0]
            for window in windows:
                (path.parent / window['id']).mkdir(exist_ok=True)
                result = {**copy.deepcopy(window), 'frames': []}
                for index in window['frameIndices']:
                    arrays = copy.deepcopy(self.fixture.fixture.arrays if label == 'a' else self.fixture.b_arrays)
                    # Both decoders encode the same image/depth evidence. B's
                    # unrelated default-camera output must never affect metrics.
                    q = self.fixture.fixture.arrays['keypoints_simcc'].copy()
                    q[0, 9, 2] = 140 + index % 12
                    arrays['keypoints_simcc'] = q
                    decoded = q.copy()
                    decoded[..., 2] = (q[..., 2] / (192 if label == 'a' else 144) - 1) * evaluation.OFFICIAL_Z_RANGE
                    arrays['keypoints' if label == 'a' else 'decoded_keypoints'] = decoded
                    xy = np.ones((1, 133, 2), dtype=np.float32) * 50
                    xy[0, [5, 7, 9, 6, 8, 10, 11, 12]] = [
                        [30, 20], [35, 40], [40 + index % 7, 32], [65, 20], [60, 40], [70, 38], [40, 65], [60, 65]]
                    arrays.update(keypoints_2d=xy, frame_index=np.array(index, dtype=np.int64),
                                  timestamp_sec=np.array(float(self.times[index])))
                    suffix = '.raw.npz' if label == 'a' else '.mmpose-raw.npz'
                    item = {**copy.deepcopy(template), 'frame': index, 'captureTimeSec': float(self.times[index]),
                            'rawPath': f'frame-{index:06d}{suffix}', 'status': 'succeeded',
                            'inferenceAttempted': True, 'inferenceSec': .25}
                    result['frames'].append(item)
                    self.arrays[label][index] = arrays
                    self.write_raw(label, result, item)
                report['windows'].append(result)

    def write_raw(self, label, window, item):
        path = self.paths[0 if label == 'a' else 1].parent / window['id'] / item['rawPath']
        np.savez_compressed(path, **self.arrays[label][item['frame']])
        item['rawSha256'] = digest(path)

    def mark_failed(self, label, frame, attempted=True):
        report = self.a if label == 'a' else self.b
        item = next(item for window in report['windows'] for item in window['frames'] if item['frame'] == frame)
        item.update(status='failed', errorCode='model_call_failed' if attempted else 'frame_not_processed',
                    rawPath=None, rawSha256=None, outputShapes=None, inferenceAttempted=attempted,
                    inferenceSec=.5 if attempted else None)
        if not attempted:
            item['bboxXYXY'] = None
        report['status'] = ('raw_evidence_collected_with_failures' if any(
            item['status'] == 'succeeded' for window in report['windows'] for item in window['frames']) else 'failed')

    def write_reports(self):
        for path, report in zip(self.paths, (self.a, self.b)):
            path.write_text(json.dumps(report))

    def evaluate(self, validator=None):
        self.write_reports()
        return evaluation.evaluate_windows(*self.paths, self.config_path, 'explicit',
                                           clock_validator=validator or (lambda *_: self.clock))

    def rejects(self, code, validator=None):
        with self.assertRaises(ProbeError) as caught:
            self.evaluate(validator)
        self.assertEqual(caught.exception.code, code)

    def test_183_frames_keep_failed_and_unattempted_denominators(self):
        self.build_windows([(0, 60), (80, 140), (160, 220)])
        self.mark_failed('a', 30)
        self.mark_failed('a', 90, attempted=False)
        result = self.evaluate()
        self.assertEqual(result['summary']['requestedFramesPerRun'], 183)
        a = result['summary']['runs']['a']
        self.assertEqual((a['successfulFrames'], a['failedFrames'], a['inferenceAttemptedFrames']), (181, 2, 182))
        self.assertEqual(a['successCoverage'], 181 / 183)
        self.assertAlmostEqual(a['inferenceTotalSec'], 181 * .25 + .5)
        self.assertEqual(result['summary']['runs']['b']['successfulFrames'], 183)
        self.assertEqual([w['runs']['a']['requestedFrames'] for w in result['windows']], [61] * 3)
        row = result['windows'][0]['frames'][30]['runs']['a']
        self.assertIsNone(row['metrics']['elbows']['left']['angleDeg'])
        self.assertEqual(row['metrics']['elbows']['left']['reason'], 'raw_frame_failed')
        self.assertFalse(result['releaseReady'])
        self.assertFalse(result['rawCoordinatesValidated'])
        self.assertFalse(result['physicalMotionMeasured'])
        self.assertEqual(result['accuracy']['status'], 'not_measurable')
        self.assertNotIn('winner', result)

    def test_zero_success_run_remains_comparable_and_not_decoder_validated(self):
        for window in self.b['windows']:
            for item in window['frames']:
                self.mark_failed('b', item['frame'], attempted=False)
        result = self.evaluate()
        b = result['summary']['runs']['b']
        self.assertEqual((b['successfulFrames'], b['failedFrames'], b['successCoverage']), (0, 15, 0))
        self.assertEqual(b['inferenceAttemptedFrames'], 0)
        self.assertFalse(b['decoderValidated'])
        for window in result['windows']:
            self.assertEqual(window['runs']['b']['temporal']['availableAdjacentPairs'], 0)
            self.assertEqual(window['imageXYDistancePx']['count'], 0)

    def test_metrics_use_common_explicit_camera_and_144_depth(self):
        result = self.evaluate()
        arrays = self.arrays['a'][0]
        z = (arrays['keypoints_simcc'][0, :, 2].astype(float) / 144 - 1) * evaluation.OFFICIAL_Z_RANGE
        valid = np.arange(133) < 17
        recovered, _ = recover_camera(arrays['keypoints_2d'][0], z, valid, self.camera)
        expected = pose_metrics(recovered['camera_root_relative_xyz'], valid)
        row = result['windows'][0]['frames'][0]
        self.assertEqual(row['runs']['a']['metrics'], expected)
        self.assertEqual(row['runs']['a']['metrics'], row['runs']['b']['metrics'])
        self.assertLess(row['runs']['a']['projectionRoundtripMaxPixelError'], 1e-7)
        self.assertEqual(row['rootCenteredRelativeZDifference']['maxAbsolute'], 0)
        self.b['windows'][0]['frames'][0]['rawSha256'] = 'f' * 64
        self.rejects('evidence_hash_mismatch')

    def test_adjacent_rates_use_actual_dt_and_never_bridge_failures_or_windows(self):
        self.mark_failed('a', 2)
        result = self.evaluate()
        windows = result['windows']
        temporal = windows[0]['runs']['a']['temporal']
        self.assertEqual(temporal['candidateAdjacentPairs'], 4)
        self.assertEqual(temporal['availableAdjacentPairs'], 2)
        self.assertEqual([(p['fromFrame'], p['toFrame'], p['available']) for p in temporal['pairs']],
                         [(0, 1, True), (1, 2, False), (2, 3, False), (3, 4, True)])
        for window in windows:
            for pair in window['runs']['a']['temporal']['pairs']:
                self.assertEqual(pair['toFrame'], pair['fromFrame'] + 1)
                dt = float(self.times[pair['toFrame']]) - float(self.times[pair['fromFrame']])
                self.assertEqual(pair['dtProxySec'], dt)
                if pair['available']:
                    self.assertAlmostEqual(pair['rootRelativeCoordinateChangePerProxySec']['meanAbsolute'],
                                           pair['rootRelativeCoordinateDisplacement']['meanAbsolute'] / dt)
                    elbow = pair['elbows']['left']
                    self.assertAlmostEqual(elbow['absoluteAngleChangeDegPerProxySec'], elbow['absoluteAngleChangeDeg'] / dt)
        self.assertEqual(sum(w['runs']['a']['temporal']['candidateAdjacentPairs'] for w in windows), 12)

    def test_unavailable_camera_geometry_has_null_metrics_and_breaks_adjacent_pairs(self):
        arrays = self.arrays['a'][2]
        arrays['scores'][0, 11] = 0
        arrays['keypoints_simcc'][0, 11] = -.5
        arrays['keypoints'][0, 11] = [-.5, -.5, (-.5 / 192 - 1) * evaluation.OFFICIAL_Z_RANGE]
        self.write_raw('a', self.a['windows'][0], self.a['windows'][0]['frames'][2])
        result = self.evaluate()
        window = result['windows'][0]
        self.assertEqual(window['runs']['a']['successfulFrames'], 5)
        self.assertEqual(window['runs']['a']['cameraRecoveredFrames'], 4)
        row = window['frames'][2]['runs']['a']
        self.assertEqual(row['cameraErrorCode'], 'root_hips_missing')
        self.assertIsNone(row['metrics']['elbows']['left']['angleDeg'])
        self.assertEqual(window['runs']['a']['temporal']['pairs'][1]['reason'], 'camera_geometry_unavailable')
        self.assertEqual(window['runs']['a']['temporal']['availableAdjacentPairs'], 2)

    def test_source_hashes_and_equal_crop_are_required(self):
        original = copy.deepcopy(self.b)
        cases = [(lambda r: r['input'].update(keypoints2dSha256='f' * 64), 'input_evidence_mismatch'),
                 (lambda r: r['input'].update(proxyClockValidation={}), 'proxy_clock_metadata_mismatch'),
                 (lambda r: r['windows'][0]['frames'][0].update(bboxXYXY=[1, 0, 50, 80]), 'frame_crop_mismatch'),
                 (lambda r: r['bboxPolicy'].update(extentPaddingRatio=.2), 'bbox_policy_mismatch'),
                 (lambda r: r['windows'][0]['frames'].pop(), 'incomplete_window_frame_coverage')]
        for mutate, code in cases:
            with self.subTest(code=code):
                self.b = copy.deepcopy(original)
                mutate(self.b)
                self.rejects(code)
        self.b = original
        self.fixture.fixture.proxy.write_bytes(b'changed proxy bytes')
        self.rejects('input_hash_mismatch')

    def test_matching_recorded_clock_does_not_override_actual_proxy_clock(self):
        actual = {'width': 100, 'height': 100, 'timestamps': (self.times * 1.01).tolist()}
        with patch('pose3d_probe.probe_proxy', return_value=actual):
            self.rejects('proxy_timestamp_mismatch', validate_proxy_clock)

    def test_report_selection_is_bound_to_hashed_spec_and_camera_is_explicit(self):
        for report in (self.a, self.b):
            report['windows'][0]['selectionReason'] = 'rewritten without changing spec'
        self.rejects('window_spec_selection_mismatch')
        for report in (self.a, self.b):
            report['windows'][0]['selectionReason'] = 'Synthetic selection without human annotation'
        self.camera.pop('rootDepth')
        self.config_path.write_text(json.dumps({'schemaVersion': CONFIG_SCHEMA, 'scenarios': [self.camera]}))
        self.rejects('camera_parameters_required')

    def test_cli_preserves_existing_and_concurrently_created_report(self):
        self.write_reports()
        original = self.paths[0].read_bytes()
        argv = ['evaluate', *(str(path) for path in self.paths), '--camera-config', str(self.config_path),
                '--scenario', 'explicit', '--output', str(self.paths[0])]
        with patch.object(sys, 'argv', argv), patch.object(evaluation, 'evaluate_windows') as evaluate, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(evaluation.main(), 2)
            evaluate.assert_not_called()
        self.assertEqual(self.paths[0].read_bytes(), original)
        output = self.root / 'result.json'
        argv[-1] = str(output)

        def raced(*_args):
            output.write_text('other research writer')
            return {'status': 'research_windows_evaluated'}

        with patch.object(sys, 'argv', argv), patch.object(evaluation, 'evaluate_windows', side_effect=raced), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(evaluation.main(), 2)
        self.assertEqual(output.read_text(), 'other research writer')
        self.assertEqual(list(self.root.glob('.result.json.*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
