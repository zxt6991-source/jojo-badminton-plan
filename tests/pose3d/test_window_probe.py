"""Window transport tests; no actual model weights or 3D accuracy claims."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from pose3d_probe import ProbeError, digest, OUTPUT_SHAPES
from pose3d_mmpose_probe import RAW_SHAPES
from pose3d_window_probe import parser, run_probe
from pose3d_window_spec import validate_window_report_selection


class StreamingCV:
    def __init__(self, count=30, wrong_dimensions_at=None):
        self.count = count
        self.wrong_dimensions_at = wrong_dimensions_at
        self.open_count = 0
        self.read_count = 0
        self.released = False
        self.previous = None
        self.retained_previous = False

    def VideoCapture(self, _path):
        self.open_count += 1
        return self

    def isOpened(self):
        return True

    def read(self):
        if self.previous is not None and self.previous() is not None:
            self.retained_previous = True
        index = self.read_count
        self.read_count += 1
        if index >= self.count:
            return False, None
        height = 31 if index == self.wrong_dimensions_at else 32
        image = np.empty((height, 48, 3), dtype=np.uint8)
        image[:] = [index, index + 40, index + 80]
        self.previous = weakref.ref(image)
        return True, image

    def release(self):
        self.released = True


class WindowProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name).resolve()
        self.video = self.directory / 'proxy.mp4'
        self.video.write_bytes(b'synthetic authorized capture proxy for transport test')
        self.times = np.arange(30, dtype=np.float32) / 30
        self.times_path = self.directory / 'timestamps.f32'
        self.times.tofile(self.times_path)
        self.points = np.zeros((30, 65, 3), dtype='<f4')
        self.points[:, :, 0] = np.linspace(.2, .8, 65)
        self.points[:, :, 1] = np.linspace(.3, .7, 65)
        self.points[:, :, 2] = .9
        self.points_path = self.directory / 'keypoints.f32'
        self.points.tofile(self.points_path)
        self.authorization = self.directory / 'authorization.json'
        self.authorization.write_text(json.dumps({
            'schemaVersion': 'pose3d-local-authorization-0.1.0',
            'scope': 'local-internal-3d-validation', 'externalUploadAllowed': False,
            'userStatement': 'local synthetic test fixture only', 'authorizedAt': '2026-11-16',
            'retentionUntil': None, 'items': [{'videoId': 'synthetic', 'originalPath': 'unused',
                'proxyPath': str(self.video), 'originalSha256': '1' * 64,
                'proxySha256': digest(self.video)}]}))
        self.spec = self.directory / 'windows.json'
        selections = [(0, 4), (8, 12), (20, 25)]
        self.selected = [i for start, end in selections for i in range(start, end + 1)]
        self.spec.write_text(json.dumps({'schemaVersion': 'pose3d-research-windows-0.1.0',
            'purpose': 'synthetic research transport only', 'windows': [
                {'id': f'window-{n}', 'startTimeSec': float(self.times[start]),
                 'endTimeSec': float(self.times[end]), 'anchorFrame': start + 2,
                 'selectionReason': 'synthetic selection', 'humanVerified': False}
                for n, (start, end) in enumerate(selections)]}))
        self.onnx = self.directory / 'model.onnx'
        self.checkpoint = self.directory / 'model.pth'
        self.config = self.directory / 'config.py'
        for path in (self.onnx, self.checkpoint, self.config):
            path.write_bytes(b'synthetic test only, never loaded as a real model')
        self.args = parser().parse_args([
            '--runtime', 'rtmlib-onnx', '--input', str(self.video),
            '--timestamps', str(self.times_path), '--keypoints2d', str(self.points_path),
            '--authorization-evidence', str(self.authorization), '--window-spec', str(self.spec),
            '--output-dir', str(self.directory / 'output'), '--pose-model', str(self.onnx),
            '--checkpoint', str(self.checkpoint), '--config', str(self.config),
            '--source-root', str(self.directory), '--color-order', 'rgb'])
        self.loads = []
        self.inferences = []
        self.cv = StreamingCV()
        self.fail_indices = set()
        self.invalid_indices = set()

    def tearDown(self):
        self.temp.cleanup()

    def record_call(self, image, bbox):
        index = int(image[0, 0].min())
        self.inferences.append((index, tuple(int(v) for v in image[0, 0]), list(bbox)))
        if index in self.fail_indices:
            raise RuntimeError('synthetic model failure')
        return index

    def rtmlib_loader(self, path, device):
        self.loads.append(('rtmlib-onnx', path, device))

        def infer(image, bboxes):
            index = self.record_call(image, bboxes[0])
            if index in self.invalid_indices:
                return ()
            return tuple(np.full(shape, index + 1.25, dtype=np.float32) for shape in OUTPUT_SHAPES)

        return infer, {'apiVersion': 'synthetic-test-only', 'executionProviders': ['synthetic-test-only']}

    def mmpose_loader(self, config, checkpoint, source, threads):
        self.loads.append(('mmpose', config, checkpoint, source, threads))

        def infer(image, bbox):
            index = self.record_call(image, bbox)
            if index in self.invalid_indices:
                return {}
            return {name: np.full(shape, index + 1.25, dtype=np.float32) for name, shape in RAW_SHAPES.items()}

        return infer, {'runtime': 'synthetic-test-only', 'cameraAssumptions': {'calibrated': False}}

    def clock(self, path, times):
        self.assertEqual(path, self.video)
        np.testing.assert_array_equal(times, self.times)
        return {'source': 'synthetic-clock-validator-test-only', 'frameCount': len(times),
                'width': 48, 'height': 32, 'maxTimestampErrorSec': 0.0, 'toleranceSec': .002}

    def run_probe(self, **overrides):
        kwargs = {'rtmlib_loader': self.rtmlib_loader, 'mmpose_loader': self.mmpose_loader,
                  'cv_module': self.cv, 'clock_validator': self.clock}
        kwargs.update(overrides)
        return run_probe(self.args, **kwargs)

    def records(self, report):
        return [item for window in report['windows'] for item in window['frames']]

    def test_rtmlib_loads_once_streams_all_frames_and_preserves_raw_contract(self):
        report = self.run_probe()
        self.assertEqual(report['status'], 'raw_evidence_collected', report)
        self.assertEqual(report['selectedFrames'], 16)  # exceeds legacy 10-frame probe limit
        self.assertEqual(len(self.loads), 1)
        self.assertEqual([item[0] for item in self.inferences], self.selected)
        self.assertEqual(self.inferences[0][1], (80, 40, 0))  # explicit RGB conversion
        self.assertEqual(self.cv.read_count, 31)
        self.assertTrue(self.cv.released)
        self.assertFalse(self.cv.retained_previous)
        self.assertTrue(report['decodeCompleted'])
        self.assertEqual(report['decodedFrames'], 30)
        self.assertEqual(report['windowSpecSha256'], digest(self.spec))
        self.assertFalse(report['rawCoordinatesValidated'])
        self.assertFalse(report['releaseReady'])
        validate_window_report_selection(report, self.times)
        output = Path(self.args.output_dir)
        for window in report['windows']:
            for item in window['frames']:
                self.assertNotIn('errorCode', item)
                self.assertEqual(item['rawPath'], f"frame-{item['frame']:06d}.raw.npz")
                raw = output / window['id'] / item['rawPath']
                self.assertEqual(item['rawSha256'], digest(raw))
                with np.load(raw, allow_pickle=False) as archive:
                    self.assertEqual(set(archive.files), {'keypoints', 'scores', 'keypoints_simcc',
                        'keypoints_2d', 'frame_index', 'timestamp_sec', 'bbox_xyxy'})
                    self.assertEqual(int(archive['frame_index']), item['frame'])
                    self.assertEqual(float(archive['timestamp_sec']), float(self.times[item['frame']]))
                    np.testing.assert_array_equal(archive['bbox_xyxy'], item['bboxXYXY'])
                    self.assertTrue((archive['scores'] == item['frame'] + 1.25).all())
        self.assertEqual(list(output.rglob('*.f32')), [])
        self.assertEqual(list(output.rglob('*.png')), [])

    def test_mmpose_uses_original_bgr_and_its_independent_raw_spaces(self):
        self.args.runtime, self.args.color_order = 'mmpose', 'bgr'
        report = self.run_probe()
        self.assertEqual(report['status'], 'raw_evidence_collected', report)
        self.assertEqual(len(self.loads), 1)
        self.assertEqual(self.loads[0][0], 'mmpose')
        self.assertEqual(self.inferences[0][1], (0, 40, 80))
        first = report['windows'][0]['frames'][0]
        self.assertEqual(first['rawPath'], 'frame-000000.mmpose-raw.npz')
        with np.load(Path(self.args.output_dir) / 'window-0' / first['rawPath'], allow_pickle=False) as archive:
            self.assertTrue(all(name in archive for name in RAW_SHAPES))
            self.assertNotIn('keypoints', archive)
        validate_window_report_selection(report, self.times)

    def test_missing_bbox_failed_inference_and_bad_shape_remain_failures_then_continue(self):
        self.points[2, :17, 2] = 0
        self.points.tofile(self.points_path)
        self.fail_indices = {9}
        self.invalid_indices = {10}
        report = self.run_probe()
        self.assertEqual(report['status'], 'raw_evidence_collected_with_failures', report)
        items = {item['frame']: item for item in self.records(report)}
        self.assertEqual(items[2]['errorCode'], 'target_bbox_unavailable')
        self.assertIsNone(items[2]['bboxXYXY'])
        self.assertFalse(items[2]['inferenceAttempted'])
        self.assertIsNone(items[2]['inferenceSec'])
        for index in (9, 10):
            self.assertIsNotNone(items[index]['bboxXYXY'])
            self.assertTrue(items[index]['inferenceAttempted'])
            self.assertGreaterEqual(items[index]['inferenceSec'], 0)
        self.assertEqual(items[10]['errorCode'], 'invalid_model_output')
        for index in (2, 9, 10):
            self.assertEqual(items[index]['status'], 'failed')
            self.assertEqual(items[index]['captureTimeSec'], float(self.times[index]))
            for field in ('rawPath', 'rawSha256', 'outputShapes'):
                self.assertIsNone(items[index][field])
        self.assertEqual(items[11]['status'], 'succeeded')
        self.assertEqual(report['failedFrames'], 3)
        self.assertEqual(len(list(Path(self.args.output_dir).rglob('*.npz'))), 13)
        validate_window_report_selection(report, self.times)

    def test_every_failed_frame_still_has_a_complete_selection_denominator(self):
        self.fail_indices = set(self.selected)
        report = self.run_probe()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['errorCode'], 'no_successful_frames')
        self.assertEqual(report['failedFrames'], 16)
        self.assertEqual(len(self.inferences), 16)
        self.assertTrue(report['decodeCompleted'])
        validate_window_report_selection(report, self.times)

    def test_failed_raw_write_is_not_success_and_later_frames_continue(self):
        save_archive = np.savez_compressed

        def fail_one_write(stream, **arrays):
            if int(arrays['frame_index']) == 9:
                stream.write(b'incomplete archive for storage failure test')
                raise OSError('synthetic storage failure')
            return save_archive(stream, **arrays)

        with patch('pose3d_window_probe.np.savez_compressed', side_effect=fail_one_write):
            report = self.run_probe()
        items = {item['frame']: item for item in self.records(report)}
        failed = items[9]
        self.assertEqual(report['status'], 'raw_evidence_collected_with_failures')
        self.assertEqual((report['succeededFrames'], report['failedFrames']), (15, 1))
        self.assertTrue(failed['inferenceAttempted'])
        self.assertGreaterEqual(failed['inferenceSec'], 0)
        for field in ('rawPath', 'rawSha256', 'outputShapes'):
            self.assertIsNone(failed[field])
        self.assertEqual(items[10]['status'], 'succeeded')
        self.assertTrue(report['decodeCompleted'])
        validate_window_report_selection(report, self.times)

    def test_global_decode_failure_preserves_successes_and_marks_unprocessed_frames(self):
        self.cv.count = 11
        report = self.run_probe()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['errorCode'], 'frame_count_mismatch')
        self.assertEqual(report['succeededFrames'], 8)
        self.assertFalse(report['decodeCompleted'])
        unprocessed = [item for item in self.records(report) if item['frame'] >= 11]
        self.assertTrue(all(item['errorCode'] == 'frame_not_processed' for item in unprocessed))
        self.assertTrue(all(item['inferenceSec'] is None for item in unprocessed))
        self.assertTrue(self.cv.released)
        validate_window_report_selection(report, self.times)

    def test_decode_dimensions_are_checked_even_on_unselected_frames(self):
        self.cv.wrong_dimensions_at = 6
        report = self.run_probe()
        self.assertEqual(report['errorCode'], 'proxy_dimensions_mismatch')
        self.assertTrue(self.cv.released)
        self.assertEqual([item[0] for item in self.inferences], list(range(5)))

    def test_missing_local_model_is_not_downloaded_or_imported(self):
        self.args.pose_model = 'https://example.invalid/model.onnx'
        report = self.run_probe()
        self.assertEqual(report['errorCode'], 'model_unavailable')
        self.assertEqual(self.loads, [])
        self.assertEqual(self.cv.open_count, 0)
        self.assertEqual(report['failedFrames'], 16)

    def test_changed_input_authorization_prevents_model_loading(self):
        self.video.write_bytes(b'unauthorized video')
        report = self.run_probe()
        self.assertEqual(report['errorCode'], 'input_not_authorized')
        self.assertEqual(self.loads, [])
        self.assertEqual(self.cv.open_count, 0)

    def test_clock_validation_failure_prevents_runtime_and_keeps_capture_times(self):
        def wrong_clock(*_args):
            raise ProbeError('proxy_timestamp_mismatch', 'test clock mismatch')

        report = self.run_probe(clock_validator=wrong_clock)
        self.assertEqual(report['errorCode'], 'proxy_timestamp_mismatch')
        self.assertEqual(self.loads, [])
        validate_window_report_selection(report, self.times)

    def test_invalid_window_is_rejected_before_runtime_loading(self):
        spec = json.loads(self.spec.read_text())
        spec['windows'][0]['id'] = '../outside'
        self.spec.write_text(json.dumps(spec))
        report = self.run_probe()
        self.assertEqual(report['errorCode'], 'invalid_window_id')
        self.assertEqual(self.loads, [])
        self.assertFalse((self.directory / 'outside').exists())

    def test_load_failure_is_evidence_of_failure_never_synthetic_success(self):
        def missing_dependency(*_args):
            raise ModuleNotFoundError('actual_runtime_dependency')

        report = self.run_probe(rtmlib_loader=missing_dependency)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['failedFrames'], 16)
        self.assertEqual(self.cv.open_count, 0)
        self.assertGreaterEqual(report['modelLoadSec'], 0)
        self.assertTrue((Path(self.args.output_dir) / 'failure-traceback.txt').is_file())
        self.assertEqual(list(Path(self.args.output_dir).rglob('*.npz')), [])
        validate_window_report_selection(report, self.times)

    def test_output_is_exclusive_even_if_existing_empty_or_symlink(self):
        output = Path(self.args.output_dir)
        output.mkdir()
        with self.assertRaises(ProbeError) as error:
            self.run_probe()
        self.assertEqual(error.exception.code, 'output_exists')
        self.assertEqual(list(output.iterdir()), [])
        link = self.directory / 'link'
        link.symlink_to(output, target_is_directory=True)
        self.args.output_dir = str(link)
        with self.assertRaises(ProbeError):
            self.run_probe()
        self.assertEqual(list(output.iterdir()), [])

    def test_second_run_cannot_replace_existing_report(self):
        self.run_probe()
        path = Path(self.args.output_dir) / 'report.json'
        original = path.read_bytes()
        with self.assertRaises(ProbeError):
            self.run_probe()
        self.assertEqual(path.read_bytes(), original)

    def test_mmpose_rejects_double_color_conversion(self):
        self.args.runtime = 'mmpose'
        report = self.run_probe()
        self.assertEqual(report['errorCode'], 'mmpose_color_override_forbidden')
        self.assertEqual(self.loads, [])

    def test_timing_totals_account_for_actual_attempts_and_report_boundaries(self):
        self.fail_indices = {9}
        report = self.run_probe()
        items = self.records(report)
        self.assertAlmostEqual(report['inferenceSec'], sum(item['inferenceSec'] or 0 for item in items))
        self.assertAlmostEqual(report['inferenceCpuSec'], sum(item['inferenceCpuSec'] or 0 for item in items))
        self.assertGreaterEqual(report['decodeSec'], sum(item['decodeSec'] or 0 for item in items))
        self.assertGreaterEqual(report['elapsedSec'], report['modelLoadSec'] + report['inferenceSec'])
        self.assertGreater(report['peakProcessRamBytes'], 0)
        self.assertIn('failed attempts included', report['timingBoundaries']['inferenceSec'])
        for window in report['windows']:
            self.assertAlmostEqual(window['inferenceSec'], sum(item['inferenceSec'] or 0 for item in window['frames']))
            self.assertGreaterEqual(window['elapsedSec'], window['inferenceSec'])


if __name__ == '__main__':
    unittest.main()
