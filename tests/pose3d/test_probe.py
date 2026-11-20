"""Probe transport/authorization tests; mocks are never model validation evidence."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from pose3d_probe import (ProbeError, parse_frames, parse_bbox, authorize, digest,
                         bbox_from_2d, validate_outputs, decode_frames, run_probe, load_rtmlib, validate_proxy_clock)


class FakeCapture:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.closed = False

    def isOpened(self):
        return True

    def read(self):
        frame = next(self.frames, None)
        return frame is not None, frame

    def release(self):
        self.closed = True


class FakeCV:
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self, frame_count=3):
        self.capture = FakeCapture([np.full((64, 96, 3), index, dtype=np.uint8) for index in range(frame_count)])

    def VideoCapture(self, _path):
        return self.capture

    def rectangle(self, *_args):
        pass

    circle = rectangle
    line = rectangle
    putText = rectangle

    def imwrite(self, path, image):
        Path(path).write_bytes(image.tobytes())
        return True


def arrays():
    xyz = np.ones((1, 133, 3), dtype=np.float32)
    xyz[..., 0] = np.arange(133)
    xyz[..., 2] = np.linspace(-2, 2, 133)
    return xyz, np.full((1, 133), 1.2, dtype=np.float32), xyz.copy() * 2, np.full((1, 133, 2), 32, dtype=np.float32)


class FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, image, bboxes):
        self.calls.append((int(image[0, 0, 0]), bboxes))
        return arrays()


class ArgumentTests(unittest.TestCase):
    def test_probe_frame_cap_and_no_duplicates_or_negative_values(self):
        self.assertEqual(parse_frames('8,0,2'), [0, 2, 8])
        for value in ('', '-1', '1,1', '1.5', '0,', ','.join(str(v) for v in range(11)), '20000'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_frames(value)

    def test_explicit_bbox_has_positive_finite_area(self):
        self.assertEqual(parse_bbox('1,2,10,20'), (1, 2, 10, 20))
        for value in ('0,0,0,2', '-1,0,1,1', '0,0,nan,2', '0,1,2'):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_bbox(value)

    def test_same_frame_bbox_uses_visible_body_and_not_hand_or_other_person(self):
        pose = np.full((65, 3), .9)
        pose[:17, :2] = np.column_stack([np.linspace(.2, .5, 17), np.linspace(.3, .7, 17)])
        pose[0] = [0, 0, .1]  # invisible body outlier must not expand box
        box = bbox_from_2d(pose, 100, 200)
        self.assertGreater(box[0], 10)
        self.assertLess(box[2], 60)
        pose[:13, 2] = 0
        with self.assertRaises(ProbeError) as error:
            bbox_from_2d(pose, 100, 200)
        self.assertEqual(error.exception.code, 'target_bbox_unavailable')


class OutputTests(unittest.TestCase):
    def test_raw_tuple_preserves_unbounded_scores_negative_z_and_original_arrays(self):
        raw = arrays()
        result = validate_outputs(raw)
        self.assertGreater(result['scores'].max(), 1)
        self.assertLess(result['keypoints'][0, 0, 2], 0)
        np.testing.assert_array_equal(result['keypoints'], raw[0])
        self.assertFalse(np.shares_memory(result['keypoints'], raw[0]))

    def test_wrong_tuple_shape_and_nonfinite_outputs_fail(self):
        for value in (arrays()[:3], (np.zeros((133, 3)), *arrays()[1:]),
                      (arrays()[0], np.ones((1, 133), dtype=object), *arrays()[2:])):
            with self.assertRaises(ProbeError) as error:
                validate_outputs(value)
            self.assertEqual(error.exception.code, 'invalid_model_output')
        raw = arrays()
        raw[0][0, 0, 2] = np.nan
        with self.assertRaises(ProbeError):
            validate_outputs(raw)

    def test_decoding_uses_sequential_frame_indices_and_checks_complete_count(self):
        cv = FakeCV()
        frames = decode_frames('unused', {0, 2}, 3, cv)
        self.assertEqual([int(frames[i][0, 0, 0]) for i in frames], [0, 2])
        self.assertTrue(cv.capture.closed)
        cv = FakeCV(2)
        with self.assertRaises(ProbeError) as error:
            decode_frames('unused', {0}, 3, cv)
        self.assertEqual(error.exception.code, 'frame_count_mismatch')
        self.assertTrue(cv.capture.closed)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.input = self.directory / 'proxy.mp4'
        self.input.write_bytes(b'synthetic-test-proxy-no-real-inference')
        self.model_path = self.directory / 'test.onnx'
        self.model_path.write_bytes(b'not-an-actual-model')
        self.times = self.directory / 'timestamps.f32'
        np.array([0, .01, .04], dtype='<f4').tofile(self.times)
        self.points = self.directory / 'keypoints.f32'
        pose = np.full((3, 65, 3), .9, dtype='<f4')
        pose[:, :17, :2] = np.column_stack([np.linspace(.2, .5, 17), np.linspace(.3, .7, 17)])
        pose.tofile(self.points)
        self.authorization = self.directory / 'authorization.json'
        self.authorization_data = {'schemaVersion': 'pose3d-local-authorization-0.1.0',
            'scope': 'local-internal-3d-validation', 'externalUploadAllowed': False,
            'userStatement': 'test authorization, not real video evidence', 'authorizedAt': '2026-11-16',
            'retentionUntil': None, 'items': [{'videoId': 'test-video', 'originalPath': 'original.mp4',
             'originalSha256': 'a' * 64, 'proxyPath': str(self.input), 'proxySha256': digest(self.input)}]}
        self.authorization.write_text(json.dumps(self.authorization_data))
        self.args = SimpleNamespace(input=str(self.input), input_kind='capture-proxy', pose_model=str(self.model_path),
            timestamps=str(self.times), keypoints2d=str(self.points), frames=[0, 2], bbox=None,
            authorization_evidence=str(self.authorization), output_dir=str(self.directory / 'out'), device='cpu')
        self.model = FakeModel()
        self.factory_calls = []
        # Explicit test-only metadata mock. The production CLI has no bypass.
        self.clock_patch = patch('pose3d_probe.probe_proxy', return_value={
            'width': 96, 'height': 64, 'timestamps': [0, .01, .04]})
        self.clock_patch.start()

    def tearDown(self):
        self.clock_patch.stop()
        self.temporary.cleanup()

    def factory(self, path, device):
        self.factory_calls.append((path, device))
        return self.model, {'implementation': 'synthetic-test-only', 'executionProviders': ['test']}

    def run_probe(self):
        return run_probe(self.args, self.factory, FakeCV())

    def test_local_raw_probe_writes_tuple_provenance_and_never_product_binary(self):
        report = self.run_probe()
        self.assertEqual(report['status'], 'raw_evidence_collected', report)
        self.assertEqual(report['input']['proxyClockValidation']['source'], 'ffprobe-best_effort_timestamp_time')
        self.assertFalse(report['releaseReady'])
        self.assertFalse(report['productCompatible'])
        self.assertEqual(report['authorization']['matchingVideoId'], 'test-video')
        self.assertEqual(report['model']['weightsSha256'], digest(self.model_path))
        self.assertEqual(report['frames'][1]['captureTimeSec'], float(np.float32(.04)))
        self.assertEqual([call[0] for call in self.model.calls], [0, 2])
        output = Path(self.args.output_dir)
        with np.load(output / 'frame-000002.raw.npz', allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw['keypoints'], arrays()[0])
            self.assertEqual(int(raw['frame_index']), 2)
            self.assertEqual(raw['bbox_xyxy'].shape, (4,))
        self.assertFalse(any(output.glob('*.f32')))
        self.assertEqual(json.loads((output / 'report.json').read_text())['coordinateStatus'], 'unverified-raw-mixed-scales')

    def test_same_length_shifted_or_scaled_timestamps_fail_before_model(self):
        for name, timestamps in (('shifted', [1000, 1000.01, 1000.04]), ('scaled', [0, .1, .4])):
            np.array(timestamps, dtype='<f4').tofile(self.times)
            self.args.output_dir = str(self.directory / name)
            report = self.run_probe()
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['errorCode'], 'proxy_timestamp_mismatch')
        self.assertEqual(self.factory_calls, [])

    def test_ffprobe_unavailable_never_falls_back_to_count_only(self):
        with patch('pose3d_probe.probe_proxy', side_effect=FileNotFoundError('ffprobe unavailable')):
            report = self.run_probe()
        self.assertEqual(report['errorCode'], 'proxy_clock_unavailable')
        self.assertEqual(self.factory_calls, [])

    def test_ffprobe_frame_count_and_decoded_dimensions_must_match(self):
        cases = [('count', {'width': 96, 'height': 64, 'timestamps': [0, .01]}, 'proxy_clock_frame_count_mismatch'),
                 ('dimensions', {'width': 100, 'height': 64, 'timestamps': [0, .01, .04]}, 'proxy_dimensions_mismatch')]
        for name, metadata, code in cases:
            self.args.output_dir = str(self.directory / name)
            with patch('pose3d_probe.probe_proxy', return_value=metadata):
                self.assertEqual(self.run_probe()['errorCode'], code)
        self.assertEqual(self.factory_calls, [])

    def test_missing_model_and_url_are_rejected_before_model_factory(self):
        for model in (str(self.directory / 'absent.onnx'), 'https://example.com/model.onnx'):
            self.args.pose_model = model
            self.args.output_dir = str(self.directory / ('out-' + str(len(self.factory_calls)) + str(len(model))))
            report = self.run_probe()
            self.assertEqual(report['errorCode'], 'model_unavailable')
        self.assertEqual(self.factory_calls, [])

    def test_input_hash_must_match_authorized_proxy_not_original(self):
        self.authorization_data['items'][0]['proxySha256'] = 'f' * 64
        self.authorization_data['items'][0]['originalSha256'] = digest(self.input)
        self.authorization.write_text(json.dumps(self.authorization_data))
        report = self.run_probe()
        self.assertEqual(report['errorCode'], 'input_not_authorized')
        self.assertEqual(self.factory_calls, [])

    def test_external_upload_scope_or_duplicate_authorization_rejected(self):
        for mutation in (lambda x: x.update(externalUploadAllowed=True),
                         lambda x: x['items'].append(copy.deepcopy(x['items'][0]))):
            document = copy.deepcopy(self.authorization_data)
            mutation(document)
            self.authorization.write_text(json.dumps(document))
            with self.assertRaises(ProbeError):
                authorize(self.authorization, digest(self.input))

    def test_bad_timestamps_out_of_bounds_and_bad_bbox_fail_before_loading(self):
        cases = [('bad-times', lambda: np.array([0, .1, .1], dtype='<f4').tofile(self.times), 'invalid_timestamps'),
                 ('bad-frame', lambda: setattr(self.args, 'frames', [3]), 'invalid_frames'),
                 ('bad-bbox', lambda: setattr(self.args, 'bbox', [0, 0, 200, 200]), 'invalid_bbox')]
        for label, mutation, code in cases:
            np.array([0, .01, .04], dtype='<f4').tofile(self.times)
            self.args.frames, self.args.bbox = [0, 2], None
            self.args.output_dir = str(self.directory / label)
            mutation()
            self.assertEqual(self.run_probe()['errorCode'], code)
        self.assertEqual(self.factory_calls, [])

    def test_failed_model_output_writes_failure_report_and_no_product_data(self):
        report = run_probe(self.args, lambda _p, _d: (lambda *_a, **_k: (np.zeros((2,)),), {}), FakeCV())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['errorCode'], 'invalid_model_output')
        self.assertEqual(report['frames'], [])
        self.assertEqual(list(Path(self.args.output_dir).glob('*.npz')), [])

    def test_explicit_rgb_probe_swaps_input_channels_and_records_override(self):
        self.args.color_order = 'rgb'
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        image[:] = [10, 20, 30]
        cv = FakeCV()
        cv.capture = FakeCapture([image.copy() for _ in range(3)])
        report = run_probe(self.args, self.factory, cv)
        self.assertEqual(report['status'], 'raw_evidence_collected')
        self.assertEqual(self.model.calls[0][0], 30)
        self.assertEqual(report['preprocess']['passedToLibraryColor'], 'rgb')

    def test_existing_evidence_directory_is_never_overwritten(self):
        self.run_probe()
        original = (Path(self.args.output_dir) / 'report.json').read_bytes()
        with self.assertRaises(ProbeError) as error:
            self.run_probe()
        self.assertEqual(error.exception.code, 'output_not_empty')
        self.assertEqual((Path(self.args.output_dir) / 'report.json').read_bytes(), original)

    def test_local_loader_rejects_missing_model_before_import_or_download(self):
        with patch('pose3d_probe.importlib.metadata.version', side_effect=AssertionError('must not import runtime')):
            with self.assertRaises(ProbeError) as error:
                load_rtmlib(self.directory / 'missing.onnx', 'cpu')
            self.assertEqual(error.exception.code, 'model_unavailable')


if __name__ == '__main__':
    unittest.main()
