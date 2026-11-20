"""Transport and raw-shape tests, independent of installing MMPose in the 2D venv."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import test_probe as fixtures
from pose3d_mmpose_probe import (ProbeError, load_mmpose, validate_prediction,
                                 validate_raw_arrays, run_probe, RAW_SHAPES)


def raw_arrays():
    return {name: np.full(shape, 1.2 if name == 'scores' else 3, dtype=np.float32)
            for name, shape in RAW_SHAPES.items()}


class OutputContractTests(unittest.TestCase):
    def test_official_decoder_output_is_kept_separate_from_assumed_camera_coordinates(self):
        xyz = np.ones((1, 133, 3), dtype=np.float32)
        simcc = xyz * 100
        decoded = [{'keypoints': xyz, 'keypoints_simcc': simcc, 'keypoint_scores': xyz[:, :, 0] * 1.2}]
        final = SimpleNamespace(keypoints=xyz * 3, transformed_keypoints=xyz[:, :, :2] * 200)
        output = validate_prediction(decoded, [SimpleNamespace(pred_instances=final)])
        np.testing.assert_array_equal(output['decoded_keypoints'], xyz)
        np.testing.assert_array_equal(output['official_camera_keypoints'], xyz * 3)
        self.assertGreater(output['scores'].mean(), 1)
        xyz[:] = 0
        self.assertTrue((output['decoded_keypoints'] == 1).all())

    def test_single_bbox_cannot_silently_select_one_of_multiple_players(self):
        for decoded, samples in [([], []), ([{}, {}], [object(), object()])]:
            with self.assertRaises(ProbeError) as error:
                validate_prediction(decoded, samples)
            self.assertEqual(error.exception.code, 'invalid_model_output')

    def test_shapes_nan_and_missing_outputs_rejected(self):
        for mutate in (lambda a: a.pop('keypoints_simcc'),
                       lambda a: a.update(scores=np.ones((1, 65))),
                       lambda a: a['decoded_keypoints'].__setitem__((0, 1, 2), np.nan)):
            arrays = raw_arrays()
            mutate(arrays)
            with self.assertRaises(ProbeError):
                validate_raw_arrays(arrays)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProbeTests()
        self.fixture.setUp()
        self.args = self.fixture.args
        self.args.checkpoint = str(self.fixture.directory / 'model.pth')
        Path(self.args.checkpoint).write_bytes(b'synthetic-test-only')
        self.args.config = str(self.fixture.directory / 'config.py')
        Path(self.args.config).write_text('# synthetic transport configuration')
        self.args.source_root = str(self.fixture.directory / 'source')
        self.args.threads = 4
        self.calls = []

    def tearDown(self):
        self.fixture.tearDown()

    def loader(self, config, checkpoint, source, threads):
        self.calls.append(('load', str(config), str(checkpoint), source, threads))

        def infer(frame, bbox):
            self.calls.append(('infer', int(frame[0, 0, 0]), list(bbox)))
            return raw_arrays()

        return infer, {'runtime': 'synthetic-unit-test-only', 'cameraAssumptions': {'calibrated': False}}

    def run_probe(self):
        return run_probe(self.args, self.loader, fixtures.FakeCV())

    def test_pipeline_writes_all_raw_spaces_and_no_product_f32(self):
        report = self.run_probe()
        self.assertEqual(report['status'], 'raw_evidence_collected', report)
        self.assertFalse(report['releaseReady'])
        self.assertFalse(report['rawCoordinatesValidated'])
        self.assertFalse(report['model']['cameraAssumptions']['calibrated'])
        self.assertEqual([call[1] for call in self.calls if call[0] == 'infer'], [0, 2])
        output = Path(self.args.output_dir)
        with np.load(output / 'frame-000002.mmpose-raw.npz', allow_pickle=False) as archive:
            self.assertTrue(all(key in archive for key in RAW_SHAPES))
            self.assertEqual(int(archive['frame_index']), 2)
            self.assertEqual(float(archive['timestamp_sec']), float(np.float32(.04)))
        self.assertEqual(list(output.glob('*.f32')), [])
        clock = report['input']['proxyClockValidation']
        self.assertEqual(clock['source'], 'ffprobe-best_effort_timestamp_time')
        self.assertLessEqual(clock['maxTimestampErrorSec'], clock['toleranceSec'])

    def test_missing_checkpoint_and_remote_url_fail_before_runtime_import(self):
        for index, checkpoint in enumerate(('https://example.com/model.pth', str(self.fixture.directory / 'absent.pth'))):
            self.args.checkpoint = checkpoint
            self.args.output_dir = str(self.fixture.directory / f'missing-{index}')
            self.assertEqual(self.run_probe()['errorCode'], 'model_unavailable')
        self.assertEqual(self.calls, [])

    def test_wrong_video_authorization_blocks_inference(self):
        self.fixture.input.write_bytes(b'not the authorized proxy')
        self.assertEqual(self.run_probe()['errorCode'], 'input_not_authorized')
        self.assertEqual(self.calls, [])

    def test_wrong_capture_clock_blocks_inference(self):
        np.array([0, .02, .08], dtype='<f4').tofile(self.fixture.times)
        self.assertEqual(self.run_probe()['errorCode'], 'proxy_timestamp_mismatch')
        self.assertEqual(self.calls, [])

    def test_decode_count_and_low_quality_bbox_fail_without_selecting_other_person(self):
        with patch('pose3d_mmpose_probe.decode_frames', side_effect=ProbeError('frame_count_mismatch', 'test')):
            self.assertEqual(self.run_probe()['errorCode'], 'frame_count_mismatch')
        self.args.output_dir = str(self.fixture.directory / 'missing-target')
        points = np.fromfile(self.fixture.points, dtype='<f4').reshape(3, 65, 3)
        points[:, :17, 2] = 0
        points.tofile(self.fixture.points)
        self.assertEqual(self.run_probe()['errorCode'], 'target_bbox_unavailable')
        self.assertEqual(self.calls, [])

    def test_dependency_failure_is_recorded_with_traceback_not_a_synthetic_result(self):
        def unavailable(*_args):
            raise ModuleNotFoundError('mmcv._ext')

        report = run_probe(self.args, unavailable, fixtures.FakeCV())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['frames'], [])
        self.assertIn('mmcv._ext', report['error'])
        self.assertTrue((Path(self.args.output_dir) / 'failure-traceback.txt').is_file())
        self.assertEqual(list(Path(self.args.output_dir).glob('*.npz')), [])

    def test_invalid_runtime_output_does_not_publish_raw_evidence(self):
        report = run_probe(self.args, lambda *_args: (lambda *_args: {}, {}), fixtures.FakeCV())
        self.assertEqual(report['errorCode'], 'invalid_model_output')
        self.assertEqual(report['frames'], [])

    def test_existing_output_is_never_replaced(self):
        self.run_probe()
        path = Path(self.args.output_dir) / 'report.json'
        original = path.read_bytes()
        with self.assertRaises(ProbeError):
            self.run_probe()
        self.assertEqual(path.read_bytes(), original)

    def test_real_loader_rejects_missing_local_inputs_without_importing_torch(self):
        with self.assertRaises(ProbeError) as error:
            load_mmpose('https://example.com/config.py', self.args.checkpoint, self.args.source_root, 4)
        self.assertEqual(error.exception.code, 'config_unavailable')


if __name__ == '__main__':
    unittest.main()
