"""Synthetic arithmetic and evidence-integrity tests, not real model evidence."""
import json
from pathlib import Path
import sys
import subprocess
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import test_probe as fixtures
from pose3d_probe import run_probe, digest, ProbeError
from pose3d_probe_compare import compare_reports


class ProbeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProbeTests()
        self.fixture.setUp()
        self.directory = self.fixture.directory
        self.paths = {}

        def factory(path, device):
            return self.fixture.model, {'implementation': 'synthetic-test-only', 'sourceFileSha256': 'c' * 64,
                'apiVersion': 'test', 'modelInputSize': [288, 384], 'backend': 'test', 'requestedDevice': device,
                'zRange': 2.1744869,
                'executionProviders': ['test'], 'upstreamDecode': {'isotropicXYZVerified': False},
                'onnxInputs': [{'shape': [1, 3, 384, 288]}], 'onnxOutputs': [{'shape': [1, 133, 576]}]}

        for color in ('bgr', 'rgb'):
            args = self.fixture.args
            args.color_order, args.output_dir = color, str(self.directory / color)
            report = run_probe(args, factory, fixtures.FakeCV())
            self.assertEqual(report['status'], 'raw_evidence_collected')
            self.paths[color] = Path(args.output_dir) / 'report.json'
        for color in ('bgr', 'rgb'):
            self.edit_arrays(color, lambda arrays, c=color: self.controlled_arrays(arrays, c))

    def tearDown(self):
        self.fixture.tearDown()

    def controlled_arrays(self, arrays, color):
        baseline = np.fromfile(self.fixture.points, dtype='<f4').reshape(3, 65, 3)
        frame = int(arrays['frame_index'])
        arrays['keypoints_2d'][0, :17] = baseline[frame, :17, :2] * [96, 64]
        arrays['scores'][:] = 1.2
        if color == 'bgr':
            arrays['scores'][0, 0] = .3
        else:
            arrays['keypoints_2d'][0, :17] += [3, 4]
            arrays['scores'][0, 1] = .2
            arrays['keypoints_simcc'][..., 2] += 2

    def edit_report(self, color, mutation):
        path = self.paths[color]
        report = json.loads(path.read_text())
        mutation(report)
        path.write_text(json.dumps(report))

    def edit_arrays(self, color, mutation, frames=None):
        path = self.paths[color]
        report = json.loads(path.read_text())
        for item in report['frames']:
            if frames is not None and item['frame'] not in frames:
                continue
            raw_path = path.parent / item['rawPath']
            with np.load(raw_path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            mutation(arrays)
            np.savez_compressed(raw_path, **arrays)
            item['rawSha256'] = digest(raw_path)
        path.write_text(json.dumps(report))

    def compare(self):
        return compare_reports(self.paths['bgr'], self.paths['rgb'], size_provider=lambda _: (96, 64))

    def test_recomputes_metrics_on_each_and_paired_common_joints_without_picking_winner(self):
        report = self.compare()
        self.assertEqual(report['status'], 'preprocessing_compared')
        self.assertFalse(report['releaseReady'])
        self.assertFalse(report['rawCoordinatesValidated'])
        self.assertIsNone(report['winner'])
        self.assertEqual(report['frameCountCompared'], 2)
        bgr, rgb = report['summary']['bgr'], report['summary']['rgb']
        self.assertEqual(bgr['eligibleExisting2DBodyJoints'], 34)
        self.assertEqual(bgr['commonBodyJoints'], 32)
        self.assertEqual(rgb['commonBodyJoints'], 32)
        self.assertEqual(bgr['pairedCommonBodyJoints'], 30)
        self.assertAlmostEqual(bgr['meanImage2DDistanceOnPairedJointsPx'], 0, places=5)
        self.assertAlmostEqual(rgb['meanImage2DDistanceOnPairedJointsPx'], 5, places=5)
        self.assertGreater(bgr['meanRawScore133'], 1)
        self.assertIn('raw_scores_are_not_depth_confidence', report['limitations'])
        self.assertEqual(report['simccZChangeRgbMinusBgr']['jointCount'], 266)
        self.assertAlmostEqual(report['simccZChangeRgbMinusBgr']['meanAbsoluteDelta'], 2)
        self.assertFalse(report['simccZChangeRgbMinusBgr']['isAccuracy'])
        self.assertEqual(bgr['inferenceTotalSec'], sum(frame['runs']['bgr']['inferenceSec'] for frame in report['frames']))

    def test_report_2d_summary_is_not_trusted_as_arithmetic_source(self):
        self.edit_report('rgb', lambda report: [frame['image2DComparison'].update(meanPixelDistance=9999)
                                               for frame in report['frames']])
        self.assertAlmostEqual(self.compare()['summary']['rgb']['meanImage2DDistancePx'], 5, places=5)

    def test_different_model_hash_runtime_or_dependencies_are_rejected(self):
        original = self.paths['rgb'].read_text()
        mutations = [lambda r: r['model'].update(weightsSha256='a' * 64),
                     lambda r: r['model'].update(sourceFileSha256='b' * 64),
                     lambda r: r['model'].update(executionProviders=['different']),
                     lambda r: r['dependencyVersions'].update(numpy='different')]
        for mutation in mutations:
            self.paths['rgb'].write_text(original)
            self.edit_report('rgb', mutation)
            with self.assertRaises(ProbeError):
                self.compare()

    def test_source_files_are_rehashed_and_not_just_metadata_compared(self):
        self.fixture.input.write_bytes(b'changed original proxy bytes')
        with self.assertRaises(ProbeError) as error:
            self.compare()
        self.assertEqual(error.exception.code, 'evidence_hash_mismatch')

    def test_timeaxis_hash_and_raw_npz_hash_are_required(self):
        original = self.paths['rgb'].read_text()
        self.edit_report('rgb', lambda r: r['input'].update(timestampsSha256='b' * 64))
        with self.assertRaises(ProbeError):
            self.compare()
        self.paths['rgb'].write_text(original)
        self.edit_report('rgb', lambda r: r['frames'][0].update(rawSha256='b' * 64))
        with self.assertRaises(ProbeError) as error:
            self.compare()
        self.assertEqual(error.exception.code, 'evidence_hash_mismatch')

    def test_npz_frame_time_and_bbox_are_verified_even_with_matching_file_hash(self):
        for key, value in [('frame_index', np.array(1)), ('timestamp_sec', np.array(99.)),
                           ('bbox_xyxy', np.array([0, 0, 96, 64]))]:
            with self.subTest(key=key):
                raw = self.directory / 'rgb/frame-000000.raw.npz'
                original_raw, original_report = raw.read_bytes(), self.paths['rgb'].read_text()
                self.edit_arrays('rgb', lambda arrays: arrays.update({key: value}), [0])
                with self.assertRaises(ProbeError):
                    self.compare()
                raw.write_bytes(original_raw)
                self.paths['rgb'].write_text(original_report)

    def test_same_valid_frame_but_different_crop_is_not_comparable(self):
        self.edit_arrays('rgb', lambda arrays: arrays['bbox_xyxy'].__setitem__(0, arrays['bbox_xyxy'][0] + 1))
        self.edit_report('rgb', lambda r: [f['bboxXYXY'].__setitem__(0, f['bboxXYXY'][0] + 1) for f in r['frames']])
        with self.assertRaises(ProbeError) as error:
            self.compare()
        self.assertEqual(error.exception.code, 'frame_crop_mismatch')

    def test_different_frame_set_duplicate_frame_and_non_bgr_rgb_pair_are_rejected(self):
        original = self.paths['rgb'].read_text()
        for mutation in (lambda r: r['frames'].pop(), lambda r: r['frames'].append(r['frames'][0]),
                         lambda r: r['preprocess'].update(passedToLibraryColor='bgr')):
            self.paths['rgb'].write_text(original)
            self.edit_report('rgb', mutation)
            with self.assertRaises(ProbeError):
                self.compare()

    def test_missing_common_points_are_explicit_not_zero_error(self):
        self.edit_arrays('rgb', lambda arrays: arrays['scores'].__setitem__((0, slice(0, 17)), 0))
        report = self.compare()
        self.assertEqual(report['summary']['rgb']['commonBodyJoints'], 0)
        self.assertEqual(report['summary']['rgb']['coverageOfEligible2DBodyJoints'], 0)
        self.assertIsNone(report['summary']['rgb']['meanImage2DDistancePx'])
        self.assertIsNone(report['summary']['bgr']['meanImage2DDistanceOnPairedJointsPx'])

    def test_cli_existing_target_preserves_all_source_evidence(self):
        report = json.loads(self.paths['bgr'].read_text())
        raw = self.paths['bgr'].parent / report['frames'][0]['rawPath']
        targets = [self.paths['bgr'], self.fixture.authorization, self.fixture.times, self.fixture.points,
                   self.fixture.input, self.fixture.model_path, raw]
        script = ROOT / 'scripts/pose3d_probe_compare.py'
        for target in targets:
            original = target.read_bytes()
            result = subprocess.run([sys.executable, str(script), str(self.paths['bgr']), str(self.paths['rgb']),
                                     '--output', str(target)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stderr)['errorCode'], 'output_already_exists')
            self.assertNotIn('Traceback', result.stderr)
            self.assertEqual(target.read_bytes(), original)

    def test_raw_path_traversal_is_rejected(self):
        self.edit_report('rgb', lambda r: r['frames'][0].update(rawPath='../frame-000000.raw.npz'))
        with self.assertRaises(ProbeError) as error:
            self.compare()
        self.assertEqual(error.exception.code, 'invalid_raw_path')


if __name__ == '__main__':
    unittest.main()
