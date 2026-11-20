"""Synthetic contract tests only. These do not validate model accuracy."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'worker'))
from pose3d.backend import Candidate, Pose3DError, create_provider
from pose3d.mapping import CWB_INDICES, missing_pose, to_cwb65
from pose3d.coordinates import normalize_root, validate_transform
from pose3d.windows import select_windows, normalize_windows, margin_frames, validate_timestamps
from pose3d.viewgate import classify_gate, gate_windows
from pose3d.association import TargetAssociation
from pose3d.temporal import smooth_window
from pose3d.quality import clean_window, summarize
from pose3d.io import validate_pose, write_pose, read_pose, write_json
from pose3d_main import run

CONFIG = json.loads((ROOT / 'worker/pose3d/config.json').read_text())
TRANSFORM = {'sourceConvention': 'test-only x-right-y-down-z-back',
             'targetConvention': 'x-right-y-up-z-forward',
             'matrix': [[1, 0, 0], [0, -1, 0], [0, 0, -1]], 'evidence': 'synthetic contract test'}


def windows(start=0, end=4, gate='pass'):
    return [{'startFrame': start, 'endFrame': end, 'origin': 'manual-annotation',
             'viewClass': 'side' if gate == 'pass' else 'unknown', 'viewGate': gate,
             'usableFrameRatio': 0}]


def populated(count=5):
    pose = np.zeros((count, 65, 4), dtype='<f4')
    pose[:, :, 0] = np.arange(65) * .01
    pose[:, :, 1] = np.arange(65) * .02
    pose[:, :, 2] = np.arange(65) * .03
    pose[:, :, 3] = .9
    pose[:, :, :3] -= pose[:, [11, 12], :3].mean(axis=1)[:, None, :]
    return pose


class WindowTests(unittest.TestCase):
    def config(self):
        config = copy.deepcopy(CONFIG)
        config['windows'].update(preSec=.2, postSec=.3)
        return config

    def test_windows_use_capture_seconds_across_frame_rates(self):
        for fps in (30, 60, 120):
            times = np.arange(3 * fps) / fps
            selected = select_windows(times, 'swing-windows', self.config(), [{'impact': {'frame': fps}}])
            self.assertAlmostEqual(times[selected[0]['startFrame']], .8)
            self.assertAlmostEqual(times[selected[0]['endFrame']], 1.3)

    def test_irregular_timestamps_use_search_not_fps(self):
        times = [0, .11, .23, .4, .55, .7, .9]
        selected = select_windows(times, 'swing-windows', self.config(), [{'impact': {'frame': 3, 'tSec': 100}}])
        self.assertEqual((selected[0]['startFrame'], selected[0]['endFrame']), (2, 5))

    def test_merges_sorted_overlapping_and_adjacent_ranges(self):
        selected = normalize_windows([{'startFrame': 7, 'endFrame': 8}, {'startFrame': 2, 'endFrame': 4},
                                      {'startFrame': 4, 'endFrame': 6}], 10, 'manual-annotation')
        self.assertEqual(selected, [{'startFrame': 2, 'endFrame': 8, 'origin': 'manual-annotation'}])

    def test_no_events_and_unconfigured_window_policy_are_distinct(self):
        with self.assertRaises(Pose3DError) as error:
            select_windows([0, .1], 'swing-windows', CONFIG)
        self.assertEqual(error.exception.code, 'no_swing_windows')
        with self.assertRaises(Pose3DError) as error:
            select_windows([0, .1], 'swing-windows', CONFIG, [{'impact': {'frame': 1}}])
        self.assertEqual(error.exception.code, 'window_policy_unconfigured')

    def test_scope_limits_manual_bounds_and_boolean_frames(self):
        for item in ({'startFrame': -1, 'endFrame': 1}, {'startFrame': 0, 'endFrame': 2},
                     {'startFrame': 1, 'endFrame': 0}, {'startFrame': False, 'endFrame': 1}):
            with self.assertRaises(Pose3DError):
                select_windows([0, .1], 'manual', CONFIG, manual=[item])
        config = self.config()
        config['limits']['maxFullFrames'] = 1
        with self.assertRaises(Pose3DError) as error:
            select_windows([0, .1], 'full', config)
        self.assertEqual(error.exception.code, 'scope_limit_exceeded')

    def test_margin_is_separate_from_output_window(self):
        window = windows(4, 6)[0]
        self.assertEqual(margin_frames(window, np.arange(11) / 10, .2), (2, 8))
        self.assertEqual((window['startFrame'], window['endFrame']), (4, 6))

    def test_rejects_invalid_capture_clock(self):
        for times in ([], [0, 0], [0, np.nan], [0, .2, .1], [-.1, 0]):
            with self.assertRaises(ValueError):
                validate_timestamps(times)


class ViewTests(unittest.TestCase):
    def test_uncertainty_blocks_and_back_maps_to_rear(self):
        policy = CONFIG['viewGate']
        for view, confidence in [('side', .1), ('unknown', 1), ('other', 1), ('side', np.nan)]:
            self.assertEqual(classify_gate(view, confidence, policy), ('unknown', 'blocked'))
        self.assertEqual(classify_gate('back', .9, policy), ('rear', 'limited'))
        self.assertEqual(classify_gate('front', .9, policy), ('front', 'limited'))
        self.assertEqual(classify_gate('oblique', .9, policy), ('oblique', 'pass'))

    def test_gate_preserves_ranges_and_does_not_mutate(self):
        original = windows()
        output = gate_windows(original, 'unknown', 1, CONFIG['viewGate'])
        self.assertEqual(output[0]['viewGate'], 'blocked')
        self.assertEqual(original[0]['viewGate'], 'pass')


class CoordinateTests(unittest.TestCase):
    def test_mapping_same_indices_as_2d_and_preserves_z(self):
        points = np.column_stack([np.arange(133)] * 3)
        output = to_cwb65(points, np.ones(133))
        np.testing.assert_array_equal(output[:, 2], CWB_INDICES)
        self.assertEqual(output[23, 2], 91)
        self.assertEqual(output[44, 2], 112)

    def test_invalid_joint_scores_and_xyz_become_exact_missing_pattern(self):
        points, scores = np.ones((133, 3)), np.ones(133)
        points[0, 2], scores[1], scores[2], scores[3] = np.inf, -1, 1.01, np.nan
        output = to_cwb65(points, scores)
        self.assertTrue(np.isnan(output[:4, :3]).all())
        self.assertTrue((output[:4, 3] == 0).all())
        with self.assertRaises(ValueError):
            to_cwb65(points[:65], scores[:65])

    def test_axis_conversion_and_hip_origin_without_losing_depth(self):
        pose = populated(1)[0]
        pose[:, :3] += [5, 8, 10]
        output = normalize_root(pose, TRANSFORM)
        np.testing.assert_allclose(output[[11, 12], :3].mean(axis=0), 0, atol=1e-6)
        self.assertGreater(output[9, 2], 0)
        self.assertLess(output[15, 2], 0)
        self.assertAlmostEqual(output[9, 3], .9, places=6)

    def test_missing_hip_does_not_invent_root(self):
        pose = populated(1)[0]
        pose[11] = [np.nan, np.nan, np.nan, 0]
        self.assertTrue(np.isnan(normalize_root(pose, TRANSFORM)[:, :3]).all())

    def test_transform_requires_evidence_and_rejects_anisotropic_coordinates(self):
        for transform in ({}, {**TRANSFORM, 'evidence': ''}, {**TRANSFORM, 'matrix': np.eye(2).tolist()},
                          {**TRANSFORM, 'matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 2]]}):
            with self.assertRaises(ValueError):
                validate_transform(transform)


class AssociationTests(unittest.TestCase):
    def candidate(self, offset=0):
        xy = np.full((133, 2), .4 + offset)
        return Candidate(xy, np.ones((133, 3)), np.ones(133))

    def test_matches_same_frame_target_instead_of_largest_person(self):
        tracker = TargetAssociation(CONFIG['association'])
        target = np.full((65, 3), .4)
        target[:, 2] = .9
        index, reason = tracker.choose([self.candidate(.4), self.candidate()], target)
        self.assertEqual(index, 1)
        self.assertIsNone(reason)

    def test_crossing_players_are_missing(self):
        tracker = TargetAssociation(CONFIG['association'])
        target = np.full((65, 3), .4)
        target[:, 2] = .9
        index, reason = tracker.choose([self.candidate(.005), self.candidate(-.005)], target)
        self.assertIsNone(index)
        self.assertEqual(reason, 'identity_ambiguous')

    def test_missing_target_and_large_continuity_jump_rejected(self):
        tracker = TargetAssociation(CONFIG['association'])
        target = np.full((65, 3), .4)
        target[:, 2] = .9
        tracker.choose([self.candidate()], target)
        target[:, :2] += .3
        self.assertIsNone(tracker.choose([self.candidate(.3)], target)[0])
        target[:, 2] = 0
        self.assertIsNone(tracker.choose([self.candidate()], target)[0])


class TemporalQualityTests(unittest.TestCase):
    def test_default_does_not_suppress_wrist_peak(self):
        pose = populated()
        pose[2, 9, 2] = 3
        result = smooth_window(pose, np.arange(5) / 100, CONFIG['temporal'])
        np.testing.assert_array_equal(result, pose)

    def test_occlusion_and_timestamp_gaps_reset_smoothing_without_carryover(self):
        pose = populated(6)
        pose[:2, 9, :3] = .2
        pose[2:4, 9] = [np.nan, np.nan, np.nan, 0]
        pose[4:, 9, :3] = .8
        policy = {**CONFIG['temporal'], 'enabled': True}
        result = smooth_window(pose, np.arange(6) / 100, policy)
        np.testing.assert_allclose(result[:2, 9, :3], .2)
        np.testing.assert_allclose(result[4:, 9, :3], .8)
        self.assertTrue(np.isnan(result[2:4, 9, :3]).all())
        pose[2:4, 9, :3], pose[2:4, 9, 3] = .8, .9
        result = smooth_window(pose, [0, .01, 1, 1.01, 1.02, 1.03], policy)
        np.testing.assert_allclose(result[2:, 9, :3], .8)

    def test_windows_are_independent(self):
        policy = {**CONFIG['temporal'], 'enabled': True}
        left, right = populated(), populated()
        left[:, 9, :3], right[:, 9, :3] = .1, .9
        np.testing.assert_allclose(smooth_window(left, np.arange(5) / 100, policy)[:, 9, :3], .1)
        np.testing.assert_allclose(smooth_window(right, np.arange(5) / 100 + 1, policy)[:, 9, :3], .9)

    def test_bone_drift_hides_abnormal_joint_and_does_not_fill_it(self):
        pose = populated()
        pose[2, 9, :3] = 20
        output, warnings = clean_window(pose, CONFIG['quality'])
        self.assertIn('bone_length_drift', warnings)
        self.assertTrue(np.isnan(output[2, 9, :3]).all())

    def test_quality_coverage_and_window_ratio_have_different_denominators(self):
        pose = missing_pose(10)
        pose[2:7] = populated(5)
        quality, selected = summarize(pose, windows(2, 6), [2, 3, 4, 5, 6])
        self.assertEqual(quality['coverageRatio'], .5)
        self.assertEqual(quality['usableFrameRatio'], 1)
        self.assertEqual(quality['identityLockedRatio'], 1)
        self.assertEqual(selected[0]['usableFrameRatio'], 1)


class BinaryTests(unittest.TestCase):
    def test_roundtrip_little_endian_and_exact_length(self):
        pose = populated()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / write_pose(directory, pose, windows())
            self.assertEqual(path.stat().st_size, 5 * 65 * 4 * 4)
            np.testing.assert_array_equal(read_pose(path, 5, windows()), pose)
            path.write_bytes(path.read_bytes() + b'\x00')
            with self.assertRaises(ValueError):
                read_pose(path, 5, windows())

    def test_rejects_partial_missing_infinity_bad_score_and_noncentered_root(self):
        for point in ([np.nan, 0, np.nan, 0], [1, 2, np.inf, .9], [1, 2, 3, 0], [1, 2, 3, 1.01]):
            pose = populated()
            pose[0, 0] = point
            with self.assertRaises(ValueError):
                validate_pose(pose, 5, windows())
        pose = populated()
        pose[:, :, 2] += 1
        with self.assertRaises(ValueError):
            validate_pose(pose, 5, windows())

    def test_outside_and_blocked_windows_must_be_missing(self):
        pose = populated()
        for selected in (windows(1, 4), windows(gate='blocked')):
            with self.assertRaises(ValueError):
                validate_pose(pose, 5, selected)
        pose[0] = missing_pose()
        validate_pose(pose, 5, windows(1, 4))

    def test_whitelisted_names_and_no_nan_json(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_pose(directory, populated(), windows(), '../pose3d.f32')
            with self.assertRaises(ValueError):
                write_json(Path(directory) / 'metadata.json', {'value': np.nan})


class IntegrationGateTests(unittest.TestCase):
    def request(self, directory):
        directory = Path(directory)
        times_path, points_path, proxy = directory / 'timestamps.f32', directory / 'keypoints.f32', directory / 'proxy.mp4'
        np.arange(5, dtype='<f4').tofile(times_path)
        points = np.full((5, 65, 3), .5, dtype='<f4')
        points.tofile(points_path)
        proxy.write_bytes(b'test only; never decoded by an unverified provider')
        return {'analysis': {'status': 'succeeded', 'skeleton': {'frameCount': 5}}, 'video': {'frameCount': 5},
                'config': copy.deepcopy(CONFIG), 'timestampsPath': str(times_path), 'keypointsPath': str(points_path),
                'inputPath': str(proxy), 'pose3d': {'status': 'queued', 'frameCount': 5, 'windows': windows(),
                'modelId': 'rtmw3d-rtmlib-x', 'viewGatePolicyVersion': CONFIG['viewGatePolicyVersion']}}

    def test_both_providers_cannot_produce_synthetic_results(self):
        for model_id in ('rtmw3d-rtmlib-x', 'rtmw3d-mmpose-l'):
            provider = create_provider(model_id, CONFIG)
            with self.assertRaises(Pose3DError) as error:
                provider.load()
            self.assertEqual(error.exception.code, 'model_unavailable')

    def test_cli_fails_independently_without_overwriting_2d_or_writing_3d(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.request(directory)
            original = Path(request['keypointsPath']).read_bytes()
            path = Path(directory) / 'request.json'
            write_json(path, request)
            result = subprocess.run([sys.executable, str(ROOT / 'worker/pose3d_main.py'), '--request', str(path),
                                     '--output-dir', str(Path(directory) / 'output')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            response = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(response['pose3d']['errorCode'], 'model_unavailable')
            self.assertEqual(response['pose3d']['status'], 'failed')
            self.assertNotIn('keypointsKey', response['pose3d'])
            self.assertEqual(Path(request['keypointsPath']).read_bytes(), original)
            self.assertFalse((Path(directory) / 'output/pose3d.f32').exists())

    def test_no_windows_view_blocked_corrupt_binary_and_mismatch_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            for mutation, code in [(lambda r: r['pose3d'].update(windows=[]), 'no_swing_windows'),
                                   (lambda r: r['pose3d'].update(windows=windows(gate='blocked')), 'view_blocked'),
                                   (lambda r: r['video'].update(frameCount=6), 'frame_mismatch'),
                                   (lambda r: Path(r['keypointsPath']).write_bytes(b'broken'), 'invalid_binary')]:
                request = self.request(directory)
                mutation(request)
                self.assertEqual(run(request, directory)['pose3d']['errorCode'], code)

    def test_legacy_zero_confidence_finite_2d_coordinates_remain_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.request(directory)
            points = np.fromfile(request['keypointsPath'], dtype='<f4').reshape(5, 65, 3)
            points[:, 0, 2] = 0
            points.tofile(request['keypointsPath'])
            self.assertEqual(run(request, directory)['pose3d']['errorCode'], 'model_unavailable')


if __name__ == '__main__':
    unittest.main()
