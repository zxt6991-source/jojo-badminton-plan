"""Camera math/evidence regression; synthetic fixtures are not accuracy evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_decode_audit as audit_fixture
from pose3d_camera_research import (CONFIG_SCHEMA, DISPLAY_MATRIX, read_config, recover_camera,
                                     project, pose_metrics, run_research)
from pose3d_probe import ProbeError, validate_proxy_clock, digest


def scenario(**changes):
    return {'id': 'base', 'source': 'assumed-research', 'evidence': 'Explicit synthetic test camera',
            'imageWidth': 480, 'imageHeight': 854, 'fx': 1145., 'fy': 1145.,
            'cx': 240., 'cy': 427., 'rootDepth': 5.14388, **changes}


class CameraMathTests(unittest.TestCase):
    def pose(self):
        # A 3D pose with nonzero depths and a known right angle at the left elbow.
        p = np.zeros((133, 3), dtype=float)
        p[11], p[12] = [-.2, 0, 0], [.2, 0, 0]
        p[5], p[7], p[9] = [-.3, -.4, 0], [-.3, -.1, 0], [-.3, -.1, .4]
        p[6], p[8], p[10] = [.3, -.4, .1], [.4, -.1, .2], [.2, -.2, .4]
        return p

    def test_backproject_root_center_display_axes_and_roundtrip_are_consistent(self):
        camera = scenario()
        relative = self.pose()
        absolute = relative + [1, -.5, camera['rootDepth']]
        uv = project(absolute, camera)
        recovered, error = recover_camera(uv, absolute[:, 2] - camera['rootDepth'], np.ones(133, dtype=bool), camera)
        np.testing.assert_allclose(recovered['camera_xyz'], absolute, atol=1e-12)
        np.testing.assert_allclose(recovered['camera_root_relative_xyz'], relative, atol=1e-12)
        np.testing.assert_allclose(recovered['camera_root_relative_xyz'][[11, 12]].mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(recovered['display_root_relative_xyz'], relative @ DISPLAY_MATRIX.T, atol=1e-12)
        self.assertLess(recovered['display_root_relative_xyz'][9, 2], 0, 'farther joint must have negative display Z')
        self.assertLess(error, 1e-10)
        self.assertAlmostEqual(pose_metrics(relative, np.ones(133, bool))['elbows']['left']['angleDeg'], 90)

    def test_rotated_pose_keeps_nonplanar_z_and_preserves_angle_and_lengths(self):
        p = self.pose()
        theta = .83
        rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                             [-np.sin(theta), 0, np.cos(theta)]])
        rotated = p @ rotation.T
        camera = scenario(fx=900., fy=1100., cx=211., cy=387.)
        absolute = rotated + [.4, -.2, camera['rootDepth']]
        recovered, _ = recover_camera(project(absolute, camera), rotated[:, 2], np.ones(133, bool), camera)
        np.testing.assert_allclose(recovered['camera_root_relative_xyz'], rotated, atol=1e-12)
        first, second = pose_metrics(p, np.ones(133, bool)), pose_metrics(recovered['camera_root_relative_xyz'], np.ones(133, bool))
        self.assertAlmostEqual(first['elbows']['left']['angleDeg'], second['elbows']['left']['angleDeg'])
        for label, length in first['boneLengthsModelRelative'].items():
            self.assertAlmostEqual(length, second['boneLengthsModelRelative'][label])

    def test_root_centering_does_not_cancel_focal_or_root_depth_assumptions(self):
        p = self.pose()
        base = scenario()
        uv = project(p + [0, 0, base['rootDepth']], base)
        first, _ = recover_camera(uv, p[:, 2], np.ones(133, bool), base)
        second, _ = recover_camera(uv, p[:, 2], np.ones(133, bool), scenario(fx=700., fy=700., rootDepth=7.))
        self.assertGreater(np.max(np.abs(first['camera_root_relative_xyz'] - second['camera_root_relative_xyz'])), .1)
        for output in (first, second):
            np.testing.assert_allclose(output['camera_root_relative_xyz'][[11, 12]].mean(axis=0), 0, atol=1e-12)

    def test_missing_hips_and_nonpositive_depth_fail_instead_of_fabricating_root(self):
        uv, z, mask = np.ones((133, 2)), np.zeros(133), np.ones(133, bool)
        for hip in (11, 12):
            valid = mask.copy(); valid[hip] = False
            with self.assertRaises(ProbeError) as error:
                recover_camera(uv, z, valid, scenario())
            self.assertEqual(error.exception.code, 'root_hips_missing')
        z[9] = -5.14388
        with self.assertRaises(ProbeError) as error:
            recover_camera(uv, z, mask, scenario())
        self.assertEqual(error.exception.code, 'nonpositive_camera_depth')
        z[9] = -10
        with self.assertRaises(ProbeError):
            recover_camera(uv, z, mask, scenario())

    def test_missing_limbs_stay_nan_and_metrics_explain_missing_or_degenerate_segments(self):
        p, mask, camera = self.pose(), np.ones(133, bool), scenario()
        mask[9] = False
        recovered, _ = recover_camera(project(p + [0, 0, camera['rootDepth']], camera), p[:, 2], mask, camera)
        self.assertTrue(np.isnan(recovered['camera_root_relative_xyz'][9]).all())
        metrics = pose_metrics(recovered['camera_root_relative_xyz'], mask)
        self.assertEqual(metrics['elbows']['left']['reason'], 'joint_missing')
        self.assertIsNone(metrics['boneLengthsModelRelative']['leftForearm'])
        self.assertEqual(pose_metrics(np.zeros((133, 3)), np.ones(133, bool))['elbows']['left']['reason'], 'degenerate_segments')


class CameraResearchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = audit_fixture.DecodeAuditTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root, self.probe = self.fixture.root, self.fixture.probe
        self.config = {'schemaVersion': CONFIG_SCHEMA, 'scenarios': [scenario()]}
        self.config_path = self.root / 'camera.json'
        self.counter = 0
        # Different image/depth positions create sensitivity in actual XYZ angles.
        self.fixture.arrays['keypoints_2d'][0, [5, 7, 9]] = [[200, 300], [230, 350], [260, 310]]
        self.fixture.write_raw()
        self.decoded_clock = {'width': 480, 'height': 854, 'timestamps': [0, .1, .2]}

    def run_research(self, dimensions=(480, 854)):
        self.counter += 1
        self.config_path.write_text(json.dumps(self.config))
        report_path = self.probe / 'report.json'
        report_path.write_text(json.dumps(self.fixture.report))
        self.output = self.root / f'camera-output-{self.counter}'
        with patch('pose3d_probe.probe_proxy', return_value=self.decoded_clock):
            return run_research(report_path, self.config_path, self.output, image_size_reader=lambda _: dimensions)

    def rejects(self, code):
        result = self.run_research()
        self.assertEqual(result['status'], 'invalid_evidence', result)
        self.assertEqual(result['errorCode'], code, result)
        self.assertEqual(list(self.output.glob('*.npz')), [])
        self.assertEqual(list(self.output.glob('*.f32')), [])

    def test_multiple_scenarios_emit_only_research_xyz_with_parameter_sensitivity(self):
        self.config['scenarios'].append(scenario(id='focal-700', fx=700., fy=700.))
        result = self.run_research()
        self.assertEqual(result['status'], 'camera_research_recovered', result)
        for key in ('rawCoordinatesValidated', 'cameraCoordinatesValidated', 'releaseReady', 'productCompatible'):
            self.assertFalse(result[key])
        self.assertEqual(result['sourceEvidence']['probeReportPath'], str((self.probe / 'report.json').resolve()))
        self.assertEqual(result['sourceEvidence']['timestampsSha256'], self.fixture.report['input']['timestampsSha256'])
        self.assertEqual(result['clockEvidence']['independentProxyValidation']['source'], 'ffprobe-best_effort_timestamp_time')
        self.assertFalse(result['clockEvidence']['originalCaptureTimingVerified'])
        self.assertFalse(result['clockEvidence']['sourceEditingHistoryVerified'])
        self.assertEqual(result['unit'], 'model-relative')
        self.assertIn('z-away-from-camera', result['cameraAxes'])
        self.assertIn('z-toward-camera', result['displayAxes'])
        self.assertEqual(len(result['scenarios']), 2)
        self.assertEqual(len(list(self.output.glob('*.npz'))), 2)
        self.assertEqual(list(self.output.glob('*.f32')), [])
        frame = result['scenarios'][0]['frames'][0]
        with np.load(self.output / frame['researchPath'], allow_pickle=False) as data:
            self.assertEqual(data['camera_xyz'].shape, (133, 3))
            self.assertGreater(np.ptp(data['camera_xyz'][:, 2]), 0)
            np.testing.assert_allclose(data['camera_root_relative_xyz'][[11, 12]].mean(axis=0), 0, atol=1e-12)
            np.testing.assert_allclose(data['display_root_relative_xyz'], data['camera_root_relative_xyz'] @ DISPLAY_MATRIX.T, atol=1e-12)
        self.assertGreater(result['parameterSensitivity'][0]['elbowAngleDegrees']['left']['span'], .01)
        self.assertGreater(result['parameterSensitivity'][0]['boneLengthsModelRelative']['leftUpperArm']['span'], 0)
        self.assertLess(frame['projectionRoundtripMaxPixelError'], 1e-7)

    def test_forged_consistent_time_files_do_not_override_actual_proxy_pts(self):
        f = self.fixture
        np.array([0, .3, .6], dtype='<f4').tofile(f.times)
        f.report['input']['timestampsSha256'] = digest(f.times)
        f.report['frames'][0]['captureTimeSec'] = float(np.float32(.3))
        f.arrays['timestamp_sec'] = np.array(float(np.float32(.3)))
        f.write_raw()
        self.rejects('proxy_timestamp_mismatch')

    def test_recorded_clock_metadata_must_match_independent_validation(self):
        self.fixture.report['input']['proxyClockValidation'] = {'source': 'unverified-claim'}
        self.rejects('proxy_clock_metadata_mismatch')

    def test_existing_output_is_not_reused_or_overwritten(self):
        result = self.run_research()
        original = (self.output / 'camera-research.json').read_bytes()
        with self.assertRaises(ProbeError) as error:
            run_research(self.probe / 'report.json', self.config_path, self.output)
        self.assertEqual(error.exception.code, 'output_not_empty')
        self.assertEqual((self.output / 'camera-research.json').read_bytes(), original)

    def test_concurrent_output_reservation_is_exclusive(self):
        self.config_path.write_text(json.dumps(self.config))
        report_path = self.probe / 'report.json'
        report_path.write_text(json.dumps(self.fixture.report))
        output = self.root / 'racing-camera-output'
        original_mkdir = Path.mkdir

        def reserve_after_preflight(path, *args, **kwargs):
            result = original_mkdir(path, *args, **kwargs)
            if path.resolve() == output.resolve():
                (path / '.camera-research-reserved').write_text('another running task')
            return result

        with patch.object(Path, 'mkdir', reserve_after_preflight):
            with self.assertRaises(ProbeError) as error:
                run_research(report_path, self.config_path, output)
        self.assertEqual(error.exception.code, 'output_not_empty')
        self.assertEqual((output / '.camera-research-reserved').read_text(), 'another running task')
        self.assertFalse((output / 'camera-research.json').exists())

    def test_calibrated_label_never_enables_product_release(self):
        self.config['scenarios'][0]['source'] = 'calibrated'
        result = self.run_research()
        self.assertEqual(result['status'], 'camera_research_recovered', result)
        self.assertFalse(result['releaseReady'])
        self.assertFalse(result['productCompatible'])
        self.assertFalse(result['cameraCoordinatesValidated'])

    def test_no_implicit_parameters_and_no_missing_evidence(self):
        original = copy.deepcopy(self.config)
        for field in ('fx', 'fy', 'cx', 'cy', 'rootDepth', 'imageWidth', 'imageHeight'):
            self.config = copy.deepcopy(original)
            del self.config['scenarios'][0][field]
            self.rejects('camera_parameters_required')
        self.config = copy.deepcopy(original)
        self.config['scenarios'][0]['evidence'] = ''
        self.rejects('camera_evidence_required')
        self.config['scenarios'][0].update(evidence='source missing', source='default')
        self.rejects('camera_evidence_required')

    def test_invalid_numeric_parameters_rejected(self):
        original = copy.deepcopy(self.config)
        for field, value in [('fx', 0), ('fy', -1), ('fx', True), ('rootDepth', 0),
                             ('rootDepth', -5), ('cx', float('nan')), ('cy', float('inf'))]:
            self.config = copy.deepcopy(original)
            self.config['scenarios'][0][field] = value
            self.rejects('invalid_camera_parameters')
        self.config = copy.deepcopy(original)
        self.config['scenarios'][0]['imageWidth'] = 480.5
        self.rejects('invalid_image_dimensions')

    def test_camera_dimensions_and_duplicate_or_unsafe_scenario_ids_rejected(self):
        self.config['scenarios'][0]['imageWidth'] = 481
        self.rejects('camera_image_dimensions_mismatch')
        self.config['scenarios'][0] = scenario(id='../escape')
        self.rejects('invalid_scenario_id')
        self.config['scenarios'] = [scenario(), scenario()]
        self.rejects('invalid_scenario_id')

    def test_raw_hash_frame_time_and_authorization_checks_are_reused(self):
        original = copy.deepcopy(self.fixture.report)
        for mutate, code in [
            (lambda r: r['frames'][0].update(rawSha256='f' * 64), 'raw_hash_mismatch'),
            (lambda r: r['frames'][0].update(frame=20), 'invalid_frame_index'),
            (lambda r: r['frames'][0].update(captureTimeSec=.6), 'raw_timestamp_mismatch'),
            (lambda r: r.pop('authorization'), 'authorization_missing'),
            (lambda r: r.update(rawCoordinatesValidated=True), 'unsafe_coordinate_claim')]:
            self.fixture.report = copy.deepcopy(original)
            mutate(self.fixture.report)
            self.rejects(code)

    def test_negative_depth_in_any_scenario_rejects_all_outputs(self):
        self.config['scenarios'].append(scenario(id='impossible-depth', rootDepth=.1))
        self.rejects('nonpositive_camera_depth')

    def test_missing_hip_in_raw_rejects_recovery(self):
        f = self.fixture
        f.arrays['scores'][0, 11] = 0
        f.arrays['keypoints_simcc'][0, 11] = -.5
        f.arrays['keypoints'][0, 11] = [-.5, -.5, (-.5 / 192 - 1) * 2.1744869]
        f.write_raw()
        self.rejects('root_hips_missing')

    def test_nonfinite_recovery_from_extreme_finite_config_fails_explicitly(self):
        self.config['scenarios'][0]['fx'] = 5e-324
        self.rejects('nonfinite_camera_reconstruction')


if __name__ == '__main__':
    unittest.main()
