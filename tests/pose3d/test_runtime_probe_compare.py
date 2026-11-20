"""Synthetic evidence-contract tests; none are model accuracy evidence."""
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
import test_decode_audit as fixtures
import pose3d_runtime_probe_compare as comparison
from pose3d_probe import ProbeError, authorize, digest


class RuntimeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DecodeAuditTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.a = self.fixture.report
        self.clock = {'source': 'ffprobe-best_effort_timestamp_time', 'frameCount': 3,
                      'width': 100, 'height': 100, 'maxTimestampErrorSec': 0, 'toleranceSec': .0001}
        self.a['input']['proxyClockValidation'] = self.clock
        self.points = self.root / 'keypoints.f32'
        np.full((3, 65, 3), [.01, .01, .9], dtype='<f4').tofile(self.points)
        self.a['input'].update(keypoints2dPath=str(self.points), keypoints2dSha256=digest(self.points))
        self.a['bboxPolicy'] = {'origin': 'same-frame-existing-2d', 'confidenceThreshold': .4,
            'extentPaddingRatio': .15, 'detectorUsed': False, 'identityValidation': 'not_independently_validated'}
        self.a['preprocess'] = {'passedToLibraryColor': 'rgb', 'decodedImageColor': 'bgr',
                                'colorConversion': 'BGR-to-RGB before RTMPose3d'}
        self.a_model = self.root / 'model-a.onnx'
        self.a_model.write_bytes(b'synthetic-model-a')
        self.a['model'].update(path=str(self.a_model), weightsSha256=digest(self.a_model))
        self.b_model = self.root / 'model-b.pth'
        self.b_model.write_bytes(b'synthetic-different-model-b')
        self.source_root = self.root / 'source'
        project = self.source_root / 'projects/rtmpose3d'
        config = project / 'configs/rtmw3d-l_8xb64_cocktail14-384x288.py'
        config.parent.mkdir(parents=True)
        config.write_text('# synthetic config, never imported')
        source_hashes = {}
        for name in comparison.SOURCE_FILES:
            path = project / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('# synthetic source, never imported: ' + name)
            source_hashes[name] = digest(path)
        self.b = copy.deepcopy(self.a)
        self.b.update(schemaVersion='pose3d-mmpose-raw-probe-0.1.0',
                      coordinateStatus='official-decoder-and-assumed-camera-unvalidated')
        self.b['model'] = {'path': str(self.b_model), 'weightsSha256': digest(self.b_model),
            'runtime': 'mmpose-pytorch', 'modelId': 'rtmw3d-mmpose-l', 'sourceRef': 'v1.3.2',
            'dependencyVersions': {'mmpose': '1.3.2'}, 'sourceRoot': str(self.source_root),
            'projectSourceSha256': source_hashes, 'configPath': str(config), 'configSha256': digest(config),
            'effectiveConfigSha256': 'e' * 64,
            'configOverrides': {'model.backbone.init_cfg': None, 'model.test_cfg.mode': '3d'},
            'preprocess': {'inputColor': 'bgr', 'modelBgrToRgb': True, 'inputSize': [288, 384]},
            'decoderObservation': 'copy official head.decode outputs before camera postprocess; no numerical replacement',
            'cameraAssumptions': {'calibrated': False},
            'sourceGeometry': {'codecInputSize': [288, 384, 288], 'simccSplitRatio': 2,
                'decodeDepthDenominator': 144, 'zRange': comparison.OFFICIAL_Z_RANGE, 'useDark': False,
                'rootIndices': [11, 12], 'isotropicXYZValidated': False}}
        self.b_dir = self.root / 'mmpose'
        self.b_dir.mkdir()
        self.b_arrays = {'keypoints_simcc': self.fixture.arrays['keypoints_simcc'].copy(),
                        'scores': self.fixture.arrays['scores'].copy(),
                        'keypoints_2d': self.fixture.arrays['keypoints_2d'] + [3, 4],
                        'official_camera_keypoints': np.full((1, 133, 3), 12345.),
                        'frame_index': self.fixture.arrays['frame_index'].copy(),
                        'timestamp_sec': self.fixture.arrays['timestamp_sec'].copy(),
                        'bbox_xyxy': self.fixture.arrays['bbox_xyxy'].copy()}
        # A global depth shift must disappear after each model's hip centering.
        self.b_arrays['keypoints_simcc'][..., 2] += 10
        self.refresh_decoded_b()
        self.b['frames'][0]['rawPath'] = 'frame-000001.mmpose-raw.npz'
        self.b['frames'][0]['outputShapes'] = {name: list(shape) for name, shape in comparison.RAW_SHAPES.items()}
        self.a_path, self.b_path = self.fixture.probe / 'report.json', self.b_dir / 'report.json'
        self.write_b()

    def tearDown(self):
        self.fixture.tearDown()

    def refresh_decoded_b(self):
        q = self.b_arrays['keypoints_simcc']
        decoded = q.copy()
        decoded[..., 2] = (q[..., 2] / 144 - 1) * comparison.OFFICIAL_Z_RANGE
        self.b_arrays['decoded_keypoints'] = decoded

    def write_b(self):
        path = self.b_dir / 'frame-000001.mmpose-raw.npz'
        np.savez_compressed(path, **self.b_arrays)
        self.b['frames'][0]['rawSha256'] = digest(path)

    def write_reports(self):
        self.a_path.write_text(json.dumps(self.a))
        self.b_path.write_text(json.dumps(self.b))

    def compare(self, clock_validator=None):
        self.write_reports()
        return comparison.compare_reports(self.a_path, self.b_path,
            clock_validator or (lambda _path, _times: self.clock))

    def rejects(self, code):
        with self.assertRaises(ProbeError) as caught:
            self.compare()
        self.assertEqual(caught.exception.code, code)

    def test_compares_different_models_in_image_xy_and_correctly_centered_relative_z(self):
        result = self.compare()
        self.assertEqual(result['status'], 'raw_outputs_compared')
        self.assertFalse(result['releaseReady'])
        self.assertFalse(result['metricDepthMeasured'])
        self.assertFalse(result['accuracyMeasured'])
        self.assertFalse(result['cameraXYZCompared'])
        self.assertFalse(result['clockEvidence']['originalCaptureTimingVerified'])
        self.assertNotIn('winner', result)
        self.assertNotEqual(result['reports']['a']['model']['weightsSha256'],
                            result['reports']['b']['model']['weightsSha256'])
        self.assertEqual(result['summary']['imageXYDistancePx']['count'], 17)
        self.assertEqual(result['summary']['imageXYDistancePx']['meanAbsolute'], 5)
        self.assertAlmostEqual(result['summary']['rootCenteredRelativeZDifference']['maxAbsolute'], 0, places=12)
        self.assertEqual(result['summary']['leftRightSignAgreement']['nonTieCount'], 1)
        self.assertEqual(result['summary']['leftRightSignAgreement']['agreeCount'], 1)
        self.assertEqual(result['summary']['leftRightSignAgreement']['tieInEitherCount'], 5)
        self.assertAlmostEqual(result['summary']['existing2DDeviationPx']['a']['meanAbsolute'], 0, places=6)
        self.assertAlmostEqual(result['summary']['existing2DDeviationPx']['b']['meanAbsolute'], 5, places=6)

    def test_depth_joint_delta_uses_144_and_camera_xyz_is_never_compared(self):
        self.b_arrays['keypoints_simcc'][0, 0, 2] += 2
        self.refresh_decoded_b()
        self.write_b()
        result = self.compare()
        self.assertAlmostEqual(result['frames'][0]['joints'][0]['relativeZBMinusA'],
                               2 / 144 * comparison.OFFICIAL_Z_RANGE, places=12)
        summary = result['summary']
        self.b_arrays['official_camera_keypoints'] *= -1000
        self.write_b()
        self.assertEqual(self.compare()['summary'], summary)

    def test_each_raw_decoder_formula_is_checked_before_comparison(self):
        self.b_arrays['decoded_keypoints'][0, 0, 2] += .1
        self.write_b()
        self.rejects('b_z_formula_mismatch')
        self.refresh_decoded_b()
        self.write_b()
        self.fixture.arrays['keypoints'][0, 0, 2] += .1
        self.fixture.write_raw()
        self.rejects('upstream_z_formula_mismatch')

    def test_missing_hip_prevents_root_relative_metrics_without_fabricating_zero(self):
        self.b_arrays['scores'][0, 11] = 0
        self.b_arrays['keypoints_simcc'][0, 11] = -.5
        self.refresh_decoded_b()
        self.write_b()
        result = self.compare()
        self.assertEqual(result['summary']['imageXYDistancePx']['count'], 16)
        self.assertEqual(result['summary']['rootCenteredRelativeZDifference']['count'], 0)
        self.assertIsNone(result['summary']['rootCenteredRelativeZDifference']['meanAbsolute'])
        self.assertEqual(result['summary']['framesWithoutCommonHipRoot'], 1)
        self.assertTrue(all(row['relativeZBMinusA'] is None for row in result['frames'][0]['joints']))

    def test_pair_sign_disagreement_and_ties_are_not_counted_as_agreement(self):
        self.b_arrays['keypoints_simcc'][0, [5, 6], 2] = [210, 110]
        self.refresh_decoded_b()
        self.write_b()
        signs = self.compare()['summary']['leftRightSignAgreement']
        self.assertEqual(signs, {'availableCount': 6, 'nonTieCount': 1, 'tieInEitherCount': 5,
                                 'agreeCount': 0, 'disagreeCount': 1, 'agreementFraction': 0})

    def test_same_clock_different_authorized_proxy_cannot_be_compared(self):
        second = self.root / 'second.mp4'
        second.write_bytes(b'a different video with the same frame count and timeline')
        document = json.loads(self.fixture.authorization.read_text())
        item = copy.deepcopy(document['items'][0])
        item.update(videoId='second', proxyPath=str(second), proxySha256=digest(second), originalSha256='b' * 64)
        document['items'].append(item)
        self.fixture.authorization.write_text(json.dumps(document))
        self.a['authorization'] = authorize(self.fixture.authorization, digest(self.fixture.proxy))
        self.b['authorization'] = authorize(self.fixture.authorization, digest(second))
        self.b['input'].update(path=str(second), sha256=digest(second))
        self.rejects('input_evidence_mismatch')

    def test_independent_actual_proxy_clock_validation_is_mandatory(self):
        self.write_reports()
        with patch.object(comparison, 'validate_proxy_clock',
                          side_effect=ProbeError('proxy_timestamp_mismatch', 'synthetic clock mismatch')) as checked:
            with self.assertRaises(ProbeError) as caught:
                comparison.compare_reports(self.a_path, self.b_path)
        self.assertEqual(caught.exception.code, 'proxy_timestamp_mismatch')
        checked.assert_called_once()
        self.b['input']['proxyClockValidation'] = dict(self.clock, width=99)
        self.rejects('proxy_clock_metadata_mismatch')

    def test_bbox_and_capture_timestamp_mismatch_rejected_even_with_matching_raw_hash(self):
        self.b_arrays['bbox_xyxy'][0] += 1
        self.b['frames'][0]['bboxXYXY'][0] += 1
        self.write_b()
        self.rejects('frame_crop_mismatch')
        self.b_arrays['bbox_xyxy'][0] -= 1
        self.b['frames'][0]['bboxXYXY'][0] -= 1
        self.b_arrays['timestamp_sec'] = np.array(.15)
        self.b['frames'][0]['captureTimeSec'] = .15
        self.write_b()
        self.rejects('capture_timestamp_mismatch')

    def test_actual_raw_hash_shape_grid_and_nonfinite_checks(self):
        self.b['frames'][0]['rawSha256'] = 'f' * 64
        self.rejects('evidence_hash_mismatch')
        self.b_arrays['scores'] = np.ones((133,))
        self.write_b()
        self.rejects('invalid_raw_array')
        self.b_arrays['scores'] = np.ones((1, 133))
        self.b_arrays['keypoints_simcc'][0, 0, 2] = 144.1
        self.write_b()
        self.rejects('invalid_simcc_locations')
        self.b_arrays['keypoints_simcc'][0, 0, 2] = np.nan
        self.write_b()
        self.rejects('invalid_raw_array')

    def test_wrong_decoder_metadata_or_changed_source_model_config_rejected(self):
        self.b['model']['sourceGeometry']['decodeDepthDenominator'] = 192
        self.rejects('unsupported_b_decoder')
        self.b['model']['sourceGeometry']['decodeDepthDenominator'] = 144
        for path in (self.b_model, Path(self.b['model']['configPath']),
                     self.source_root / 'projects/rtmpose3d' / comparison.SOURCE_FILES[0]):
            original = path.read_bytes()
            path.write_bytes(b'changed evidence')
            self.rejects('evidence_hash_mismatch')
            path.write_bytes(original)

    def test_cli_preserves_existing_evidence_and_rejects_publish_race(self):
        self.write_reports()
        original = self.a_path.read_bytes()
        with patch.object(sys, 'argv', ['compare', str(self.a_path), str(self.b_path), '--output', str(self.a_path)]), \
             patch.object(comparison, 'compare_reports') as compare, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(comparison.main(), 2)
            compare.assert_not_called()
        self.assertEqual(self.a_path.read_bytes(), original)
        output = self.root / 'comparison.json'

        def raced(*_args):
            output.write_bytes(b'existing evidence from another writer')
            return {'status': 'raw_outputs_compared'}

        with patch.object(sys, 'argv', ['compare', str(self.a_path), str(self.b_path), '--output', str(output)]), \
             patch.object(comparison, 'compare_reports', side_effect=raced), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(comparison.main(), 2)
        self.assertEqual(output.read_bytes(), b'existing evidence from another writer')
        self.assertEqual(list(self.root.glob('.comparison.json.*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
