"""Synthetic decoder-contract checks, never model accuracy evidence."""
import copy
from datetime import date
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from pose3d_decode_audit import run_audit, ONNX_SHAPES, OFFICIAL_Z_RANGE
from pose3d_probe import authorize, digest, OUTPUT_NAMES, OUTPUT_SHAPES


class DecodeAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.probe = self.root / 'probe'
        self.probe.mkdir()
        self.proxy = self.root / 'proxy.mp4'
        self.proxy.write_bytes(b'synthetic-contract-test-no-video')
        self.times = self.root / 'timestamps.f32'
        np.array([0, .1, .2], dtype='<f4').tofile(self.times)
        self.authorization = self.root / 'authorization.json'
        self.authorization.write_text(json.dumps({
            'schemaVersion': 'pose3d-local-authorization-0.1.0',
            'scope': 'local-internal-3d-validation', 'externalUploadAllowed': False,
            'authorizedAt': date.today().isoformat(), 'retentionUntil': None,
            'userStatement': 'Synthetic test only', 'items': [{
                'videoId': 'test', 'originalPath': 'test-original.mp4', 'originalSha256': 'a' * 64,
                'proxyPath': str(self.proxy), 'proxySha256': digest(self.proxy)}]}))
        self.report = {
            'schemaVersion': 'pose3d-raw-probe-0.1.0', 'status': 'raw_evidence_collected',
            'evaluationMode': True, 'releaseReady': False, 'productCompatible': False,
            'rawCoordinatesValidated': False, 'coordinateStatus': 'unverified-raw-mixed-scales',
            'authorization': authorize(self.authorization, digest(self.proxy)),
            'input': {'kind': 'capture-proxy', 'path': str(self.proxy), 'sha256': digest(self.proxy),
                      'clockSource': 'existing-2d-capture-timestamps', 'frameCount': 3,
                      'timestampsPath': str(self.times), 'timestampsSha256': digest(self.times)},
            'model': {'apiVersion': '0.0.16', 'weightsSha256': 'b' * 64, 'sourceFileSha256': 'c' * 64,
                'implementation': 'rtmlib.tools.pose_estimation.rtmpose3d.RTMPose3d',
                'modelInputSize': [288, 384], 'zRange': OFFICIAL_Z_RANGE,
                'onnxOutputs': [{'name': name, 'shape': shape} for name, shape in zip('xyz', ONNX_SHAPES)],
                'upstreamDecode': {'simccSplitRatio': 2, 'zDenominator': 192, 'isotropicXYZVerified': False}},
            'frames': [{'frame': 1, 'captureTimeSec': float(np.float32(.1)), 'bboxXYXY': [0, 0, 50, 80],
                'rawPath': 'frame-000001.raw.npz',
                'outputShapes': {name: list(shape) for name, shape in zip(OUTPUT_NAMES, OUTPUT_SHAPES)}}]}
        q = np.full((1, 133, 3), [144, 192, 144], dtype=np.float32)
        q[0, 5, 2], q[0, 6, 2] = 100, 200
        mixed = q.copy()
        mixed[..., 2] = (q[..., 2] / 192 - 1) * OFFICIAL_Z_RANGE
        self.arrays = {'keypoints': mixed, 'scores': np.full((1, 133), .9, np.float32),
                       'keypoints_simcc': q, 'keypoints_2d': np.ones((1, 133, 2), np.float32),
                       'frame_index': np.array(1, dtype=np.int64), 'timestamp_sec': np.array(float(np.float32(.1))),
                       'bbox_xyxy': np.array([0, 0, 50, 80])}
        self.write_raw()
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def write_raw(self):
        path = self.probe / 'frame-000001.raw.npz'
        np.savez_compressed(path, **self.arrays)
        self.report['frames'][0]['rawSha256'] = digest(path)

    def run_audit(self):
        self.counter += 1
        path = self.probe / 'report.json'
        path.write_text(json.dumps(self.report))
        self.output = self.root / f'audit-{self.counter}'
        return run_audit(path, self.output)

    def rejects(self, code):
        result = self.run_audit()
        self.assertEqual(result['status'], 'invalid_evidence', result)
        self.assertEqual(result['errorCode'], code, result)
        self.assertEqual(list(self.output.glob('*.npz')), [])
        self.assertEqual(list(self.output.glob('*.f32')), [])

    def test_exact_independent_z_decode_and_per_point_difference_without_xyz(self):
        result = self.run_audit()
        self.assertEqual(result['status'], 'decode_audited', result)
        self.assertFalse(result['cameraXYZRecovered'])
        self.assertFalse(result['rawCoordinatesValidated'])
        self.assertFalse(result['productCompatible'])
        self.assertFalse(result['releaseReady'])
        self.assertEqual(result['decoder']['officialDenominator'], 144)
        self.assertEqual(result['decoder']['officialDepthInputSize'], 288)
        self.assertEqual(result['decoder']['upstreamDenominator'], 192)
        item = result['frames'][0]
        with np.load(self.output / item['researchPath'], allow_pickle=False) as data:
            self.assertNotIn('keypoints', data.files)
            self.assertAlmostEqual(float(data['upstream_relative_z'][0, 0]), -.543621725, places=6)
            self.assertEqual(float(data['official_codec_relative_z'][0, 0]), 0)
            self.assertAlmostEqual(float(data['relative_z_difference'][0, 0]), .543621725, places=6)
            np.testing.assert_allclose(data['official_codec_relative_z'],
                (self.arrays['keypoints_simcc'][..., 2].astype(float) / 144 - 1) * OFFICIAL_Z_RANGE)
        self.assertEqual(item['finiteRatios']['keypoints'], 1)
        self.assertEqual(item['officialCodecRelativeZ']['count'], 133)
        self.assertLess(item['pairDepthDifferences']['shoulders']['leftMinusRightRelativeZ'], 0)
        self.assertEqual(list(self.output.glob('*.f32')), [])
        self.assertEqual(digest(self.probe / 'frame-000001.raw.npz'), self.report['frames'][0]['rawSha256'])

    def test_wrong_schema_incomplete_status_or_validated_coordinate_claim_rejected(self):
        original = copy.deepcopy(self.report)
        for key, value, code in [('schemaVersion', '0.2.0', 'unsupported_probe_schema_or_status'),
                                ('status', 'running', 'unsupported_probe_schema_or_status'),
                                ('rawCoordinatesValidated', True, 'unsafe_coordinate_claim'),
                                ('releaseReady', True, 'unsafe_coordinate_claim'),
                                ('productCompatible', True, 'unsafe_coordinate_claim')]:
            self.report = copy.deepcopy(original)
            self.report[key] = value
            self.rejects(code)
        self.report = copy.deepcopy(original)
        del self.report['rawCoordinatesValidated']
        self.rejects('unsafe_coordinate_claim')

    def test_authorization_missing_hash_mismatch_and_forged_metadata_rejected(self):
        original = copy.deepcopy(self.report)
        cases = [(lambda r: r.pop('authorization'), 'authorization_missing'),
                 (lambda r: r['authorization'].pop('originalSha256'), 'authorization_hash_missing'),
                 (lambda r: r['authorization'].update(evidenceSha256='f' * 64), 'authorization_hash_mismatch'),
                 (lambda r: r['authorization'].update(proxySha256='f' * 64), 'authorization_proxy_mismatch'),
                 (lambda r: r['authorization'].update(originalSha256='f' * 64), 'authorization_metadata_mismatch')]
        for mutate, code in cases:
            self.report = copy.deepcopy(original)
            mutate(self.report)
            self.rejects(code)

    def test_modified_proxy_and_timestamps_rejected(self):
        self.proxy.write_bytes(b'changed')
        self.rejects('input_hash_mismatch')
        self.proxy.write_bytes(b'synthetic-contract-test-no-video')
        np.array([0, .11, .2], dtype='<f4').tofile(self.times)
        self.rejects('timestamps_hash_mismatch')

    def test_onnx_output_lengths_ratio_and_z_range_must_match_reviewed_codec(self):
        original = copy.deepcopy(self.report)
        for mutate, code in [
            (lambda m: m['onnxOutputs'][2].update(shape=[1, 133, 768]), 'incompatible_onnx_outputs'),
            (lambda m: m['upstreamDecode'].update(simccSplitRatio=1), 'unsupported_decoder'),
            (lambda m: m['upstreamDecode'].update(zDenominator=144), 'unsupported_decoder'),
            (lambda m: m.update(zRange=float('nan')), 'unsupported_z_range'),
            (lambda m: m.pop('sourceFileSha256'), 'model_hash_missing')]:
            self.report = copy.deepcopy(original)
            mutate(self.report['model'])
            self.rejects(code)

    def test_raw_hash_mismatch_and_path_traversal_rejected(self):
        self.report['frames'][0]['rawSha256'] = 'f' * 64
        self.rejects('raw_hash_mismatch')
        self.write_raw()
        self.report['frames'][0]['rawPath'] = '../frame-000001.raw.npz'
        self.rejects('invalid_raw_path')

    def test_raw_symlink_outside_probe_directory_rejected(self):
        path = self.probe / 'frame-000001.raw.npz'
        outside = self.root / 'outside.npz'
        path.rename(outside)
        path.symlink_to(outside)
        self.rejects('raw_path_escape_or_missing')

    def test_npz_nonfinite_values_shapes_and_pickle_arrays_rejected(self):
        original = copy.deepcopy(self.arrays)
        for name in OUTPUT_NAMES:
            self.arrays = copy.deepcopy(original)
            self.arrays[name].flat[0] = np.nan
            self.write_raw()
            self.rejects('nonfinite_raw_array')
        self.arrays = copy.deepcopy(original)
        self.arrays['scores'] = np.zeros((133,))
        self.write_raw()
        self.rejects('invalid_raw_array')
        self.arrays['scores'] = np.full((1, 133), object(), dtype=object)
        self.write_raw()
        self.rejects('invalid_raw_archive')

    def test_frame_out_of_bounds_duplicate_and_raw_index_mismatch_rejected(self):
        original = copy.deepcopy(self.report)
        self.report['frames'][0]['frame'] = 3
        self.rejects('invalid_frame_index')
        self.report = copy.deepcopy(original)
        self.report['frames'].append(copy.deepcopy(self.report['frames'][0]))
        self.rejects('duplicate_or_unsorted_frames')
        self.report = copy.deepcopy(original)
        self.arrays['frame_index'] = np.array(2)
        self.write_raw()
        self.rejects('raw_frame_mismatch')

    def test_time_and_bbox_mismatch_rejected_even_when_raw_hash_updated(self):
        self.arrays['timestamp_sec'] = np.array(.3)
        self.write_raw()
        self.rejects('raw_timestamp_mismatch')
        self.report['frames'][0]['captureTimeSec'] = .3
        self.rejects('capture_timestamp_mismatch')
        self.arrays['timestamp_sec'] = np.array(float(np.float32(.1)))
        self.report['frames'][0]['captureTimeSec'] = float(np.float32(.1))
        self.arrays['bbox_xyxy'][0] = 5
        self.write_raw()
        self.rejects('raw_bbox_mismatch')

    def test_simcc_out_of_bounds_or_wrong_bin_grid_rejected(self):
        for value in (288, -.5, 144.1):
            self.arrays['keypoints_simcc'][0, 0, 2] = value
            self.write_raw()
            self.rejects('invalid_simcc_locations')

    def test_upstream_must_match_claimed_decode_before_comparing_formulas(self):
        self.arrays['keypoints'][0, 0, 2] += .1
        self.write_raw()
        self.rejects('upstream_z_formula_mismatch')

    def test_nonpositive_scores_keep_raw_values_but_exclude_missing_points_from_statistics(self):
        self.arrays['scores'][0, 5] = 0
        self.arrays['keypoints_simcc'][0, 5] = -.5
        self.arrays['keypoints'][0, 5] = [-.5, -.5, (-.5 / 192 - 1) * OFFICIAL_Z_RANGE]
        self.write_raw()
        result = self.run_audit()
        self.assertEqual(result['status'], 'decode_audited', result)
        item = result['frames'][0]
        self.assertEqual(item['officialCodecRelativeZ']['count'], 132)
        self.assertIsNone(item['pairDepthDifferences']['shoulders']['leftMinusRightRelativeZ'])


if __name__ == '__main__':
    unittest.main()
