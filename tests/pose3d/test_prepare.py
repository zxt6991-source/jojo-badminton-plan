"""Synthetic input-contract fixtures, not model or human-annotation evidence."""
from pathlib import Path
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from pose3d_prepare import prepare, prepare_analysis
from pose3d_evaluate import digest


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / 'data'
        for name in ('analyses', 'videos/video-one', 'results/analysis-one'):
            (self.root / name).mkdir(parents=True)
        self.original = self.root / 'videos/video-one/original.mp4'
        self.proxy = self.root / 'results/analysis-one/proxy.mp4'
        self.original.write_bytes(b'synthetic-format-test-original')
        self.proxy.write_bytes(b'synthetic-format-test-proxy')
        self.timestamps = self.root / 'results/analysis-one/timestamps.f32'
        self.keypoints = self.root / 'results/analysis-one/keypoints.f32'
        self.timestamps.write_bytes(struct.pack('<3f', 0, .5, 1))
        points = []
        for frame in range(3):
            for joint in range(65):
                points.extend([.2 + .01 * joint, .1 + .01 * joint, .9])
        self.keypoints.write_bytes(struct.pack('<' + 'f' * len(points), *points))
        self.record = {
            'id': 'analysis-one', 'status': 'succeeded', 'createdAt': '2026-11-16T00:00:00Z',
            'analysis': {'id': 'analysis-one', 'videoId': 'video-one', 'status': 'succeeded', 'schemaVersion': '0.1.0',
                         'skeleton': {'spec': 'cwb-65', 'coordSpace': 'image-normalized', 'source': 'mono-2d',
                                      'frameCount': 3, 'timestampsKey': '/data/results/analysis-one/timestamps.f32',
                                      'keypointsKey': '/data/results/analysis-one/keypoints.f32'},
                         'quality': {'view': 'oblique', 'viewConfidence': .55},
                         'swings': [{'impact': {'frame': 1, 'tSec': .5, 'confidence': .8, 'evidence': ['wrist-peak']}}]},
            'video': {'id': 'video-one', 'frameCount': 3, 'width': 200, 'height': 100, 'captureFps': 2,
                      'proxyKey': '/data/results/analysis-one/proxy.mp4',
                      'storageKey': '/data/videos/video-one/original.mp4'}}
        self.record_path = self.root / 'analyses/analysis-one.json'
        self.save_record()
        (self.root / 'videos/video-one.json').write_text(json.dumps({'id': 'video-one', 'inputPath': str(self.original)}))
        self.auth = Path(self.tmp.name) / 'authorization.json'
        self.authorization = {'schemaVersion': 'pose3d-local-authorization-0.1.0',
            'scope': 'local-internal-3d-validation', 'externalUploadAllowed': False,
            'userStatement': 'synthetic authorization contract test', 'authorizedAt': '2026-11-16', 'retentionUntil': None,
            'items': [{'videoId': 'video-one', 'originalPath': str(self.original), 'originalSha256': digest(self.original),
                       'proxyPath': str(self.proxy), 'proxySha256': digest(self.proxy)}]}
        self.save_authorization()

    def save_record(self):
        self.record_path.write_text(json.dumps(self.record))

    def save_authorization(self):
        self.auth.write_text(json.dumps(self.authorization))

    def probe(self, path):
        self.assertEqual(path, self.proxy)
        return {'width': 200, 'height': 100, 'timestamps': [0, .5, 1]}

    def test_prepares_same_frame_bboxes_with_no_human_labels(self):
        report = prepare(self.root, self.auth, self.probe)
        self.assertEqual(report['status'], 'draft')
        self.assertEqual(report['scope'], 'local-research')
        self.assertFalse(report['releaseReady'])
        self.assertFalse(report['externalTransferAllowed'])
        self.assertFalse(report['inferenceExecuted'])
        item = report['inputs'][0]
        self.assertEqual(item['frameCount'], 3)
        self.assertEqual(item['usableBBoxFrames'], 3)
        self.assertEqual(item['modelInputPath'], str(self.proxy))
        self.assertEqual(item['originalVideoSha256'], digest(self.original))
        self.assertEqual(item['candidateSwingFrames'][0]['frame'], 1)
        self.assertEqual(item['candidateSwingFrames'][0]['tSec'], .5)
        self.assertAlmostEqual(item['candidateSwingFrames'][0]['bboxPixels'][0], 40, places=4)
        self.assertFalse(item['predictedView']['humanVerified'])
        self.assertEqual(item['currentViewGate']['gate'], 'blocked')
        self.assertFalse(item['currentViewGate']['userDisplayAllowed'])
        self.assertNotIn('viewClass', item)
        self.assertNotIn('retentionUntil', item)
        self.assertNotIn('ordinalDepth', item)

    def test_requires_explicit_authorization(self):
        with self.assertRaisesRegex(ValueError, 'authorization_evidence_required'):
            prepare(self.root, None, self.probe)
        self.authorization['externalUploadAllowed'] = True
        self.save_authorization()
        with self.assertRaisesRegex(ValueError, 'authorization_scope_mismatch'):
            prepare(self.root, self.auth, self.probe)

    def test_expired_authorization_is_rejected_before_any_video_probe(self):
        self.authorization['retentionUntil'] = '2000-01-01'
        self.save_authorization()
        with self.assertRaisesRegex(ValueError, 'authorization_expired'):
            prepare(self.root, self.auth, lambda _: self.fail('expired media must not be decoded'))

    def test_authorization_date_must_be_recorded(self):
        del self.authorization['authorizedAt']
        self.save_authorization()
        with self.assertRaisesRegex(ValueError, 'invalid_authorization_date'):
            prepare(self.root, self.auth, self.probe)

    def test_missing_authorization_cli_creates_no_output(self):
        output = Path(self.tmp.name) / 'report.json'
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / 'scripts/pose3d_prepare.py'),
                               '--data-dir', str(self.root), '--output', str(output)], capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(output.exists())

    def test_wrong_original_hash_rejected(self):
        self.original.write_bytes(b'changed-media')
        report = prepare(self.root, self.auth, self.probe)
        self.assertFalse(report['inputs'])
        self.assertIn('authorization_hash_mismatch', report['excluded'][0]['detail'])

    def test_wrong_proxy_hash_rejected(self):
        self.proxy.write_bytes(b'changed-proxy')
        report = prepare(self.root, self.auth, self.probe)
        self.assertFalse(report['inputs'])
        self.assertIn('authorization_hash_mismatch', report['excluded'][0]['detail'])

    def test_rejected_2d_does_not_create_candidates(self):
        self.record['status'] = self.record['analysis']['status'] = 'rejected_quality'
        self.save_record()
        report = prepare(self.root, self.auth, lambda _: self.fail('must not probe rejected input'))
        self.assertFalse(report['inputs'])
        self.assertEqual(report['excluded'][0]['reason'], 'analysis_not_succeeded')

    def test_proxy_frame_count_and_timestamps_must_match(self):
        for times, message in (([0, .5], 'proxy_frame_count_mismatch'), ([0, .6, 1], 'proxy_timestamp_mismatch')):
            report = prepare(self.root, self.auth, lambda _: {'width': 200, 'height': 100, 'timestamps': times})
            self.assertFalse(report['inputs'])
            self.assertIn(message, report['excluded'][0]['detail'])

    def test_legacy_finite_zero_confidence_points_are_valid_but_not_bbox_observations(self):
        content = bytearray(self.keypoints.read_bytes())
        # Legacy 2D schema allows finite XY at confidence 0. Keep this outlier
        # accepted as storage, but exclude it from the same-frame body bbox.
        struct.pack_into('<3f', content, (1 * 65 + 0) * 3 * 4, 0, 0, 0)
        self.keypoints.write_bytes(content)
        report = prepare(self.root, self.auth, self.probe)
        self.assertEqual(len(report['inputs']), 1, report)
        frame = report['inputs'][0]['candidateSwingFrames'][0]
        self.assertEqual(frame['visibleBodyJoints'], 16)
        self.assertTrue(frame['usable'])
        self.assertAlmostEqual(frame['bboxNormalized'][0], .21, places=6)
        self.assertAlmostEqual(frame['bboxNormalized'][1], .11, places=6)

    def test_missing_keypoint_pattern_rejected(self):
        content = bytearray(self.keypoints.read_bytes())
        struct.pack_into('<f', content, 0, math.nan)
        self.keypoints.write_bytes(content)
        report = prepare(self.root, self.auth, self.probe)
        self.assertFalse(report['inputs'])
        self.assertIn('invalid_keypoints_encoding', report['excluded'][0]['detail'])

    def test_symlink_artifact_is_rejected(self):
        alternate = self.keypoints.with_name('alternate.f32')
        self.keypoints.rename(alternate)
        self.keypoints.symlink_to(alternate)
        report = prepare(self.root, self.auth, self.probe)
        self.assertFalse(report['inputs'])
        self.assertIn('input_symlink_not_allowed', report['excluded'][0]['detail'])

    def test_swing_clock_mismatch_rejected(self):
        self.record['analysis']['swings'][0]['impact']['tSec'] = .25
        self.save_record()
        report = prepare(self.root, self.auth, self.probe)
        self.assertFalse(report['inputs'])
        self.assertIn('swing_timestamp_mismatch', report['excluded'][0]['detail'])

    def test_latest_invalid_analysis_does_not_fallback_to_old(self):
        older = json.loads(json.dumps(self.record))
        older['id'] = older['analysis']['id'] = 'analysis-old'
        older['createdAt'] = '2026-11-15T00:00:00Z'
        (self.root / 'analyses/analysis-old.json').write_text(json.dumps(older))
        self.keypoints.write_bytes(b'bad')
        report = prepare(self.root, self.auth, self.probe)
        self.assertFalse(report['inputs'])
        self.assertEqual({e['reason'] for e in report['excluded']}, {'invalid_research_input', 'older_successful_analysis'})

    def test_markdown_requires_structured_bound_consent(self):
        markdown = self.auth.with_suffix('.md')
        markdown.write_text('some sentence without hashes')
        with self.assertRaisesRegex(ValueError, 'markdown_requires'):
            prepare(self.root, markdown, self.probe)
        markdown.write_text('Test fixture\n```json\n' + json.dumps(self.authorization) + '\n```')
        self.assertEqual(len(prepare(self.root, markdown, self.probe)['inputs']), 1)


if __name__ == '__main__':
    unittest.main()
