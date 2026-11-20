"""Synthetic contract tests only. No model accuracy or speed claim."""
from array import array
from datetime import date
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from pose3d_evaluate import evaluate, ordinal_metrics, drift, floats, load_clip, load_run, digest, write_report
from pose3d_compare import compare


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.path = self.base / 'manifest.json'
        self.path.write_text(json.dumps({'schemaVersion': 'pose3d-evaluation-0.1.0', 'frozen': False, 'videos': []}))

    def test_evaluation_clis_never_overwrite_manifest_or_existing_evidence(self):
        scripts = Path(__file__).resolve().parents[2] / 'scripts'
        for script in ('pose3d_evaluate.py', 'pose3d_compare.py'):
            for name in ('manifest.json', 'authorization.json', 'timestamps.f32', 'raw.npz', 'report.json'):
                target = self.base / name
                if target != self.path:
                    target.write_bytes(b'preserve-existing-evidence')
                original = target.read_bytes()
                result = subprocess.run([sys.executable, str(scripts / script), str(self.path), '--output', str(target)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stderr)['errorCode'], 'output_already_exists')
                self.assertEqual(target.read_bytes(), original)
                self.assertNotIn('Traceback', result.stderr)

    def test_new_output_publishes_blocked_report_and_rejects_repeat(self):
        scripts = Path(__file__).resolve().parents[2] / 'scripts'
        for script in ('pose3d_evaluate.py', 'pose3d_compare.py'):
            target = self.base / (script + '.json')
            result = subprocess.run([sys.executable, str(scripts / script), str(self.path), '--output', str(target)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)  # Empty real dataset stays blocked.
            self.assertEqual(json.loads(target.read_text())['status'], 'blocked')
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                write_report(target, {'replacement': True})
            self.assertEqual(target.read_bytes(), original)

    def test_report_writer_refuses_symlinks_including_dangling_ones(self):
        for name, target in (('existing', self.path), ('dangling', self.base / 'missing-target.json')):
            link = self.base / (name + '.json')
            link.symlink_to(target)
            with self.assertRaises(FileExistsError):
                write_report(link, {'replacement': True})
            self.assertTrue(link.is_symlink())
        self.assertFalse((self.base / 'missing-target.json').exists())

    def test_report_publish_race_preserves_competing_file_and_cleans_temporary(self):
        target = self.base / 'raced.json'
        real_link = os.link

        def competing_writer(source, destination):
            Path(destination).write_bytes(b'created-by-another-writer')
            return real_link(source, destination)

        with patch('pose3d_evaluate.os.link', side_effect=competing_writer):
            with self.assertRaises(FileExistsError):
                write_report(target, {'replacement': True})
        self.assertEqual(target.read_bytes(), b'created-by-another-writer')
        self.assertEqual(list(self.base.glob('.*.tmp')), [])

    def test_empty_manifest_never_passes(self):
        for report in (evaluate(self.path), compare(self.path)):
            self.assertEqual(report['status'], 'blocked')
            self.assertFalse(report['releaseReady'])
            self.assertEqual(report['gateDecision'], 'not_assessed')

    def test_synthetic_video_rejected_before_any_io(self):
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            load_clip(self.base, {'id': 'fixture', 'synthetic': True}, date(2026, 11, 16))

    def test_missing_consent_is_not_model_evidence(self):
        self.path.write_text(json.dumps({'schemaVersion': 'pose3d-evaluation-0.1.0', 'videos': [
            {'id': 'unconsented', 'viewClass': 'side', 'split': 'validation'}]}))
        report = evaluate(self.path)
        self.assertEqual(report['status'], 'blocked')
        self.assertEqual(report['clips'][0]['status'], 'invalid_evidence')

    def test_binary_length_and_little_endian(self):
        import struct
        path = self.base / 'values.f32'
        path.write_bytes(struct.pack('<2f', .25, .5))
        self.assertEqual(list(floats(path, 2)), [.25, .5])
        with self.assertRaisesRegex(ValueError, 'length'):
            floats(path, 3)

    def test_depth_sign_and_coverage_do_not_hide_missing(self):
        data = array('f', [float('nan'), float('nan'), float('nan'), 0] * 65 * 2)
        data[5 * 4:5 * 4 + 4] = array('f', [0, 0, -1, 1])
        data[9 * 4:9 * 4 + 4] = array('f', [0, 0, -2, 1])
        clip = {'frameCount': 2, 'viewClass': 'side', 'ordinalDepth': [
            {'frame': 0, 'a': 9, 'b': 5, 'closer': 'a'}, {'frame': 1, 'a': 9, 'b': 5, 'closer': 'a'}]}
        result = ordinal_metrics(clip, data, -1)
        self.assertEqual(result['side']['accuracy']['value'], 1)
        self.assertEqual(result['side']['coverage']['value'], .5)
        self.assertIsNone(result['front']['accuracy']['value'])
        self.assertEqual(ordinal_metrics(clip, data, 1)['side']['accuracy']['value'], 0)

    def test_duplicate_annotations_rejected(self):
        label = {'frame': 0, 'a': 9, 'b': 5, 'closer': 'a'}
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            ordinal_metrics({'frameCount': 1, 'viewClass': 'side', 'ordinalDepth': [label, label]},
                            array('f', [float('nan'), float('nan'), float('nan'), 0] * 65), -1)

    def test_drift_never_crosses_windows(self):
        data = array('f', [0, 0, 0, 1] * 65 * 2)
        data[7 * 4] = 1
        data[(65 + 7) * 4] = 100
        result = drift(data, 2, [{'startFrame': 0, 'endFrame': 0}, {'startFrame': 1, 'endFrame': 1}])
        self.assertEqual(result['count'], 0)
        self.assertIsNone(result['meanRelativeChange'])

    def make_run_fixture(self, runtime='rtmlib-onnx'):
        """Format fixture: payload bytes are synthetic, never a POC manifest."""
        import struct
        count = 2
        values = [float('nan'), float('nan'), float('nan'), 0] * 65 * count
        for joint, xyz in {5: (-1, 1, 0), 9: (-1, 2, -1), 11: (-1, 0, 0), 12: (1, 0, 0)}.items():
            values[joint * 4:joint * 4 + 4] = [*xyz, 1]
        (self.base / 'coordinates.md').write_text('Synthetic contract evidence; not a model result.')
        (self.base / 'times.f32').write_bytes(struct.pack('<2f', 0, .5))
        (self.base / 'pose.f32').write_bytes(struct.pack('<' + 'f' * len(values), *values))
        pose = {'status': 'succeeded', 'source': 'ai-monocular-predicted', 'frameCount': count,
                'pointSpec': 'cwb-65', 'coordinateSpace': 'camera-root-relative',
                'axisConvention': 'x-right-y-up-z-forward', 'unit': 'model-relative', 'rootJoint': 'mid-hip',
                'weightsSha256': '0' * 64, 'runtime': runtime, 'modelId': 'synthetic-test',
                'modelVersion': 'test', 'pipelineVersion': 'test', 'viewGatePolicyVersion': 'test',
                'windows': [{'startFrame': 0, 'endFrame': 0, 'viewClass': 'side', 'viewGate': 'pass'}]}
        run = {'pose3d': pose, 'timestampsPath': 'times.f32', 'keypointsPath': 'pose.f32',
               'provenance': {'sourceRevision': 'synthetic', 'dependencyVersions': {'test': '1'},
                              'device': 'test', 'configHash': 'test', 'coordinateEvidence': 'coordinates.md',
                              'depthNearerSign': -1, 'inputVideoSha256': 'a' * 64}}
        (self.base / 'run.json').write_text(json.dumps(run))
        return run, array('f', [0, .5]), values

    def test_normalized_artifact_and_timestamp_mismatch(self):
        run, times, values = self.make_run_fixture()
        loaded, data = load_run(self.base, {'path': 'run.json'}, 2, times, 'a' * 64)
        self.assertEqual(loaded['pose3d']['frameCount'], 2)
        self.assertEqual(data[9 * 4 + 2], -1)
        with self.assertRaisesRegex(ValueError, 'timestamp_mismatch'):
            load_run(self.base, {'path': 'run.json'}, 2, array('f', [0, .25]), 'a' * 64)

    def test_points_outside_windows_rejected(self):
        import struct
        run, times, values = self.make_run_fixture()
        values[65 * 4:65 * 4 + 4] = [0, 0, 0, 1]
        (self.base / 'pose.f32').write_bytes(struct.pack('<' + 'f' * len(values), *values))
        with self.assertRaisesRegex(ValueError, 'outside_windows'):
            load_run(self.base, {'path': 'run.json'}, 2, times, 'a' * 64)

    def test_missing_pattern_and_root_rejected(self):
        import struct
        run, times, values = self.make_run_fixture()
        values[3] = 1  # Invalid coordinates with positive confidence.
        (self.base / 'pose.f32').write_bytes(struct.pack('<' + 'f' * len(values), *values))
        with self.assertRaisesRegex(ValueError, 'invalid_point'):
            load_run(self.base, {'path': 'run.json'}, 2, times, 'a' * 64)
        run, times, values = self.make_run_fixture()
        values[11 * 4] = 0
        (self.base / 'pose.f32').write_bytes(struct.pack('<' + 'f' * len(values), *values))
        with self.assertRaisesRegex(ValueError, 'nonzero_root'):
            load_run(self.base, {'path': 'run.json'}, 2, times, 'a' * 64)

    def make_bound_manifest_fixture(self):
        # Byte-only test inputs. These are never used as model accuracy evidence.
        first_video, other_video = self.base / 'video-a.bin', self.base / 'video-b.bin'
        first_video.write_bytes(b'synthetic-video-A')
        other_video.write_bytes(b'synthetic-video-B')
        (self.base / 'authorization.md').write_text('Synthetic fixture authorization; not real consent.')
        for name, runtime in (('run-a.json', 'rtmlib-onnx'), ('run-b.json', 'mmpose-pytorch')):
            run, _, _ = self.make_run_fixture(runtime)
            run['provenance']['inputVideoSha256'] = digest(first_video)
            (self.base / name).write_text(json.dumps(run))
        manifest = {'schemaVersion': 'pose3d-evaluation-0.1.0', 'frozen': False, 'videos': [{
            'id': 'synthetic-format-fixture', 'videoPath': first_video.name, 'videoSha256': digest(first_video),
            'authorizationEvidence': 'authorization.md', 'retentionUntil': '2999-01-01',
            'frameCount': 2, 'timestampsPath': 'times.f32', 'viewClass': 'side', 'split': 'calibration',
            'runs': [{'path': 'run-a.json'}, {'path': 'run-b.json'}]}]}
        self.path.write_text(json.dumps(manifest))
        return first_video, other_video

    def test_same_frame_count_and_clock_cannot_bind_results_from_different_video(self):
        first, other = self.make_bound_manifest_fixture()
        # Both references are initially bound to the manifest's actual video hash.
        self.assertEqual(evaluate(self.path)['evaluatedRuns'], 2)
        self.assertEqual(compare(self.path)['clips'][0]['status'], 'compared')
        second_path = self.base / 'run-b.json'
        second = json.loads(second_path.read_text())
        second['provenance']['inputVideoSha256'] = digest(other)
        second_path.write_text(json.dumps(second))
        # Frame count, timestamps and pose bytes remain identical; only source
        # provenance differs. The second run must never enter A/B comparison.
        evaluated = evaluate(self.path)
        self.assertEqual(evaluated['evaluatedRuns'], 1)
        self.assertEqual(evaluated['clips'][0]['runs'][1]['status'], 'invalid_evidence')
        self.assertEqual(evaluated['clips'][0]['runs'][1]['error'], 'run_video_hash_mismatch')
        compared = compare(self.path)
        self.assertEqual(compared['status'], 'blocked')
        self.assertEqual(compared['clips'][0]['error'], 'run_video_hash_mismatch')
        self.assertFalse(compared['releaseReady'])

    def test_missing_or_malformed_input_video_hash_is_never_inferred_from_clock(self):
        for value in (None, '', 'invalid-hash'):
            self.make_bound_manifest_fixture()
            run_path = self.base / 'run-b.json'
            run = json.loads(run_path.read_text())
            if value is None:
                del run['provenance']['inputVideoSha256']
            else:
                run['provenance']['inputVideoSha256'] = value
            run_path.write_text(json.dumps(run))
            self.assertEqual(evaluate(self.path)['clips'][0]['runs'][1]['error'], 'invalid_input_video_hash')
            result = compare(self.path)
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(result['clips'][0]['error'], 'invalid_input_video_hash')

    def test_coordinate_evidence_is_required(self):
        pose = {'status': 'succeeded', 'source': 'ai-monocular-predicted', 'frameCount': 1,
                'pointSpec': 'cwb-65', 'coordinateSpace': 'camera-root-relative',
                'axisConvention': 'x-right-y-up-z-forward', 'unit': 'model-relative', 'rootJoint': 'mid-hip',
                'weightsSha256': '0' * 64, 'runtime': 'rtmlib-onnx', 'modelId': 'synthetic-test',
                'modelVersion': 'test', 'pipelineVersion': 'test', 'viewGatePolicyVersion': 'test'}
        (self.base / 'run.json').write_text(json.dumps({'pose3d': pose, 'provenance': {}}))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            load_run(self.base, {'path': 'run.json'}, 1, array('f', [0]), 'a' * 64)


if __name__ == '__main__':
    unittest.main()
