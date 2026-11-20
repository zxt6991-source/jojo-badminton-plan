"""Evidence, offline output and HTML safety checks; no synthetic model claims."""
import copy
import json
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_camera_research as research_fixture
from pose3d_research_preview import collect_evidence, create_preview, atomic_create
from pose3d_probe import ProbeError, digest


class ResearchPreviewTests(unittest.TestCase):
    def setUp(self):
        self.research = research_fixture.CameraResearchTests()
        self.research.setUp()
        self.addCleanup(self.research.doCleanups)
        self.root, self.probe = self.research.root, self.research.probe
        self.research.config['scenarios'].append(research_fixture.scenario(id='focal-700', fx=700., fy=700.))
        item = self.research.fixture.report['frames'][0]
        item['overlayPath'] = 'frame-000001.overlay.png'
        import cv2
        cv2.imwrite(str(self.probe / item['overlayPath']), np.zeros((854, 960, 3), np.uint8))
        self.rebuild()
        self.output = self.root / 'preview.html'

    def rebuild(self):
        report = self.research.run_research()
        self.assertEqual(report['status'], 'camera_research_recovered', report)
        self.camera_path = self.research.output / 'camera-research.json'
        self.probe_path = self.probe / 'report.json'

    def create(self, output=None, config=None):
        with patch('pose3d_probe.probe_proxy', return_value=self.research.decoded_clock):
            return create_preview(self.camera_path, self.probe_path, output or self.output, config)

    def update_camera(self, mutate):
        report = json.loads(self.camera_path.read_text())
        mutate(report)
        self.camera_path.write_text(json.dumps(report))

    def rejects(self, code):
        with self.assertRaises(ProbeError) as error:
            self.create()
        self.assertEqual(error.exception.code, code)
        self.assertFalse(self.output.exists())

    def payload(self):
        html = self.output.read_text()
        match = re.search(r'<script id="preview-data" type="application/json">(.*?)</script>', html, re.S)
        return html, json.loads(match[1])

    def test_offline_preview_embeds_images_true_xyz_and_one_scale_for_all_scenarios(self):
        result = self.create()
        self.assertEqual(result['status'], 'research_preview_created')
        self.assertEqual(result['scenarios'], 2)
        html, payload = self.payload()
        self.assertTrue(payload['frames'][0]['image'].startswith('data:image/png;base64,'))
        self.assertFalse(payload['frames'][0]['upstreamHashRecorded'])
        self.assertFalse(payload['productCompatible'])
        self.assertFalse(payload['originalCaptureTimingVerified'])
        self.assertIn("connect-src 'none'", html)
        self.assertNotIn('fetch(', html)
        self.assertNotIn('innerHTML', html)
        self.assertIsNone(re.search(r'(?:src|href)=[\"\']https?://', html))
        self.assertNotIn('setInterval', html)
        all_points = np.array([p for s in payload['scenarios'] for f in s['poses'] for p in f['points'] if p is not None])
        self.assertEqual(all_points.shape[1], 3)
        self.assertGreater(np.ptp(all_points[:, 2]), 0)
        self.assertAlmostEqual(payload['sharedRadius'], np.linalg.norm(all_points, axis=1).max())
        report = json.loads(self.camera_path.read_text())
        npz = self.camera_path.parent / report['scenarios'][0]['frames'][0]['researchPath']
        with np.load(npz, allow_pickle=False) as arrays:
            np.testing.assert_allclose(payload['scenarios'][0]['poses'][0]['points'], arrays['display_root_relative_xyz'][:17])
        self.assertEqual(result['sha256'], digest(self.output))

    def test_untrusted_evidence_cannot_close_data_script_or_inject_html(self):
        unsafe = '</script><img src=x onerror=alert(1)><script>\u2028&'
        self.research.config['scenarios'][0]['evidence'] = unsafe
        self.rebuild()
        self.create()
        html, payload = self.payload()
        self.assertNotIn(unsafe, html)
        self.assertNotIn('<img src=x', html)
        self.assertEqual(payload['scenarios'][0]['camera']['evidence'], unsafe)
        self.assertIn('\\u003c/script\\u003e', html)
        self.assertIn('textContent=c.evidence', html)

    def test_changed_raw_or_research_hash_rejected(self):
        raw_path = self.probe / 'frame-000001.raw.npz'
        raw_path.write_bytes(raw_path.read_bytes() + b'changed')
        self.rejects('raw_hash_mismatch')

    def test_research_values_recomputed_even_if_attacker_updates_artifact_hash(self):
        report = json.loads(self.camera_path.read_text())
        frame = report['scenarios'][0]['frames'][0]
        path = self.camera_path.parent / frame['researchPath']
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        arrays['display_root_relative_xyz'][9, 2] += .5
        np.savez_compressed(path, **arrays)
        frame['researchSha256'] = digest(path)
        self.camera_path.write_text(json.dumps(report))
        self.rejects('research_array_mismatch')

    def test_scenario_frame_and_path_escape_rejected(self):
        original = self.camera_path.read_text()
        self.update_camera(lambda r: r['scenarios'][0]['frames'][0].update(frame=2))
        self.rejects('scenario_frame_mismatch')
        self.camera_path.write_text(original)
        self.update_camera(lambda r: r['scenarios'][0]['frames'][0].update(researchPath='../escape.npz'))
        self.rejects('invalid_research_path')

    def test_authorization_and_real_proxy_clock_rechecked(self):
        auth = self.research.fixture.authorization
        auth.write_text(auth.read_text() + '\n')
        self.rejects('authorization_hash_mismatch')

    def test_same_length_shifted_proxy_clock_rejected(self):
        self.research.decoded_clock['timestamps'] = [0, .2, .4]
        self.rejects('proxy_timestamp_mismatch')

    def test_config_source_hash_and_probe_report_chain_rechecked(self):
        original = self.camera_path.read_text()
        self.update_camera(lambda r: r.update(probeReportSha256='f' * 64))
        self.rejects('probe_report_hash_mismatch')
        self.camera_path.write_text(original)
        self.research.config_path.write_text(self.research.config_path.read_text() + '\n')
        self.rejects('camera_config_hash_mismatch')

    def test_legacy_report_requires_explicit_matching_config_and_still_checks_pts(self):
        self.update_camera(lambda r: (r.pop('sourceEvidence'), r.pop('clockEvidence')))
        self.rejects('camera_config_required_for_legacy')
        result = self.create(config=self.research.config_path)
        self.assertEqual(result['status'], 'research_preview_created')
        _, payload = self.payload()
        self.assertEqual(payload['clock']['source'], 'ffprobe-best_effort_timestamp_time')

    def test_wrong_overlay_size_and_png_contents_rejected(self):
        path = self.probe / 'frame-000001.overlay.png'
        import cv2
        cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))
        self.rejects('overlay_dimensions_mismatch')
        path.write_bytes(b'not-a-png' * 20)
        self.rejects('invalid_overlay_png')

    def test_existing_and_symlink_output_never_overwritten(self):
        self.output.write_text('keep this')
        with self.assertRaises(ProbeError) as error:
            self.create()
        self.assertEqual(error.exception.code, 'output_exists')
        self.assertEqual(self.output.read_text(), 'keep this')
        self.output.unlink()
        self.output.symlink_to(self.root / 'missing-target')
        with self.assertRaises(ProbeError) as error:
            self.create()
        self.assertEqual(error.exception.code, 'output_exists')
        self.assertTrue(self.output.is_symlink())
        self.assertFalse((self.root / 'missing-target').exists())

    def test_atomic_publication_preserves_concurrent_writers_and_removes_temporary(self):
        original_link = os.link

        def competing_writer(source, destination):
            Path(destination).write_text('other writer')
            original_link(source, destination)

        with patch('pose3d_research_preview.os.link', side_effect=competing_writer):
            with self.assertRaises(ProbeError) as error:
                atomic_create(self.output, b'our preview')
        self.assertEqual(error.exception.code, 'output_exists')
        self.assertEqual(self.output.read_text(), 'other writer')
        self.assertEqual(list(self.root.glob('.pose3d-preview-*')), [])


if __name__ == '__main__':
    unittest.main()
