"""Playback must have observations on every decodable, detectable frame."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'worker'))
from main import infer, JobError
from algorithms import smooth_pose


class InferenceTests(unittest.TestCase):
    def setup_video(self, frame_count, missing=(), fail_at=None):
        frames = [np.full((80, 100, 3), i, dtype=np.uint8) for i in range(frame_count)]
        capture = Mock()
        capture.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        observed = []

        def model(frame):
            index = int(frame[0, 0, 0])
            observed.append(index)
            if index == fail_at:
                raise RuntimeError('inference failed')
            if index in missing:
                return np.empty((0, 133, 2)), np.empty((0, 133))
            # A slowly moving player without a wrist-speed peak. These frames
            # still need a complete overlay outside any detected swing window.
            points = np.column_stack((np.linspace(20, 60, 133) + index * .02,
                                      np.linspace(10, 70, 133)))
            return points[None], np.full((1, 133), .95)

        cv2 = SimpleNamespace(VideoCapture=Mock(return_value=capture),
                              COLOR_BGR2GRAY=1, cvtColor=lambda frame, _: frame[:, :, 0])
        return capture, cv2, model, observed

    def test_every_frame_has_pose_even_without_motion_peak(self):
        for fps in (30, 60, 120):
            with self.subTest(fps=fps):
                capture, cv2, model, observed = self.setup_video(24)
                times = np.arange(24) / fps
                with patch.dict(sys.modules, {'cv2': cv2}), patch('main.load_model', return_value=model):
                    pose, multiple, brightness, sampled = infer('video.mp4', times,
                        {'captureFps': fps, 'width': 100, 'height': 80}, None)
                self.assertEqual(observed, list(range(24)))
                self.assertEqual(sampled, list(range(24)))
                self.assertFalse(multiple)
                self.assertAlmostEqual(brightness, 11.5)
                self.assertTrue(np.isfinite(smooth_pose(pose, times)).all())
                # cwb-65 nose retains this frame's measured position.
                np.testing.assert_allclose(pose[:, 0, 0], (.02 * np.arange(24) + 20) / 100)
                cv2.VideoCapture.assert_called_once()
                capture.release.assert_called_once()

    def test_actual_detection_gaps_remain_missing_and_tracking_resumes(self):
        capture, cv2, model, observed = self.setup_video(18, missing=(5, 6, 7))
        with patch.dict(sys.modules, {'cv2': cv2}), patch('main.load_model', return_value=model):
            pose, _, _, _ = infer('video.mp4', np.arange(18) / 30,
                {'captureFps': 30, 'width': 100, 'height': 80}, None)
        self.assertEqual(observed, list(range(18)))
        self.assertTrue(np.isnan(pose[5:8, :, :2]).all())
        self.assertTrue((pose[5:8, :, 2] == 0).all())
        self.assertTrue(np.isfinite(pose[8:]).all())
        capture.release.assert_called_once()

    def test_frame_count_mismatch_releases_decoder(self):
        for decoded_count in (9, 11):
            with self.subTest(decoded_count=decoded_count):
                capture, cv2, model, _ = self.setup_video(decoded_count)
                with patch.dict(sys.modules, {'cv2': cv2}), patch('main.load_model', return_value=model):
                    with self.assertRaises(JobError) as error:
                        infer('video.mp4', np.arange(10) / 30,
                            {'captureFps': 30, 'width': 100, 'height': 80}, None)
                self.assertEqual(error.exception.code, 'decode_frame_mismatch')
                capture.release.assert_called_once()

    def test_model_failure_releases_decoder(self):
        capture, cv2, model, _ = self.setup_video(10, fail_at=2)
        with patch.dict(sys.modules, {'cv2': cv2}), patch('main.load_model', return_value=model):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                infer('video.mp4', np.arange(10) / 30,
                    {'captureFps': 30, 'width': 100, 'height': 80}, None)
        capture.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
