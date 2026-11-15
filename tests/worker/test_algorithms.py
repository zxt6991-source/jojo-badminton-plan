import json
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'worker'))
sys.path.insert(0, str(ROOT/'scripts'))
from algorithms import capture_timestamps, to_cwb65, smooth_pose, wrist_speed, detect_swings, audio_onsets
from main import Tracker, JobError, ingest
from evaluate import match_events, evaluate


class TimingTests(unittest.TestCase):
    def test_irregular_real_clock_preserved(self):
        times, fps = capture_timestamps([3., 3.01, 3.022, 3.031])
        np.testing.assert_allclose(times, [0., .01, .022, .031])

    def test_capture_clock_does_not_mutate_playback_origin(self):
        playback = np.array([3., 3.01, 3.02])
        capture_timestamps(playback)
        self.assertEqual(playback[0], 3.)

    def test_explicit_slow_motion(self):
        times, fps = capture_timestamps(np.arange(20)/30, 120)
        self.assertEqual(fps, 120)
        self.assertAlmostEqual(times[-1], 19/120)

    def test_reject_nonmonotonic_and_variable_remapping(self):
        for values, fps in [([0., .1, .1], None), ([0., .1, .2, .4], 120)]:
            with self.assertRaises(ValueError):
                capture_timestamps(values, fps)

    def test_derivative_uses_timestamps(self):
        times = np.array([0., .01, .04, .10])
        pose = np.zeros((4, 65, 3))
        pose[:, 10, 0] = times*2
        pose[:, 10, 2] = 1
        np.testing.assert_allclose(wrist_speed(pose, times, 10)[1:3], [2., 2.])


class PoseTests(unittest.TestCase):
    def test_mapping_excludes_face(self):
        points = np.column_stack([np.arange(133), np.arange(133)])
        result = to_cwb65(points, np.ones(133), 200, 200)
        self.assertEqual(result.shape, (65, 3))
        self.assertAlmostEqual(result[23, 0], 91/200)
        self.assertAlmostEqual(result[-1, 0], 132/200)

    def test_short_gaps_only(self):
        pose = np.ones((20, 65, 3))*.8
        pose[3, 10] = [np.nan, np.nan, 0]
        pose[7:16, 10] = [np.nan, np.nan, 0]
        result = smooth_pose(pose, np.arange(20)/120)
        self.assertTrue(np.isfinite(result[3, 10, 0]))
        self.assertLess(result[3, 10, 2], .8)
        self.assertTrue(np.isnan(result[10, 10, 0]))
        self.assertEqual(result[10, 10, 2], 0)

    def test_low_confidence_outlier_is_treated_as_missing_before_filtering(self):
        times = np.arange(20)/120
        pose = np.ones((20, 65, 3))*.8
        pose[:, 10, 0] = np.linspace(.2, .6, 20)
        pose[8, 10] = [.99, .01, .3]
        before = pose.copy()
        missing = pose.copy()
        missing[8, 10] = [np.nan, np.nan, 0]
        # The rejected coordinates must not contaminate either filter direction;
        # a short gap may still be interpolated from its reliable endpoints.
        np.testing.assert_allclose(smooth_pose(pose, times), smooth_pose(missing, times), equal_nan=True)
        np.testing.assert_array_equal(pose, before)

    def test_occlusion_resets_both_filters_and_recovers_at_current_position(self):
        times = np.arange(20)/120
        pose = np.ones((20, 65, 3))*.9
        pose[:5, 10, :2] = .2
        pose[5:15, 10] = [.5, .5, .3]
        pose[15:, 10, :2] = .8
        result = smooth_pose(pose, times)
        np.testing.assert_allclose(result[:5, 10, :2], .2)
        np.testing.assert_allclose(result[15:, 10, :2], .8)
        self.assertTrue(np.isnan(result[5:15, 10, :2]).all())
        self.assertTrue((result[5:15, 10, 2] == 0).all())

    def test_timestamp_gap_does_not_blend_disconnected_observations(self):
        times = np.array([0., .01, .02, .5, .51, .52])
        pose = np.ones((len(times), 65, 3))*.9
        pose[:3, :, :2] = .2
        pose[3:, :, :2] = .8
        result = smooth_pose(pose, times)
        np.testing.assert_allclose(result[:, :, :2], pose[:, :, :2])

    def test_gap_fill_uses_elapsed_time_and_reduces_confidence(self):
        times = np.array([0., .01, .04, .05])
        pose = np.ones((4, 65, 3))*.8
        pose[:, 10, 0] = [.2, np.nan, .8, .9]
        pose[1, 10] = [np.nan, np.nan, 0]
        # The missing point is a quarter of the way between its endpoints.
        expected = pose.copy()
        expected[1, 10] = [.35, .8, .8]
        result = smooth_pose(pose, times)
        np.testing.assert_allclose(result[:, 10, :2], smooth_pose(expected, times)[:, 10, :2])
        self.assertAlmostEqual(result[1, 10, 2], .64)
        # One missing frame at 30fps spans >50ms, so it stays missing.
        sparse = smooth_pose(pose, np.arange(4)/30)
        self.assertTrue(np.isnan(sparse[1, 10, :2]).all())

    def test_invalid_confidence_and_timestamps_do_not_enter_filter(self):
        pose = np.ones((4, 65, 3))*.8
        pose[:, 10, 2] = [np.nan, np.inf, 1.1, -.2]
        result = smooth_pose(pose, np.arange(4)/30)
        self.assertTrue(np.isnan(result[:, 10, :2]).all())
        self.assertTrue((result[:, 10, 2] == 0).all())
        for times in [[0., .01, .01, .03], [0., .02, .01, .03], [0., .01, np.nan, .03], [0., .01]]:
            with self.subTest(times=times), self.assertRaises(ValueError):
                smooth_pose(pose, times)

    def test_ambiguous_target_is_not_silently_chosen(self):
        person = np.ones((65, 3))
        person[:, 0] = np.linspace(.1, .3, 65)
        person[:, 1] = np.linspace(.1, .7, 65)
        other = person.copy()
        other[:, 0] += .5
        with self.assertRaises(JobError):
            Tracker().choose([person, other], 0.)
        self.assertEqual(Tracker([.2, .2]).choose([person, other], 0.), 0)

    def test_lost_target_not_reassigned(self):
        person = np.ones((65, 3))
        person[:, 0] = np.linspace(.1, .3, 65)
        person[:, 1] = np.linspace(.1, .7, 65)
        tracker = Tracker()
        self.assertEqual(tracker.choose([person], 0.), 0)
        self.assertIsNone(tracker.choose([person], 1.))


class EventTests(unittest.TestCase):
    def movement(self):
        times = np.arange(241)/120
        pose = np.zeros((241, 65, 3))
        pose[:, :, 2] = .95
        pose[:, 10, 0] = .2 + .3*np.tanh((times-1.)/.06)
        return times, pose

    def test_audio_gates_not_replaces_visual_peak(self):
        times, pose = self.movement()
        visual = detect_swings(pose, times, [], 'smash')
        fused = detect_swings(pose, times, [1.025], 'smash')
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]['impact']['frame'], 120)
        self.assertIn('audio', fused[0]['impact']['evidence'])
        self.assertGreater(fused[0]['impact']['confidence'], visual[0]['impact']['confidence'])
        self.assertEqual(visual[0]['impact']['evidence'], ['wrist-peak'])

    def test_smoothing_preserves_clean_fast_motion_peak(self):
        for fps in [30, 60, 120]:
            with self.subTest(fps=fps):
                times = np.arange(fps*2+1)/fps
                pose = np.ones((len(times), 65, 3))*.95
                pose[:, :, :2] = .5
                pose[:, 10, 0] = .5 + .3*np.tanh((times-1.)/.06)
                raw = wrist_speed(pose, times, 10)
                filtered = wrist_speed(smooth_pose(pose, times), times, 10)
                self.assertEqual(np.nanargmax(raw), np.nanargmax(filtered))
                self.assertGreater(np.nanmax(filtered)/np.nanmax(raw), .9)

    def test_phase_boundaries_do_not_bridge_long_missing_gap(self):
        times, pose = self.movement()
        # Occlude before the speed falls below the phase threshold, not after a
        # valid phase boundary (the symmetric filter no longer delays that fall).
        pose[125:150, 10] = [np.nan, np.nan, 0]
        smoothed = smooth_pose(pose, times)
        events = detect_swings(smoothed, times, [], 'smash')
        self.assertEqual(len(events), 1)
        self.assertLess(events[0]['impact']['frame'], 125)
        self.assertNotIn('followThroughEndFrame', events[0]['phases'])
        self.assertTrue(np.isnan(smoothed[140, 10, 0]))

    def test_missing_impact_window_does_not_create_event(self):
        times, pose = self.movement()
        pose[110:132, 10] = [np.nan, np.nan, 0]
        events = detect_swings(smooth_pose(pose, times), times, [1.], 'smash')
        self.assertFalse(any(110 <= e['impact']['frame'] <= 132 for e in events))

    def test_audio_only_does_not_create_event(self):
        times, pose = self.movement()
        pose[:, :, :2] = .5
        self.assertEqual(detect_swings(pose, times, [1.], 'smash'), [])

    def test_onset_detection(self):
        signal = np.zeros(16000)
        signal[8000:8160] = .8
        self.assertAlmostEqual(audio_onsets(signal, 16000)[0], .5, delta=.01)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg is required for integration test')
class IngestIntegrationTests(unittest.TestCase):
    def test_proxy_pts_match_capture_clock_for_30_to_120_remap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'source.mp4'
            output = root/'output'
            output.mkdir()
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'testsrc2=s=64x64:r=30:d=1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
            args = SimpleNamespace(input=str(source), capture_fps=120., video_id='timing-test')
            video, capture_clock, _ = ingest(args, output)
            decoded = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(output/'proxy.mp4')]))['frames']
            proxy_clock = np.array([float(f['best_effort_timestamp_time']) for f in decoded])
            self.assertEqual(len(proxy_clock), 30)
            self.assertEqual(video['captureFps'], 120)
            self.assertTrue(video['timeRemapped'])
            self.assertTrue(np.all(np.diff(proxy_clock) > 0))
            np.testing.assert_allclose(proxy_clock, capture_clock, atol=2e-5)


class EvaluationTests(unittest.TestCase):
    def test_one_prediction_cannot_match_two_events(self):
        pairs, missed, extras = match_events([100, 104], [102])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(missed, 1)
        self.assertEqual(extras, 0)

    def test_false_positives_and_misses_count(self):
        pairs, missed, extras = match_events([10, 80], [12, 40, 81])
        self.assertEqual(len(pairs), 2)
        self.assertEqual(missed, 0)
        self.assertEqual(extras, 1)

    def test_synthetic_manifest_cannot_claim_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'manifest.json'
            path.write_text(json.dumps({'synthetic': True, 'cases': [{'id': 'demo'}]}))
            with self.assertRaises(ValueError):
                evaluate(path)


if __name__ == '__main__':
    unittest.main()
