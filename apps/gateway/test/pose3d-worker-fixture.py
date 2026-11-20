"""Synthetic TRANSPORT fixture only. Never used by a production provider or P0 evaluation."""
import argparse
import json
import struct
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
result = request['pose3d']
mode = request['config'].get('fixtureMode', 'success')
out = Path(args.output_dir)
if mode == 'crash':
    raise RuntimeError('fixture crash')
if mode == 'timeout':
    time.sleep(10)
if mode == 'unavailable':
    result.update(status='failed', errorCode='model_unavailable')
    print(json.dumps({'pose3d': result}))
    raise SystemExit(0)

missing = struct.pack('<ffff', float('nan'), float('nan'), float('nan'), 0)
point = struct.pack('<ffff', 0, 0, 0, 1)
valid_frames = set()
for window in result['windows']:
    if window['viewGate'] != 'blocked':
        valid_frames.update(range(window['startFrame'], window['endFrame'] + 1))
        window['usableFrameRatio'] = 1
binary = b''.join((point if f in valid_frames else missing) * 65 for f in range(result['frameCount']))
file = out / 'pose3d.f32'
if mode == 'corrupt':
    binary = binary[:-1]
file.write_bytes(binary)
if mode == 'symlink':
    file.unlink()
    file.symlink_to(Path(request['timestampsPath']))
result.update(status='succeeded', keypointsKey='../pose3d.f32' if mode == 'traversal' else 'pose3d.f32')
result['quality'].update(coverageRatio=len(valid_frames) / result['frameCount'], usableFrameRatio=1,
                         identityLockedRatio=1, meanJointConfidence=1)
(out / 'provenance.json').write_text(json.dumps({
    'weightsSha256': result['weightsSha256'], 'evaluationMode': mode == 'evaluation',
    'coordinateTransform': {'matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 0 if mode == 'transform' else 1]],
                            'sourceConvention': 'synthetic-transport-test', 'targetConvention': result['axisConvention'],
                            'evidence': 'transport-only-test-not-model-evidence'},
    'runtime': result['runtime'], 'configHash': request['configHash'],
    'versions': {'fixture': '1'}, 'device': 'test', 'config': request['config'],
}))
print(json.dumps({'pose3d': result}))
