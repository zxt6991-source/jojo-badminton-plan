"""Transport-only fixture. Not an inference backend or an accuracy test."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
for name in ['input', 'output-dir', 'analysis-id', 'video-id', 'swing-type', 'target-point', 'capture-fps']:
    parser.add_argument('--' + name)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
mode = Path(args.input).read_text()
if mode == 'timeout':
    time.sleep(5)
video = json.loads((root / 'fixtures/demo-video.json').read_text())
analysis = json.loads((root / 'fixtures/demo-analysis.json').read_text())
video['id'] = args.video_id
analysis['id'] = args.analysis_id
analysis['videoId'] = args.video_id
analysis['schemaVersion'] = '0.2.0'
analysis['pipelineVersion'] = json.loads((root / 'pipeline-version.json').read_text())['pipelineVersion']
analysis['metrics'] = []
if mode == 'dependencies':
    analysis['status'] = 'failed'
    analysis.pop('skeleton', None)
    analysis['swings'] = []
    analysis['quality']['usable'] = False
    analysis['quality']['issues'] = [{'code': 'worker_dependencies_missing', 'severity': 'block'}]
    print(json.dumps({'video': video, 'analysis': analysis, 'error': "No module named 'numpy'", 'errorCode': 'worker_dependencies_missing'}))
    sys.exit(0)
for name in ['timestamps.f32', 'keypoints.f32']:
    shutil.copyfile(root / 'fixtures' / name, Path(args.output_dir) / name)
print(json.dumps({'video': video, 'analysis': analysis}))
if mode == 'crash':
    sys.exit(1)
