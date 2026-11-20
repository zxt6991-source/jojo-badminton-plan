#!/usr/bin/env python3
"""Prepare local-only research inputs from existing successful 2D analyses.

Does not run 3D inference, upload media, label ground truth, or pass a gate.
Authorization must be supplied explicitly as an existing local JSON/Markdown file.
"""
from __future__ import annotations
import argparse
from datetime import datetime, date
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from pose3d_evaluate import digest, floats, read_json, require, write_report

ID = re.compile(r'^[a-zA-Z0-9_-]+$')
MIN_CONFIDENCE = .4
MIN_BODY_JOINTS = 5
TIMESTAMP_TOLERANCE_SEC = .0001


def identifier(value):
    require(isinstance(value, str) and ID.fullmatch(value), 'invalid_id')
    return value


def local_file(root, path, expected_directory=None):
    """Reject traversal and symlink substitution, including parent directories."""
    root = Path(root).resolve(strict=True)
    path = Path(path)
    require(path.is_absolute(), 'absolute_input_path_required')
    require('..' not in path.parts, 'input_path_traversal')
    resolved = path.resolve(strict=True)
    require(path == resolved, 'input_symlink_not_allowed')
    require(resolved.is_file() and root in resolved.parents, 'input_outside_data_root')
    if expected_directory is not None:
        require(resolved.parent == expected_directory, 'input_outside_expected_directory')
    return resolved


def artifact(root, analysis_id, key, filename):
    expected = f'/data/results/{analysis_id}/{filename}'
    require(key == expected, 'invalid_artifact_key')
    directory = root / 'results' / analysis_id
    return local_file(root, directory / filename, directory)


def probe_proxy(path):
    args = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_frames',
            '-show_entries', 'stream=width,height:frame=best_effort_timestamp_time', '-of', 'json', str(path)]
    result = subprocess.run(args, check=True, capture_output=True, timeout=120)
    value = json.loads(result.stdout)
    require(len(value.get('streams', [])) == 1, 'proxy_video_stream_missing')
    times = [float(frame['best_effort_timestamp_time']) for frame in value['frames']]
    require(times and all(math.isfinite(t) for t in times), 'proxy_timestamps_missing')
    require(all(a < b for a, b in zip(times, times[1:])), 'proxy_timestamps_not_increasing')
    require(abs(times[0]) <= TIMESTAMP_TOLERANCE_SEC, 'proxy_clock_origin_mismatch')
    stream = value['streams'][0]
    return {'width': stream['width'], 'height': stream['height'], 'timestamps': times}


def validate_keypoints(data, count):
    require(len(data) == count * 65 * 3, 'keypoints_frame_count_mismatch')
    for offset in range(0, len(data), 3):
        x, y, confidence = data[offset:offset + 3]
        valid = (all(math.isfinite(n) for n in (x, y, confidence))
                 and 0 <= x <= 1 and 0 <= y <= 1 and 0 <= confidence <= 1)
        missing = math.isnan(x) and math.isnan(y) and confidence == 0
        require(valid or missing, 'invalid_keypoints_encoding')


def bbox_at(data, frame, width, height):
    points = [data[(frame * 65 + joint) * 3:(frame * 65 + joint) * 3 + 3] for joint in range(17)]
    usable = [p for p in points if all(math.isfinite(v) for v in p) and p[2] >= MIN_CONFIDENCE]
    if len(usable) < MIN_BODY_JOINTS:
        return {'bboxNormalized': None, 'bboxPixels': None, 'visibleBodyJoints': len(usable), 'usable': False}
    x1, y1 = min(p[0] for p in usable), min(p[1] for p in usable)
    x2, y2 = max(p[0] for p in usable), max(p[1] for p in usable)
    if x2 <= x1 or y2 <= y1:
        return {'bboxNormalized': None, 'bboxPixels': None, 'visibleBodyJoints': len(usable), 'usable': False}
    return {'bboxNormalized': [x1, y1, x2, y2], 'bboxPixels': [x1 * width, y1 * height, x2 * width, y2 * height],
            'visibleBodyJoints': len(usable), 'usable': True}


def prepare_analysis(root, record_path, probe=probe_proxy):
    record_path = local_file(root, record_path, root / 'analyses')
    record = read_json(record_path)
    analysis = record['analysis']
    analysis_id, video_id = identifier(record['id']), identifier(analysis['videoId'])
    require(record_path.stem == analysis_id == analysis['id'], 'analysis_id_mismatch')
    require(record['status'] == analysis['status'] == 'succeeded', 'analysis_not_succeeded')
    require(analysis['schemaVersion'] in ('0.1.0', '0.2.0'), 'unsupported_analysis_version')
    skeleton, video = analysis['skeleton'], record['video']
    require(skeleton['spec'] == 'cwb-65' and skeleton['coordSpace'] == 'image-normalized'
            and skeleton['source'] == 'mono-2d', 'not_existing_mono_2d')
    count = skeleton['frameCount']
    require(type(count) is int and 0 < count <= 20000 and video['frameCount'] == count, 'frame_count_mismatch')
    require(video['id'] == video_id, 'video_id_mismatch')
    width, height = video['width'], video['height']
    require(type(width) is int and type(height) is int and width > 0 and height > 0, 'invalid_video_dimensions')
    timestamps_path = artifact(root, analysis_id, skeleton['timestampsKey'], 'timestamps.f32')
    keypoints_path = artifact(root, analysis_id, skeleton['keypointsKey'], 'keypoints.f32')
    proxy_path = artifact(root, analysis_id, video['proxyKey'], 'proxy.mp4')
    times = floats(timestamps_path, count)
    require(all(math.isfinite(t) and t >= 0 for t in times)
            and all(a < b for a, b in zip(times, times[1:])) and abs(times[0]) <= TIMESTAMP_TOLERANCE_SEC,
            'invalid_timestamps')
    data = floats(keypoints_path, count * 65 * 3)
    validate_keypoints(data, count)
    decoded = probe(proxy_path)
    require(decoded['width'] == width and decoded['height'] == height, 'proxy_dimensions_mismatch')
    require(len(decoded['timestamps']) == count, 'proxy_frame_count_mismatch')
    require(all(math.isfinite(t) for t in decoded['timestamps'])
            and all(a < b for a, b in zip(decoded['timestamps'], decoded['timestamps'][1:])), 'invalid_proxy_timestamps')
    max_error = max(abs(a - b) for a, b in zip(times, decoded['timestamps']))
    require(max_error <= TIMESTAMP_TOLERANCE_SEC, 'proxy_timestamp_mismatch')
    video_record_path = local_file(root, root / 'videos' / f'{video_id}.json', root / 'videos')
    video_record = read_json(video_record_path)
    require(video_record['id'] == video_id, 'original_video_id_mismatch')
    original_path = local_file(root, Path(video_record['inputPath']), root / 'videos' / video_id)
    require(original_path.name in ('original.mp4', 'original.mov', 'original.m4v', 'original.webm'), 'invalid_original_filename')
    if 'storageKey' in video:
        require(video['storageKey'] == f'/data/videos/{video_id}/{original_path.name}', 'original_storage_key_mismatch')
    frames = [{'frame': f, 'tSec': float(times[f]), **bbox_at(data, f, width, height)} for f in range(count)]
    swings = []
    seen = set()
    for swing in analysis['swings']:
        impact = swing['impact']
        frame = impact['frame']
        require(type(frame) is int and 0 <= frame < count and frame not in seen, 'invalid_swing_frame')
        seen.add(frame)
        require(type(impact['tSec']) in (int, float) and math.isfinite(impact['tSec'])
                and abs(impact['tSec'] - times[frame]) <= TIMESTAMP_TOLERANCE_SEC, 'swing_timestamp_mismatch')
        confidence = impact['confidence']
        require(type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1, 'invalid_swing_confidence')
        swings.append({**frames[frame], 'existing2DConfidence': confidence,
                       'existing2DEvidence': impact.get('evidence', []),
                       'source': 'existing-2d-event-candidate', 'humanVerified': False})
    quality = analysis.get('quality', {})
    require(isinstance(quality, dict), 'invalid_quality_record')
    predicted_view = quality.get('view', 'unknown')
    require(predicted_view in ('side', 'oblique', 'front', 'back', 'unknown'), 'invalid_predicted_view')
    view_confidence = quality.get('viewConfidence', 0)
    require(type(view_confidence) in (int, float) and math.isfinite(view_confidence) and 0 <= view_confidence <= 1, 'invalid_view_confidence')
    return {
        'analysisId': analysis_id, 'videoId': video_id,
        'analysisPath': str(record_path), 'analysisSha256': digest(record_path),
        'originalVideoPath': str(original_path), 'originalVideoSha256': digest(original_path),
        'proxyPath': str(proxy_path), 'proxySha256': digest(proxy_path),
        'modelInputPath': str(proxy_path), 'modelInputClock': 'existing-2d-capture-timestamps',
        'timestampsPath': str(timestamps_path), 'timestampsSha256': digest(timestamps_path),
        'keypointsPath': str(keypoints_path), 'keypointsSha256': digest(keypoints_path),
        'frameCount': count, 'width': width, 'height': height,
        'captureFpsFrom2D': video.get('captureFps'),
        'proxyValidation': {'decodedFrameCount': count, 'maxTimestampErrorSec': max_error,
                            'timestampToleranceSec': TIMESTAMP_TOLERANCE_SEC},
        'predictedView': {'value': predicted_view, 'confidence': view_confidence,
                          'source': 'existing-2d-heuristic', 'humanVerified': False},
        'target': {'existing2DTrack': analysis.get('target'), 'humanIdentityVerified': False},
        'bboxPolicy': {'source': 'existing-2d-body17', 'minConfidence': MIN_CONFIDENCE,
                       'minBodyJoints': MIN_BODY_JOINTS, 'padding': 0, 'humanVerified': False},
        'usableBBoxFrames': sum(f['usable'] for f in frames), 'frames': frames,
        'candidateSwingFrames': swings, 'annotationStatus': 'not_annotated',
    }


def read_authorization(evidence):
    if evidence.suffix.lower() == '.json':
        authorization = read_json(evidence)
    else:
        # A plain affirmative sentence has no file hashes to bind the consent.
        # Markdown must embed exactly one structured authorization JSON block.
        blocks = re.findall(r'```json\s*([\s\S]*?)```', evidence.read_text())
        require(len(blocks) == 1, 'markdown_requires_one_authorization_json_block')
        authorization = json.loads(blocks[0])
    require(isinstance(authorization, dict), 'invalid_authorization_object')
    require(authorization.get('schemaVersion') == 'pose3d-local-authorization-0.1.0'
            and authorization.get('scope') == 'local-internal-3d-validation'
            and authorization.get('externalUploadAllowed') is False, 'authorization_scope_mismatch')
    require(isinstance(authorization.get('userStatement'), str) and authorization['userStatement'].strip(),
            'authorization_statement_missing')
    try:
        date.fromisoformat(authorization['authorizedAt'])
        retained = authorization.get('retentionUntil')
        require(retained is None or date.fromisoformat(retained) >= date.today(), 'authorization_expired')
    except (KeyError, TypeError) as error:
        raise ValueError('invalid_authorization_date') from error
    items = authorization.get('items')
    require(isinstance(items, list) and items, 'authorization_items_missing')
    seen = set()
    for item in items:
        require(isinstance(item, dict), 'invalid_authorization_item')
        video_id = identifier(item.get('videoId'))
        require(video_id not in seen, 'duplicate_authorized_video')
        seen.add(video_id)
        for key in ('originalSha256', 'proxySha256'):
            require(isinstance(item.get(key), str) and re.fullmatch('[a-fA-F0-9]{64}', item[key]), 'invalid_authorization_hash')
        for key in ('originalPath', 'proxyPath'):
            require(isinstance(item.get(key), str) and Path(item[key]).is_absolute(), 'authorization_absolute_path_required')
    return authorization


def check_authorized(item, authorization):
    consent = next((v for v in authorization['items'] if v['videoId'] == item['videoId']), None)
    require(consent is not None, 'video_not_authorized')
    for key, source in (('originalPath', 'originalVideoPath'), ('proxyPath', 'proxyPath')):
        require(Path(consent[key]).resolve(strict=True) == Path(item[source]), 'authorization_path_mismatch')
    for key, source in (('originalSha256', 'originalVideoSha256'), ('proxySha256', 'proxySha256')):
        require(consent[key].lower() == item[source], 'authorization_hash_mismatch')


def prepare(data_dir, authorization_evidence, probe=probe_proxy):
    root = Path(data_dir).resolve(strict=True)
    require(authorization_evidence is not None, 'authorization_evidence_required')
    evidence = Path(authorization_evidence).resolve(strict=True)
    require(evidence.is_file() and evidence.suffix.lower() in ('.json', '.md') and evidence.stat().st_size > 0,
            'invalid_authorization_evidence')
    authorization = read_authorization(evidence)
    config_path = Path(__file__).resolve().parents[1] / 'worker' / 'pose3d' / 'config.json'
    policy = read_json(config_path)
    min_view_confidence = policy['viewGate']['minConfidence']
    require(type(min_view_confidence) in (int, float) and 0 <= min_view_confidence <= 1, 'invalid_view_policy')
    records, excluded = [], []
    for path in sorted((root / 'analyses').glob('*.json')):
        try:
            path = local_file(root, path, root / 'analyses')
            record = read_json(path)
            require(isinstance(record.get('analysis'), dict), 'invalid_analysis_record')
            identifier(record['id'])
            require(record['id'] == path.stem, 'analysis_id_mismatch')
            video_id = identifier(record['analysis']['videoId'])
            if record.get('status') != 'succeeded' or record['analysis'].get('status') != 'succeeded':
                excluded.append({'analysisId': record['id'], 'videoId': video_id, 'reason': 'analysis_not_succeeded',
                                 'existingStatus': record.get('status')})
                continue
            created = datetime.fromisoformat(record['createdAt'].replace('Z', '+00:00'))
            require(created.tzinfo is not None, 'analysis_creation_timezone_required')
            records.append((created, record['id'], video_id, path))
        except (ValueError, TypeError, KeyError, OSError) as error:
            excluded.append({'analysisPath': str(path), 'reason': 'invalid_analysis_record', 'detail': str(error)})
    # Keep one result per source video, latest successful analysis first. An invalid
    # latest artifact is reported, not silently replaced by an older run.
    latest, inputs = set(), []
    for _created, analysis_id, video_id, path in sorted(records, reverse=True):
        if video_id in latest:
            excluded.append({'analysisId': analysis_id, 'videoId': video_id, 'reason': 'older_successful_analysis'})
            continue
        latest.add(video_id)
        try:
            require(any(v['videoId'] == video_id for v in authorization['items']), 'video_not_authorized')
            item = prepare_analysis(root, path, probe)
            check_authorized(item, authorization)
            view = item['predictedView']
            gate = ('blocked' if view['value'] == 'unknown' or view['confidence'] < min_view_confidence
                    else 'pass' if view['value'] in ('side', 'oblique') else 'limited')
            item['currentViewGate'] = {'gate': gate, 'policyVersion': policy['viewGatePolicyVersion'],
                'minConfidence': min_view_confidence, 'validationStatus': policy['validationStatus'],
                'userDisplayAllowed': False, 'purpose': 'raw-research-input-only'}
            inputs.append(item)
        except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError) as error:
            excluded.append({'analysisId': analysis_id, 'videoId': video_id, 'reason': 'invalid_research_input', 'detail': str(error)})
    return {'schemaVersion': 'pose3d-research-inputs-0.1.0', 'status': 'draft', 'scope': 'local-research',
            'authorization': {'evidencePath': str(evidence), 'evidenceSha256': digest(evidence)},
            'externalTransferAllowed': False, 'inferenceExecuted': False,
            'annotationStatus': 'not_annotated', 'gateDecision': 'not_assessed', 'releaseReady': False,
            'selectionPolicy': 'latest_successful_2d_analysis_per_video', 'inputs': inputs, 'excluded': excluded,
            'limitations': ['2d_bbox_does_not_verify_identity', 'swing_candidates_are_not_human_labels',
                           'predicted_view_is_not_manual_annotation', 'authorization_evidence_requires_human_scope_review']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--authorization-evidence', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        report = prepare(args.data_dir, args.authorization_evidence)
        # Never overwrite source data or the authorization record with a report.
        output = Path(args.output).resolve()
        protected = {Path(report['authorization']['evidencePath'])}
        for item in report['inputs']:
            protected.update(Path(item[key]) for key in ('analysisPath', 'originalVideoPath', 'proxyPath', 'timestampsPath', 'keypointsPath'))
        require(output not in protected, 'output_must_not_overwrite_source')
        root = Path(args.data_dir).resolve()
        require(not any(root / directory == output or root / directory in output.parents
                        for directory in ('analyses', 'videos', 'results', 'uploads', 'models')),
                'output_must_not_overwrite_source_directory')
        require(output.suffix.lower() == '.json', 'output_must_be_json')
        write_report(output, report)
    except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({'status': 'rejected', 'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({'status': report['status'], 'inputs': len(report['inputs']), 'excluded': len(report['excluded']),
                      'output': str(output), 'gateDecision': 'not_assessed'}, ensure_ascii=False))
    return 0 if report['inputs'] else 2


if __name__ == '__main__':
    sys.exit(main())
