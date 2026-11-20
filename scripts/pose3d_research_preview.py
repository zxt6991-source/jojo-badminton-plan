#!/usr/bin/env python3
"""Create a self-contained local research preview after rechecking its evidence."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile
import numpy as np

from pose3d_probe import ProbeError, require, digest, validate_proxy_clock, MAX_PROBE_FRAMES
from pose3d_decode_audit import (read_report, verify_authorization, verify_decoder,
                                read_frame, decode_frame, sha)
from pose3d_camera_research import read_config, recover_camera, pose_metrics, DISPLAY_MATRIX


def json_file(path):
    require(path.is_file() and path.stat().st_size <= 5_000_000, 'invalid_camera_report')
    try:
        value = json.loads(path.read_text())
    except (ValueError, UnicodeError) as error:
        raise ProbeError('invalid_camera_report', str(error)) from error
    require(isinstance(value, dict), 'invalid_camera_report')
    return value


def contained_file(directory, supplied, expected):
    require(supplied == expected, 'invalid_research_path')
    path = directory / expected
    require(path.is_file() and path.resolve().parent == directory.resolve(), 'research_path_escape_or_missing')
    return path


def research_arrays(path, expected_hash, expected):
    require(path.stat().st_size <= 1_000_000, 'research_size_limit')
    require(digest(path) == sha(expected_hash, 'research_hash_missing'), 'research_hash_mismatch')
    try:
        with zipfile.ZipFile(path) as archive:
            require(len(archive.infolist()) == len(expected)
                    and sum(v.file_size for v in archive.infolist()) <= 1_000_000, 'research_size_limit')
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == set(expected), 'research_fields_mismatch')
            for key, reference in expected.items():
                actual = archive[key]
                require(actual.shape == reference.shape and actual.dtype.kind in 'bif', 'research_array_mismatch', key)
                if reference.dtype.kind in 'bi':
                    require(actual.dtype.kind == reference.dtype.kind and np.array_equal(actual, reference), 'research_array_mismatch', key)
                else:
                    require(np.allclose(actual, reference, rtol=1e-10, atol=1e-10, equal_nan=True), 'research_array_mismatch', key)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        if isinstance(error, ProbeError):
            raise
        raise ProbeError('invalid_research_archive', str(error)) from error


def overlay_data(directory, item, width, height):
    path = contained_file(directory, item.get('overlayPath'), f"frame-{item['frame']:06d}.overlay.png")
    require(24 <= path.stat().st_size <= 20_000_000, 'invalid_overlay_size')
    content = path.read_bytes()
    require(content[:8] == b'\x89PNG\r\n\x1a\n' and content[12:16] == b'IHDR', 'invalid_overlay_png')
    require(int.from_bytes(content[16:20], 'big') == width * 2
            and int.from_bytes(content[20:24], 'big') == height, 'overlay_dimensions_mismatch')
    import cv2
    decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    require(decoded is not None and decoded.shape == (height, width * 2, 3), 'invalid_overlay_png')
    actual_hash = hashlib.sha256(content).hexdigest()
    if 'overlaySha256' in item:
        require(actual_hash == sha(item['overlaySha256'], 'invalid_overlay_hash'), 'overlay_hash_mismatch')
    return {'image': 'data:image/png;base64,' + base64.b64encode(content).decode('ascii'),
            'sha256': actual_hash, 'upstreamHashRecorded': 'overlaySha256' in item}


def collect_evidence(camera_path, probe_path, config_path=None, clock_validator=None):
    camera_path, probe_path = Path(camera_path).resolve(), Path(probe_path).resolve()
    camera, probe = json_file(camera_path), read_report(probe_path)
    require(camera.get('schemaVersion') == 'pose3d-camera-research-0.1.0'
            and camera.get('status') == 'camera_research_recovered', 'unsupported_camera_report')
    require(camera.get('evaluationMode') is True
            and all(camera.get(key) is False for key in ('releaseReady', 'productCompatible', 'rawCoordinatesValidated', 'cameraCoordinatesValidated')),
            'unsafe_research_claim')
    require(camera.get('pointSpec') == 'coco-wholebody-133' and camera.get('unit') == 'model-relative'
            and camera.get('rootJoint') == 'mid-hip'
            and camera.get('cameraAxes') == 'x-right-y-down-z-away-from-camera'
            and camera.get('displayAxes') == 'x-right-y-up-z-toward-camera'
            and camera.get('displayMatrix') == DISPLAY_MATRIX.tolist(), 'unsupported_research_coordinates')
    require(camera.get('probeReportSha256') == digest(probe_path), 'probe_report_hash_mismatch')
    timestamps = verify_authorization(probe)
    denominator = verify_decoder(probe)
    require(camera.get('authorization') == probe['authorization']
            and camera.get('inputSha256') == probe['input']['sha256']
            and camera.get('weightsSha256') == probe['model']['weightsSha256'], 'camera_provenance_mismatch')
    clock = (clock_validator or validate_proxy_clock)(Path(probe['input']['path']), timestamps)
    width, height = clock['width'], clock['height']
    require(camera.get('imageDimensions') == [width, height], 'camera_image_dimensions_mismatch')
    recorded_clock = probe['input'].get('proxyClockValidation')
    require(recorded_clock is None or recorded_clock == clock, 'proxy_clock_metadata_mismatch')
    if 'clockEvidence' in camera:
        require(camera['clockEvidence'].get('independentProxyValidation') == clock, 'camera_clock_metadata_mismatch')
    sources = camera.get('sourceEvidence', {})
    if sources:
        require(isinstance(sources, dict) and sources.get('probeReportSha256') == digest(probe_path)
                and sources.get('proxySha256') == probe['input']['sha256']
                and sources.get('timestampsSha256') == probe['input']['timestampsSha256']
                and sources.get('frameCount') == probe['input']['frameCount'], 'source_evidence_mismatch')
    config_location = config_path or sources.get('cameraConfigPath')
    require(isinstance(config_location, (str, Path)) and '://' not in str(config_location),
            'camera_config_required_for_legacy', '旧报告须显式传 --camera-config 以重验参数来源')
    config_location = Path(config_location).expanduser().resolve()
    config = read_config(config_location)
    require(digest(config_location) == camera.get('cameraConfigSha256'), 'camera_config_hash_mismatch')
    require(not sources or sources.get('cameraConfigSha256') == camera['cameraConfigSha256'], 'source_evidence_mismatch')
    scenarios = camera.get('scenarios')
    require(isinstance(scenarios, list) and [s.get('camera') for s in scenarios if isinstance(s, dict)] == config['scenarios'],
            'camera_scenarios_mismatch')
    require(all((s['imageWidth'], s['imageHeight']) == (width, height) for s in config['scenarios']), 'camera_image_dimensions_mismatch')
    items = probe.get('frames')
    require(isinstance(items, list) and 1 <= len(items) <= MAX_PROBE_FRAMES, 'invalid_probe_frames')
    raw_evidence, frames, indices = [], [], []
    for item in items:
        arrays = read_frame(probe_path.parent, item, probe['input']['frameCount'])
        require(item['captureTimeSec'] == float(timestamps[item['frame']]), 'capture_timestamp_mismatch')
        x1, y1, x2, y2 = arrays['bbox_xyxy']
        require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, 'bbox_outside_camera_image')
        raw_evidence.append((arrays, decode_frame(arrays, denominator)))
        overlay = overlay_data(probe_path.parent, item, width, height)
        frames.append({'frame': item['frame'], 'time': item['captureTimeSec'], **overlay})
        indices.append(item['frame'])
    require(indices == sorted(set(indices)), 'duplicate_or_unsorted_frames')
    rendered, radius = [], 0.
    for scenario in scenarios:
        parameter, stored_frames = scenario['camera'], scenario.get('frames')
        require(isinstance(stored_frames, list) and len(stored_frames) == len(items), 'scenario_frame_count_mismatch')
        poses = []
        for position, (stored, item, raw) in enumerate(zip(stored_frames, items, raw_evidence)):
            require(isinstance(stored, dict) and stored.get('frame') == item['frame']
                    and stored.get('captureTimeSec') == item['captureTimeSec']
                    and stored.get('rawSha256') == item['rawSha256'], 'scenario_frame_mismatch')
            arrays, decoded = raw
            recovered, _ = recover_camera(arrays['keypoints_2d'][0], decoded['official_codec_relative_z'][0],
                                           decoded['decoded_point_mask'][0], parameter)
            expected = {**recovered, 'image_xy': arrays['keypoints_2d'][0], 'scores': arrays['scores'][0],
                        'official_relative_z': decoded['official_codec_relative_z'][0],
                        'frame_index': arrays['frame_index'], 'timestamp_sec': arrays['timestamp_sec']}
            name = f"{parameter['id']}.frame-{item['frame']:06d}.camera-research.npz"
            file = contained_file(camera_path.parent, stored.get('researchPath'), name)
            research_arrays(file, stored.get('researchSha256'), expected)
            valid, xyz = recovered['decoded_point_mask'][:17], recovered['display_root_relative_xyz'][:17]
            points = [xyz[i].tolist() if valid[i] else None for i in range(17)]
            radius = max(radius, float(np.linalg.norm(xyz[valid], axis=1).max()))
            metrics = pose_metrics(recovered['camera_root_relative_xyz'], recovered['decoded_point_mask'])
            poses.append({'points': points, 'elbows': metrics['elbows']})
        rendered.append({'camera': parameter, 'poses': poses})
    require(np.isfinite(radius) and radius > 0, 'degenerate_preview_extent')
    return {'schemaVersion': 'pose3d-offline-preview-0.1.0', 'frames': frames, 'scenarios': rendered,
            'sharedRadius': radius, 'clock': clock, 'originalCaptureTimingVerified': False,
            'sourceEditingHistoryVerified': False, 'releaseReady': False, 'productCompatible': False,
            'cameraReportSha256': digest(camera_path), 'probeReportSha256': digest(probe_path),
            'cameraConfigSha256': digest(config_location), 'weightsSha256': camera['weightsSha256']}


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>羽毛球 · 本机 3D 研究预览</title>
<style>
:root{color-scheme:light;font:15px/1.6 system-ui,-apple-system,sans-serif;color:#233040;background:#f3f5f7}*{box-sizing:border-box}body{margin:0}main{max-width:1400px;margin:auto;padding:28px}h1{font-size:26px;margin:0 0 8px}h2{font-size:17px;margin:0 0 10px}p{margin:8px 0}.muted{color:#586575;font-size:13px}.notice{background:#fff2d8;border-left:4px solid #c57912;padding:12px 16px;margin:18px 0;border-radius:5px}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin:18px 0}label{display:flex;flex-direction:column;gap:4px;font-weight:600}select,button{font:inherit;border:1px solid #c6d0d9;background:white;border-radius:7px;padding:8px 12px;color:inherit}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}button:focus-visible,select:focus-visible{outline:3px solid #6ea9dc}.grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:18px}.card{background:white;border:1px solid #dde3e9;border-radius:12px;padding:18px;min-width:0}.image-wrap{background:#edf0f3;min-height:300px;display:flex;align-items:center;justify-content:center;border-radius:8px;overflow:hidden}img{width:100%;max-height:640px;object-fit:contain;display:block}canvas{display:block;width:100%;height:480px;touch-action:none;cursor:grab;background:#f8fafc;border-radius:8px}canvas:active{cursor:grabbing}.legend{display:flex;flex-wrap:wrap;gap:15px;font-size:13px;margin:10px 0}.left{color:#1673bd}.right{color:#ad640a}.metrics{display:flex;gap:20px;flex-wrap:wrap;font-size:14px}.parameters{word-break:break-word}.small-controls{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}.small-controls button{font-size:13px;padding:5px 9px}details{margin-top:18px;padding:12px;background:white;border-radius:8px}pre{white-space:pre-wrap;word-break:break-all;font-size:12px}footer{margin:20px 0}.frame-title{font-weight:600;margin-bottom:12px}@media(max-width:850px){main{padding:16px}.grid{grid-template-columns:1fr}canvas{height:420px}h1{font-size:22px}}
</style></head><body><main>
<h1>羽毛球 · 本机 3D 研究预览</h1><p class="muted">同一帧的图像对照与参数敏感性观察 · 离线文件，不发送数据</p>
<div class="notice"><strong>研究假设，不可用于实际测量或动作评分。</strong><br>相机参数与深度尚未独立验证；骨架形状、角度随参数改变。所有视图使用相同显示尺度。这里只切换离散采样帧，不播放连续动作。</div>
<div class="controls"><button id="previous">上一采样帧</button><label>采样帧<select id="frame-select"></select></label><button id="next">下一采样帧</button><label>相机参数情景<select id="scenario-select"></select></label></div>
<div class="frame-title" id="frame-title"></div><div class="grid">
<section class="card"><h2>同帧图像与 2D 对照</h2><div class="image-wrap"><img id="overlay" alt="同一采样帧的已有2D与RTMW3D图像关键点对照"></div><p class="muted">左侧：原图或已有 2D；右侧：RTMW3D 图像关键点。图像 XY 一致不能证明深度正确。</p></section>
<section class="card"><h2>研究 XYZ 骨架 · 仅身体 17 点</h2><canvas id="pose-canvas" aria-label="可拖动旋转的研究三维骨架"></canvas>
<div class="legend"><span class="left">● 人体左侧</span><span class="right">● 人体右侧</span><span>● 中线</span></div>
<div class="small-controls"><button id="turn-left">向左旋转</button><button id="turn-right">向右旋转</button><button id="tilt-up">俯视</button><button id="reset">重置视角</button></div>
<p class="muted">拖动旋转；初始坐标 +X 向右、+Y 向上、+Z 朝观察者。标记按人体关节索引，左右识别尚未人工核验。缺失点不补齐。</p>
<div class="metrics"><span id="left-angle"></span><span id="right-angle"></span></div>
</section></div>
<section class="card" style="margin-top:18px"><h2>当前显式参数</h2><p class="parameters" id="parameters"></p><p class="muted" id="evidence"></p><p class="muted">角度只用于比较参数情景；长度为模型相对量，不是米制。切换参数不会重新推理，旋转视图不会改写 XYZ。</p></section>
<footer class="muted">生成时已复核本机授权、文件 hash、逐帧时间与代理 PTS，并重算核对 XYZ。PTS 一致不证明原始采集帧率或原片未剪辑；相机校准、身份与深度准确性仍未完成验证。产品 3D 保持关闭。此 HTML 含视频采样图，仅供本机研究，勿上传或分享。</footer>
<details><summary>查看本次证据摘要</summary><pre id="provenance"></pre></details>
</main><script id="preview-data" type="application/json">__PREVIEW_DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('preview-data').textContent);
const byId=id=>document.getElementById(id), frameSelect=byId('frame-select'), scenarioSelect=byId('scenario-select'), canvas=byId('pose-canvas'), ctx=canvas.getContext('2d');
data.frames.forEach((f,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent=`帧 ${f.frame} · ${f.time.toFixed(4)} 秒`;frameSelect.append(o)});
data.scenarios.forEach((s,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent=s.camera.id;scenarioSelect.append(o)});
let frameIndex=0,scenarioIndex=0,yaw=0,pitch=0,drag=null;
const left=new Set([1,3,5,7,9,11,13,15]),right=new Set([2,4,6,8,10,12,14,16]);
const bones=[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
function rotated(p){const c=Math.cos(yaw),s=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x=c*p[0]+s*p[2],z=-s*p[0]+c*p[2];return [x,cp*p[1]-sp*z,sp*p[1]+cp*z]}
function draw(){const box=canvas.getBoundingClientRect(),w=box.width,h=box.height,dpr=window.devicePixelRatio||1;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const scale=.39*Math.min(w,h)/data.sharedRadius;
const screen=p=>{const r=rotated(p);return [w/2+r[0]*scale,h/2-r[1]*scale,r[2]]};
ctx.strokeStyle='#e2e8ee';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(w/2,25);ctx.lineTo(w/2,h-25);ctx.moveTo(25,h/2);ctx.lineTo(w-25,h/2);ctx.stroke();
const points=data.scenarios[scenarioIndex].poses[frameIndex].points, valid=p=>p&&p.length===3&&p.every(Number.isFinite), projected=points.map(p=>valid(p)?screen(p):null);
const color=i=>left.has(i)?'#1673bd':right.has(i)?'#b87514':'#637184';
bones.filter(([a,b])=>projected[a]&&projected[b]).sort((a,b)=>(projected[a[0]][2]+projected[a[1]][2])-(projected[b[0]][2]+projected[b[1]][2])).forEach(([a,b])=>{ctx.strokeStyle=left.has(a)&&left.has(b)?color(a):right.has(a)&&right.has(b)?color(a):'#8794a0';ctx.lineWidth=4;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(projected[a][0],projected[a][1]);ctx.lineTo(projected[b][0],projected[b][1]);ctx.stroke()});
projected.forEach((p,i)=>{if(!p)return;ctx.beginPath();ctx.fillStyle=color(i);ctx.arc(p[0],p[1],4.5,0,Math.PI*2);ctx.fill();if([5,6,9,10].includes(i)){ctx.font='12px system-ui';ctx.fillText(({5:'左肩',6:'右肩',9:'左腕',10:'右腕'})[i],p[0]+7,p[1]-7)}});
[[[1,0,0],'X','#cf5e5e'],[[0,1,0],'Y','#338855'],[[0,0,1],'Z','#467bb5']].forEach(([axis,label,color])=>{const r=rotated(axis);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(45,h-45);ctx.lineTo(45+r[0]*26,h-45-r[1]*26);ctx.stroke();ctx.font='12px system-ui';ctx.fillText(label,47+r[0]*32,h-45-r[1]*32)});
}
function update(){const frame=data.frames[frameIndex],scenario=data.scenarios[scenarioIndex],c=scenario.camera;frameSelect.value=String(frameIndex);scenarioSelect.value=String(scenarioIndex);byId('previous').disabled=frameIndex===0;byId('next').disabled=frameIndex===data.frames.length-1;byId('overlay').src=frame.image;byId('frame-title').textContent=`采样 ${frameIndex+1}/${data.frames.length} · 帧 ${frame.frame} · 既有时间轴 ${frame.time.toFixed(4)} 秒`;
byId('parameters').textContent=`${c.id} · ${c.source==='calibrated'?'标定来源声明（未独立核验）':'研究假设'} · ${c.imageWidth}×${c.imageHeight} · fx=${c.fx}, fy=${c.fy}, cx=${c.cx}, cy=${c.cy} · rootDepth=${c.rootDepth}（模型相对量）`;byId('evidence').textContent=c.evidence;
for(const [side,label] of [['left','左'],['right','右']]){const angle=scenario.poses[frameIndex].elbows[side].angleDeg;byId(`${side}-angle`).textContent=`${label}肘研究角：${angle===null?'不可用':angle.toFixed(2)+'°'}`;}draw()}
frameSelect.addEventListener('change',()=>{frameIndex=Number(frameSelect.value);update()});scenarioSelect.addEventListener('change',()=>{scenarioIndex=Number(scenarioSelect.value);update()});byId('previous').addEventListener('click',()=>{frameIndex=Math.max(0,frameIndex-1);update()});byId('next').addEventListener('click',()=>{frameIndex=Math.min(data.frames.length-1,frameIndex+1);update()});
canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag.x)*.009;pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-drag.y)*.009));drag={x:e.clientX,y:e.clientY};draw()});for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>drag=null);
byId('turn-left').addEventListener('click',()=>{yaw-=Math.PI/12;draw()});byId('turn-right').addEventListener('click',()=>{yaw+=Math.PI/12;draw()});byId('tilt-up').addEventListener('click',()=>{pitch=Math.max(-1.45,pitch-.15);draw()});byId('reset').addEventListener('click',()=>{yaw=0;pitch=0;draw()});new ResizeObserver(draw).observe(canvas);
byId('provenance').textContent=JSON.stringify({cameraReportSha256:data.cameraReportSha256,probeReportSha256:data.probeReportSha256,cameraConfigSha256:data.cameraConfigSha256,weightsSha256:data.weightsSha256,proxyClock:data.clock,overlays:data.frames.map(f=>({frame:f.frame,sha256:f.sha256,upstreamHashRecorded:f.upstreamHashRecorded})),originalCaptureTimingVerified:false,sourceEditingHistoryVerified:false,releaseReady:false},null,2);update();
</script></body></html>'''


def render_html(data):
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    for character, escape in (('<', '\\u003c'), ('>', '\\u003e'), ('&', '\\u0026'), ('\u2028', '\\u2028'), ('\u2029', '\\u2029')):
        payload = payload.replace(character, escape)
    return HTML.replace('__PREVIEW_DATA__', payload).encode('utf-8')


def atomic_create(path, content):
    """Hard-link publication is atomic and fails if another writer wins the path."""
    path = Path(path).expanduser().absolute()
    require(path.suffix.lower() == '.html', 'html_output_required')
    require(not os.path.lexists(path), 'output_exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.pose3d-preview-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ProbeError('output_exists', '输出已存在，不覆盖既有研究证据') from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def create_preview(camera_report, probe_report, output, camera_config=None, clock_validator=None):
    require(not os.path.lexists(Path(output).expanduser()), 'output_exists')
    data = collect_evidence(camera_report, probe_report, camera_config, clock_validator)
    path = atomic_create(output, render_html(data))
    return {'status': 'research_preview_created', 'path': str(path), 'sha256': digest(path),
            'frames': len(data['frames']), 'scenarios': len(data['scenarios']), 'releaseReady': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-report', required=True)
    parser.add_argument('--probe-report', required=True)
    parser.add_argument('--camera-config', help='旧版相机报告无配置来源路径时必须显式提供')
    parser.add_argument('--output', required=True, help='不存在的本机 .html 文件')
    args = parser.parse_args()
    try:
        result = create_preview(args.camera_report, args.probe_report, args.output, args.camera_config)
    except (ProbeError, OSError, ValueError, TypeError, KeyError) as error:
        result = {'status': 'invalid_evidence', 'errorCode': getattr(error, 'code', 'invalid_evidence'), 'error': str(error)}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'research_preview_created' else 2


if __name__ == '__main__':
    sys.exit(main())
