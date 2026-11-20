import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, readFile, writeFile, copyFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {createApp} from '../src/app.js';
import type {Analysis, VideoAsset} from '@badminton/schema';
import type {Pose3DConfig} from '../src/pose3d-jobs.js';
import {selectPose3DWindows} from '../src/pose3d-windows.js';

async function setup(mode = 'success', runWorker = true, enabled = true, workerScript = path.resolve('apps/gateway/test/pose3d-worker-fixture.py')) {
  const dataDir = await mkdtemp(path.join(tmpdir(), 'pose3d-api-'));
  const weightsPath = path.join(dataDir, 'fixture.weights'); await writeFile(weightsPath, 'synthetic transport fixture');
  const config: Pose3DConfig = {pipelineVersion: 'pose3d-test', viewGatePolicyVersion: 'view-test', modelId: 'rtmw3d-rtmlib-x',
    runtime: 'rtmlib-onnx', modelVersion: 'test-only', weightsPath, preSec: .2, postSec: .3, minViewConfidence: .6,
    maxFullDurationSec: 10, maxFrames: 1000, maxFullFrames: 1000, maxWindows: 100, fixtureMode: mode};
  const service = await createApp({dataDir, runWorker: false, pose3d: {enabled, runWorker, config, timeoutMs: mode === 'timeout' ? 200 : 2000,
    workerScript}});
  const analysis: Analysis = JSON.parse(await readFile('fixtures/demo-analysis.json', 'utf8'));
  const video: VideoAsset = JSON.parse(await readFile('fixtures/demo-video.json', 'utf8'));
  analysis.id = 'transport'; analysis.videoId = video.id = 'fixture';
  analysis.quality!.view = 'side'; analysis.quality!.viewConfidence = 1;
  const dir = path.join(dataDir, 'results', analysis.id); await mkdir(dir, {recursive: true});
  for (const name of ['timestamps.f32', 'keypoints.f32']) await copyFile(path.join('fixtures', name), path.join(dir, name));
  analysis.skeleton!.timestampsKey = '/data/results/transport/timestamps.f32';
  analysis.skeleton!.keypointsKey = '/data/results/transport/keypoints.f32';
  video.proxyKey = '/data/results/transport/proxy.mp4'; await writeFile(path.join(dir, 'proxy.mp4'), 'fixture');
  await service.store.put('analyses', analysis.id, {id: analysis.id, status: 'succeeded', analysis, video});
  const server = service.app.listen(0, '127.0.0.1'); await new Promise<void>(resolve => server.once('listening', resolve));
  const base = `http://127.0.0.1:${(server.address() as {port: number}).port}`;
  const request = async (suffix = '', body?: unknown) => fetch(`${base}/api/analyses/transport${suffix}`, {
    method: body === undefined ? 'GET' : 'POST', headers: {'Content-Type': 'application/json'}, body: body === undefined ? undefined : JSON.stringify(body)});
  return {...service, base, dataDir, config, request, analysis, async update() {
    await service.store.put('analyses', analysis.id, {id: analysis.id, status: 'succeeded', analysis, video});
  }, async close() {service.close(); await new Promise<void>(resolve => server.close(() => resolve()));
    // Allow killed children to persist their terminal state before removing the store.
    await new Promise(resolve => setTimeout(resolve, 80)); await rm(dataDir, {recursive: true, force: true});}};
}
async function terminal(s: Awaited<ReturnType<typeof setup>>) {
  for (let i = 0; i < 100; i++) {
    const body = await (await s.request('/pose3d')).json();
    if (body.pose3d && !['queued', 'running'].includes(body.pose3d.status)) return body;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  throw new Error('3D job did not finish');
}
const manual = {scope: 'manual', windows: [{startFrame: 10, endFrame: 20}]};

test('3D default off keeps original legacy 2D record and reports distinct errors', async () => {
  const s = await setup('success', false, false);
  try {
    assert.equal((await (await s.request('/pose3d')).json()).errorCode, 'feature_disabled');
    const post = await s.request('/pose3d', manual); assert.equal(post.status, 503);
    assert.equal((await post.json()).errorCode, 'feature_disabled');
    const original = await (await s.request()).json(); assert.equal(original.analysis.schemaVersion, '0.1.0');
    assert.equal(original.analysis.status, 'succeeded');
    assert.equal((await fetch(s.base + '/api/analyses/absent/pose3d')).status, 404);
  } finally {await s.close();}
});

test('normalized windows are idempotent, concurrent submissions cannot duplicate, cancel is isolated', async () => {
  const s = await setup('success', false);
  try {
    const a = {scope: 'manual', windows: [{startFrame: 15, endFrame: 20}, {startFrame: 10, endFrame: 14}]};
    const responses = await Promise.all([s.request('/pose3d', a), s.request('/pose3d', manual)]);
    const bodies = await Promise.all(responses.map(r => r.json()));
    assert.equal(bodies[0].jobId, bodies[1].jobId); assert.equal(bodies[0].pose3d.windows.length, 1);
    assert.equal((await s.request('/pose3d', {scope: 'manual', windows: [{startFrame: 30, endFrame: 40}]})).status, 409);
    assert.equal((await s.request('/pose3d/cancel', {})).status, 200);
    assert.equal((await terminal(s)).pose3d.errorCode, 'cancelled');
    assert.equal((await (await s.request()).json()).analysis.status, 'succeeded');
    assert.equal((await s.request('/pose3d/retry', {})).status, 202);
  } finally {await s.close();}
});

test('valid artifact is checked and published without changing the original 2D status or metrics', async () => {
  const s = await setup();
  try {
    const original = JSON.stringify(s.analysis.metrics);
    assert.equal((await s.request('/pose3d', manual)).status, 202);
    const job = await terminal(s); assert.equal(job.pose3d.status, 'succeeded', JSON.stringify(job));
    const response = await fetch(s.base + job.pose3d.keypointsKey);
    assert.equal(response.status, 200); assert.equal((await response.arrayBuffer()).byteLength, 720 * 65 * 4 * 4);
    assert.equal((await fetch(s.base + job.pose3d.keypointsKey.replace('pose3d.f32', 'request.json'))).status, 404);
    for (const route of [
      `/data/results/transport/%70ose3d/${job.jobId}/request.json`,
      `/data/results/transport/po%73e3d/${job.jobId}/provenance.json`,
      `/data/results/transport%2Fpose3d%2F${job.jobId}%2Frequest.json`,
      `/data/results/transport/%70ose3d/${job.jobId}/pose3d.f32`,
    ]) assert.equal((await fetch(s.base + route)).status, 404, route);
    const result = await (await s.request()).json(); assert.equal(result.analysis.schemaVersion, '0.2.0');
    assert.equal(result.analysis.status, 'succeeded'); assert.equal(JSON.stringify(result.analysis.metrics), original);
    const stored = await s.store.get<{analysis: Analysis}>('analyses', 'transport'); assert.equal(stored!.analysis.schemaVersion, '0.1.0');
    assert.equal((await s.request('/pose3d/retry', {})).status, 409);
    const cached = await (await s.request('/pose3d', manual)).json(); assert.equal(cached.jobId, job.jobId); assert.equal(cached.attempts, 1);
  } finally {await s.close();}
});

test('model failure, timeout, crash, corrupted binary, traversal, symlink and evaluation output never replace 2D success', async () => {
  for (const mode of ['unavailable', 'timeout', 'crash', 'corrupt', 'traversal', 'symlink', 'evaluation', 'transform']) {
    const s = await setup(mode);
    try {
      await s.request('/pose3d', manual); const job = await terminal(s);
      assert.equal(job.pose3d.status, 'failed', mode); assert.equal(job.pose3d.keypointsKey, undefined);
      assert.equal((await (await s.request()).json()).analysis.status, 'succeeded');
      assert.equal((await fetch(`${s.base}/data/results/transport/pose3d/${job.jobId}/pose3d.f32`)).status, 404);
    } finally {await s.close();}
  }
});

test('scope limits, unknown view, no events, and missing weights have explicit outcomes', async () => {
  const s = await setup('success', false);
  try {
    for (const body of [{scope: 'bogus'}, {scope: 'manual', windows: [{startFrame: -1, endFrame: 20}]}, {evaluationMode: true}])
      assert.equal((await s.request('/pose3d', body)).status, 400);
    assert.equal((await s.request('/pose3d', {...manual, modelId: 'unknown'})).status, 422);
    s.config.maxFullDurationSec = 1;
    assert.equal((await (await s.request('/pose3d', {scope: 'full'})).json()).errorCode, 'scope_limit_exceeded');
    s.config.maxFullDurationSec = 10; s.config.maxFullFrames = 100;
    assert.equal((await (await s.request('/pose3d', {scope: 'full'})).json()).errorCode, 'scope_limit_exceeded');
    s.config.maxWindows = 1;
    assert.equal((await (await s.request('/pose3d', {scope: 'manual', windows: [{startFrame: 0, endFrame: 10}, {startFrame: 20, endFrame: 30}]})).json()).errorCode, 'invalid_windows');
    s.analysis.quality!.view = 'unknown'; await s.update();
    const blocked = await (await s.request('/pose3d', manual)).json(); assert.equal(blocked.pose3d.errorCode, 'view_blocked');
    assert.equal(blocked.attempts, 0);
    s.analysis.swings = []; s.analysis.metrics = []; await s.update();
    assert.equal((await (await s.request('/pose3d', {})).json()).errorCode, 'no_swing_windows');
    s.config.weightsPath = undefined;
    assert.equal((await (await s.request('/pose3d', manual)).json()).errorCode, 'model_unavailable');
  } finally {await s.close();}
});

test('running cancellation cannot be overwritten by worker exit, and the next independent job runs', async () => {
  const s = await setup('timeout');
  try {
    await s.request('/pose3d', manual);
    for (let i = 0; i < 50; i++) {
      const job = await (await s.request('/pose3d')).json();
      if (job.pose3d.status === 'running') break;
      await new Promise(resolve => setTimeout(resolve, 5));
    }
    const cancelled = await (await s.request('/pose3d/cancel', {})).json();
    assert.equal(cancelled.pose3d.errorCode, 'cancelled');
    await new Promise(resolve => setTimeout(resolve, 60));
    assert.equal((await terminal(s)).pose3d.errorCode, 'cancelled');
    s.config.fixtureMode = 'success';
    await s.request('/pose3d', manual);
    assert.equal((await terminal(s)).pose3d.status, 'succeeded');
    assert.equal((await (await s.request()).json()).analysis.status, 'succeeded');
  } finally {await s.close();}
});

test('restart recovers an interrupted 3D job without editing the stored 2D analysis', async () => {
  const s = await setup('success', false);
  try {
    const queued = await (await s.request('/pose3d', manual)).json();
    const job = await s.store.get<{result: {status: string}}> ('pose3d-jobs', queued.jobId);
    job!.result.status = 'running'; await s.store.put('pose3d-jobs', queued.jobId, job);
    const restarted = await createApp({dataDir: s.dataDir, runWorker: false, pose3d: {enabled: true}});
    try {
      const recovered = await terminal(s); assert.equal(recovered.pose3d.errorCode, 'interrupted');
      assert.equal((await (await s.request()).json()).analysis.status, 'succeeded');
      assert.equal((await s.request('/pose3d/retry', {})).status, 202);
    } finally {restarted.close();}
  } finally {await s.close();}
});

test('real research CLI fails closed even with a configured weight file and leaves all 2D artifacts readable', async () => {
  const s = await setup('unused', true, true, path.resolve('worker/pose3d_main.py'));
  try {
    const defaults = JSON.parse(await readFile('worker/pose3d/config.json', 'utf8'));
    const weightsPath = s.config.weightsPath;
    Object.assign(s.config, defaults, {weightsPath});
    await s.request('/pose3d', manual);
    const job = await terminal(s);
    assert.equal(job.pose3d.status, 'failed'); assert.equal(job.pose3d.errorCode, 'model_unavailable', JSON.stringify(job));
    assert.equal((await fetch(s.base + s.analysis.skeleton!.keypointsKey)).status, 200);
  } finally {await s.close();}
});

test('time windows have equal duration at 30/60/120 FPS and low-confidence view is blocked', async () => {
  const s = await setup('success', false);
  try {
    for (const fps of [30, 60, 120]) {
      const a = structuredClone(s.analysis); a.skeleton!.frameCount = fps * 3;
      a.swings = [{...a.swings[0], impact: {...a.swings[0].impact, frame: fps, tSec: 1}}];
      a.quality!.viewConfidence = .1;
      const ts = Float32Array.from({length: fps * 3}, (_, i) => i / fps);
      const [w] = selectPose3DWindows(a, ts, 'swing-windows', undefined, s.config);
      assert.ok(Math.abs(ts[w.endFrame] - ts[w.startFrame] - .5) < 2 / fps);
      assert.equal(w.viewGate, 'blocked');
    }
  } finally {await s.close();}
});
