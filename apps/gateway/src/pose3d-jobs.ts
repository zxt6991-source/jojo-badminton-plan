import type {Express, Request, Response, NextFunction} from 'express';
import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {spawn} from 'node:child_process';
import {mkdir, readFile, realpath, lstat, stat} from 'node:fs/promises';
import path from 'node:path';
import {parseAnalysis, parsePose3DResult, validatePose3DBuffer, validateSkeletonData} from '@badminton/schema';
import type {Analysis, VideoAsset, Pose3DResult, Pose3DScope} from '@badminton/schema';
import {Store} from './store.js';
import {resolveWorkerPython} from './worker-runtime.js';
import {poseError, selectPose3DWindows, type WindowPolicy} from './pose3d-windows.js';

export interface Pose3DConfig extends WindowPolicy {
  pipelineVersion: string;
  viewGatePolicyVersion: string;
  modelId: Pose3DResult['modelId'];
  runtime: Pose3DResult['runtime'];
  modelVersion: string;
  weightsPath?: string;
  [key: string]: unknown;
}
export interface Pose3DOptions {enabled?: boolean; runWorker?: boolean; workerScript?: string; python?: string; timeoutMs?: number; config?: Pose3DConfig}
interface RecordData {id: string; status: Analysis['status']; analysis: Analysis; video?: VideoAsset}
interface Job {
  id: string; analysisId: string; createdAt: string; attempts: number;
  result: Pose3DResult; config: Pose3DConfig; configHash: string; error?: string;
}
const wrap = (fn: (req: Request, res: Response) => Promise<unknown>) => (req: Request, res: Response, next: NextFunction) => {void fn(req, res).catch(next);};
const digest = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
async function fileDigest(file: string) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}
const canonical = (value: unknown): string => JSON.stringify(value, (_key, v) => v && typeof v === 'object' && !Array.isArray(v) ? Object.fromEntries(Object.keys(v).sort().map(k => [k, v[k]])) : v);
const busy = (job?: Job) => job && ['queued', 'running'].includes(job.result.status);
function validTransform(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false;
  const transform = value as {matrix?: unknown; sourceConvention?: unknown; targetConvention?: unknown; evidence?: unknown};
  if (typeof transform.sourceConvention !== 'string' || !transform.sourceConvention.trim() ||
    typeof transform.evidence !== 'string' || !transform.evidence.trim() || transform.targetConvention !== 'x-right-y-up-z-forward') return false;
  const matrix = transform.matrix;
  if (!Array.isArray(matrix) || matrix.length !== 3 || !matrix.every(row => Array.isArray(row) && row.length === 3 && row.every(v => typeof v === 'number' && Number.isFinite(v)))) return false;
  const gram = matrix.map(row => matrix.map(other => row.reduce((sum: number, v: number, i: number) => sum + v * other[i], 0)));
  const scale = (gram[0][0] + gram[1][1] + gram[2][2]) / 3;
  return Number.isFinite(scale) && scale > 0 && gram.every((row, i) => row.every((v, j) => Math.abs(v - (i === j ? scale : 0)) <= 1e-6 * scale));
}
const floats = (b: Buffer) => {
  if (b.length % 4) throw poseError('invalid_binary', '二维文件长度损坏', 422);
  const out = new Float32Array(b.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = b.readFloatLE(i * 4);
  return out;
};

/** Only regular, explicitly named files below this exact job directory are trusted. */
export async function containedFile(directory: string, name: string, trustedRoot = path.dirname(directory)): Promise<string> {
  if (path.basename(name) !== name || name === '.' || name === '..') throw poseError('invalid_artifact_path', '产物路径无效', 422);
  const resolvedDirectory = await realpath(directory), file = path.join(directory, name);
  const resolvedRoot = await realpath(trustedRoot);
  if (resolvedDirectory !== resolvedRoot && !resolvedDirectory.startsWith(resolvedRoot + path.sep))
    throw poseError('invalid_artifact_path', '产物目录超出数据目录', 422);
  const metadata = await lstat(file), resolvedFile = await realpath(file);
  if (!metadata.isFile() || metadata.isSymbolicLink() || path.dirname(resolvedFile) !== resolvedDirectory)
    throw poseError('invalid_artifact_path', '产物必须为任务目录内的普通文件', 422);
  return resolvedFile;
}

export async function installPose3DJobs(app: Express, store: Store, root: string, options: Pose3DOptions = {}) {
  for (const directory of ['pose3d-jobs', 'pose3d-selections']) await mkdir(path.join(store.root, directory), {recursive: true});
  const enabled = options.enabled ?? process.env.POSE3D_ENABLED === '1';
  const locks = new Map<string, Promise<unknown>>();
  async function locked<T>(id: string, fn: () => Promise<T>): Promise<T> {
    const task = (locks.get(id) ?? Promise.resolve()).catch(() => {}).then(fn);
    locks.set(id, task);
    try {return await task;} finally {if (locks.get(id) === task) locks.delete(id);}
  }
  let active = false, closed = false;
  let child: ReturnType<typeof spawn> | undefined, childJobId: string | undefined;
  const cancelled = new Set<string>();
  const directory = (job: Job) => path.join(store.root, 'results', job.analysisId, 'pose3d', job.id);
  async function record(id: string) {
    const r = await store.get<RecordData>('analyses', id);
    if (!r) throw poseError('analysis_not_found', '分析任务不存在', 404);
    parseAnalysis(r.analysis);
    return r;
  }
  async function selected(id: string) {
    const selection = await store.get<{jobId: string}>('pose3d-selections', id);
    return selection ? store.get<Job>('pose3d-jobs', selection.jobId) : undefined;
  }
  async function attach(analysis: Analysis): Promise<Analysis> {
    const job = await selected(analysis.id);
    if (!job) return analysis;
    // Validate legacy input before exposing an extended v0.2 DTO; stored 2D is untouched.
    parseAnalysis(analysis);
    return parseAnalysis({...analysis, schemaVersion: '0.2.0', pose3d: job.result});
  }
  async function config(): Promise<Pose3DConfig> {
    const c: Pose3DConfig = options.config ?? JSON.parse(await readFile(process.env.POSE3D_CONFIG ?? path.join(root, 'worker/pose3d/config.json'), 'utf8'));
    const windows = c.windows as {preSec: number | null; postSec: number | null} | undefined;
    const gate = c.viewGate as {minConfidence: number} | undefined;
    const limits = c.limits as {maxFrames: number; maxFullDurationSec: number; maxFullFrames: number; maxWindows: number} | undefined;
    const normalized = structuredClone(c);
    if (windows) {normalized.preSec = windows.preSec; normalized.postSec = windows.postSec;}
    if (gate) normalized.minViewConfidence = gate.minConfidence;
    if (limits) {normalized.maxFrames = limits.maxFrames; normalized.maxFullDurationSec = limits.maxFullDurationSec;
      normalized.maxFullFrames = limits.maxFullFrames; normalized.maxWindows = limits.maxWindows;}
    return validateConfig(normalized);
  }
  function validateConfig(c: Pose3DConfig): Pose3DConfig {
    for (const key of ['pipelineVersion', 'viewGatePolicyVersion', 'modelVersion'])
      if (typeof c[key] !== 'string' || !String(c[key]).trim()) throw poseError('invalid_config', `3D 配置缺少 ${key}`, 503);
    for (const key of ['maxFullDurationSec', 'maxFrames'])
      if (typeof c[key] !== 'number' || !Number.isFinite(c[key]) || Number(c[key]) <= 0) throw poseError('invalid_config', `3D 配置 ${key} 无效`, 503);
    if (![c.maxFrames, c.maxFullFrames, c.maxWindows].every(n => Number.isSafeInteger(n) && n > 0) || !Number.isFinite(c.minViewConfidence) || c.minViewConfidence < 0 || c.minViewConfidence > 1)
      throw poseError('invalid_config', '3D 门控配置无效', 503);
    for (const key of ['preSec', 'postSec'] as const)
      if (c[key] !== null && (typeof c[key] !== 'number' || !Number.isFinite(c[key]) || c[key]! < 0)) throw poseError('invalid_config', '3D 窗口配置无效', 503);
    return c;
  }
  async function inputFiles(r: RecordData) {
    const skeleton = r.analysis.skeleton;
    if (r.status !== 'succeeded' || !skeleton || !r.video) throw poseError('analysis_not_succeeded', '须先完成二维分析', 409);
    const dir = path.join(store.root, 'results', r.id), prefix = `/data/results/${r.id}/`;
    const local = async (key: string) => {
      if (!key.startsWith(prefix) || key.slice(prefix.length).includes('/')) throw poseError('invalid_artifact_path', '二维产物路径无效', 422);
      return containedFile(dir, key.slice(prefix.length), store.root);
    };
    const timestampsPath = await local(skeleton.timestampsKey), keypointsPath = await local(skeleton.keypointsKey);
    const timestamps = floats(await readFile(timestampsPath)), keypoints = floats(await readFile(keypointsPath));
    try {validateSkeletonData(timestamps, keypoints, skeleton.frameCount);} catch {throw poseError('invalid_binary', '二维产物校验失败', 422);}
    return {timestampsPath, keypointsPath, timestamps, local};
  }
  async function save(job: Job) {parsePose3DResult(job.result); await store.put('pose3d-jobs', job.id, job);}
  function fail(job: Job, code: string, message: string) {
    job.result.status = 'failed'; job.result.errorCode = code; delete job.result.keypointsKey;
    job.error = message;
  }
  async function run(job: Job) {
    if (cancelled.has(job.id) || closed) throw poseError('cancelled', '3D 子任务已取消', 409);
    const r = await record(job.analysisId), input = await inputFiles(r);
    // Always use the proxy on the established acquisition clock; no guessed FPS.
    if (!r.video?.proxyKey) throw poseError('proxy_unavailable', '缺少采集时钟代理视频', 422);
    const inputPath = await input.local(r.video.proxyKey);
    if (!job.config.weightsPath || await fileDigest(job.config.weightsPath) !== job.result.weightsSha256)
      throw poseError('weights_changed', '权重内容已改变，请重新提交 3D 任务', 409);
    const outputDir = directory(job); await mkdir(outputDir, {recursive: true});
    // The request is private: the public artifact route serves only validated f32.
    const requestPath = path.join(outputDir, 'request.json');
    const {writeFile} = await import('node:fs/promises');
    await writeFile(requestPath, JSON.stringify({analysis: r.analysis, video: r.video, inputPath,
      timestampsPath: input.timestampsPath, keypointsPath: input.keypointsPath, pose3d: job.result, config: job.config, configHash: job.configHash}));
    const timeout = options.timeoutMs ?? Number(process.env.POSE3D_TIMEOUT_MS ?? 120000);
    if (!Number.isFinite(timeout) || timeout < 1) throw poseError('invalid_config', '3D 超时配置无效', 503);
    const stdout = await new Promise<string>((resolve, reject) => {
      if (cancelled.has(job.id) || closed) {reject(poseError('cancelled', '3D 子任务已取消', 409)); return;}
      const proc = spawn(options.python ?? process.env.POSE3D_PYTHON ?? resolveWorkerPython(root),
        [options.workerScript ?? path.join(root, 'worker/pose3d_main.py'), '--request', requestPath, '--output-dir', outputDir],
        {cwd: root, stdio: ['ignore', 'pipe', 'pipe']});
      child = proc; childJobId = job.id;
      let output = '', stderr = '', reason = '';
      const timer = setTimeout(() => {reason = 'timeout'; proc.kill('SIGKILL');}, timeout);
      proc.stdout!.on('data', data => {output += data; if (output.length > 1024 * 1024) {reason = 'output_limit'; proc.kill('SIGKILL');}});
      proc.stderr!.on('data', data => {stderr = (stderr + data).slice(-4000);});
      proc.on('error', error => {clearTimeout(timer); reject(poseError('worker_unavailable', error.message, 503));});
      proc.on('close', code => {
        clearTimeout(timer); if (child === proc) {child = undefined; childJobId = undefined;}
        if (cancelled.has(job.id) || closed) reject(poseError('cancelled', '3D 子任务已取消', 409));
        else if (reason) reject(poseError(reason, reason === 'timeout' ? '3D 推理超时' : '3D 输出超过限制', 422));
        else if (code !== 0) reject(poseError('worker_crashed', `3D 进程退出 (${code})：${stderr}`, 422));
        else resolve(output);
      });
    });
    const payload = JSON.parse(stdout.trim().split('\n').at(-1) ?? '{}');
    const result = parsePose3DResult(payload.pose3d);
    if (!['succeeded', 'failed', 'rejected_quality'].includes(result.status)) throw poseError('invalid_result', '3D 进程未返回终态', 422);
    for (const key of ['source', 'modelId', 'runtime', 'modelVersion', 'pipelineVersion', 'weightsSha256', 'scope', 'viewGatePolicyVersion', 'frameCount'] as const)
      if (result[key] !== job.result[key]) throw poseError('invalid_result', `3D 元数据不匹配：${key}`, 422);
    const ranges = (p: Pose3DResult) => p.windows.map(({usableFrameRatio: _ratio, ...w}) => w);
    if (canonical(ranges(result)) !== canonical(ranges(job.result))) throw poseError('invalid_result', '3D 窗口或视角门控被修改', 422);
    result.quality.warnings = [...new Set([...job.result.quality.warnings, ...result.quality.warnings])];
    // Provenance includes local paths/configuration and is retained privately.
    delete result.provenanceKey;
    if (result.status === 'succeeded') {
      if (result.keypointsKey !== 'pose3d.f32') throw poseError('invalid_artifact_path', '3D 文件名必须为 pose3d.f32', 422);
      const file = await containedFile(outputDir, result.keypointsKey, store.root);
      if ((await stat(file)).size !== result.frameCount * 65 * 4 * 4) throw poseError('invalid_binary', '3D 文件长度不匹配', 422);
      validatePose3DBuffer(await readFile(file), result);
      const provenance = JSON.parse(await readFile(await containedFile(outputDir, 'provenance.json', store.root), 'utf8'));
      if (provenance.weightsSha256 !== result.weightsSha256 || provenance.evaluationMode !== false ||
        !validTransform(provenance.coordinateTransform) || provenance.configHash !== job.configHash ||
        provenance.runtime !== result.runtime || !provenance.versions || typeof provenance.versions !== 'object' || !Object.keys(provenance.versions).length ||
        !Object.values(provenance.versions).every(v => typeof v === 'string' && v.trim()) ||
        typeof provenance.device !== 'string' || !provenance.device || canonical(provenance.config) !== canonical(job.config))
        throw poseError('invalid_provenance', '3D 缺少可复现来源或属于仅评测结果', 422);
      result.keypointsKey = `/data/results/${job.analysisId}/pose3d/${job.id}/pose3d.f32`;
    }
    await locked(job.analysisId, async () => {
      const current = await store.get<Job>('pose3d-jobs', job.id);
      if (cancelled.has(job.id) || closed || current?.result.status !== 'running')
        throw poseError('cancelled', '3D 子任务已取消', 409);
      job.result = result; job.error = typeof payload.error === 'string' ? payload.error : undefined; await save(job);
    });
  }
  async function drain() {
    if (active || closed || !enabled || options.runWorker === false) return;
    active = true;
    try {
      while (!closed) {
        const job = (await store.list<Job>('pose3d-jobs')).filter(j => j.result.status === 'queued').sort((a, b) => a.createdAt.localeCompare(b.createdAt))[0];
        if (!job) break;
        // Serialize claim with cancel/retry; never run an already-cancelled snapshot.
        const claimed = await locked(job.analysisId, async () => {
          const current = await store.get<Job>('pose3d-jobs', job.id);
          if (current?.result.status !== 'queued') return false;
          job.result.status = 'running'; job.attempts++; await save(job); return true;
        });
        if (!claimed) continue;
        try {await run(job);} catch (error) {
          await locked(job.analysisId, async () => {
            const current = await store.get<Job>('pose3d-jobs', job.id);
            if (current?.result.status !== 'running') return;
            fail(job, (error as {errorCode?: string}).errorCode ?? 'invalid_result', (error as Error).message); await save(job);
          });
        } finally {cancelled.delete(job.id);}
      }
    } finally {active = false;}
  }
  for (const job of await store.list<Job>('pose3d-jobs')) {
    if (job.result.status === 'running') {fail(job, 'interrupted', '3D 子任务被中断，可单独重试'); await save(job);}
  }
  const timer = setInterval(() => {void drain().catch(console.error);}, 1000); timer.unref();
  const reply = (job: Job) => ({jobId: job.id, pose3d: job.result, error: job.error, attempts: job.attempts});
  app.get('/api/analyses/:id/pose3d', wrap(async (req, res) => {
    const id = String(req.params.id); await record(id); const job = await selected(id);
    res.json(job ? reply(job) : {status: 'disabled', pose3d: null, errorCode: enabled ? 'not_requested' : 'feature_disabled'});
  }));
  app.post('/api/analyses/:id/pose3d', wrap(async (req, res) => {
    const id = String(req.params.id);
    const job = await locked(id, async () => {
      const r = await record(id);
      if (r.status !== 'succeeded') throw poseError('analysis_not_succeeded', '须先完成二维分析', 409);
      if (!enabled) throw poseError('feature_disabled', 'AI 3D 尚未启用，现有视频与示意回放可继续使用', 503);
      const body = req.body ?? {}, scope = body.scope ?? 'swing-windows';
      if (!['swing-windows', 'manual', 'full'].includes(scope)) throw poseError('invalid_scope', '3D 处理范围无效');
      if (body.evaluationMode !== undefined || (scope !== 'manual' && body.windows !== undefined)) throw poseError('invalid_request', '用户接口不接受评测模式或非手动窗口');
      const c = await config();
      if (body.modelId !== undefined && body.modelId !== c.modelId) throw poseError('unsupported_model', '该 3D 模型尚未配置', 422);
      const input = await inputFiles(r), windows = selectPose3DWindows(r.analysis, input.timestamps, scope as Pose3DScope, body.windows, c);
      if (!c.weightsPath) throw poseError('model_unavailable', '未配置 3D 模型权重', 503);
      let weightsSha256: string;
      try {weightsSha256 = await fileDigest(c.weightsPath);} catch {throw poseError('model_unavailable', '3D 模型权重不可读取', 503);}
      const configHash = digest(canonical(c));
      const jobId = digest(canonical({analysisId: id, modelId: c.modelId, modelVersion: c.modelVersion, weightsSha256,
        configHash, scope, windows, viewGatePolicyVersion: c.viewGatePolicyVersion}));
      const previous = await selected(id);
      if (busy(previous) && previous!.id !== jobId) throw poseError('already_running', '已有 3D 子任务执行中，请先取消或等待完成', 409);
      let job = await store.get<Job>('pose3d-jobs', jobId);
      if (!job) {
        const blocked = windows.every(w => w.viewGate === 'blocked');
        const result: Pose3DResult = {status: blocked ? 'rejected_quality' : 'queued', source: 'ai-monocular-predicted', modelId: c.modelId,
          runtime: c.runtime, modelVersion: c.modelVersion, weightsSha256, pipelineVersion: c.pipelineVersion,
          viewGatePolicyVersion: c.viewGatePolicyVersion, scope, windows, frameCount: r.analysis.skeleton!.frameCount,
          pointSpec: 'cwb-65', coordinateSpace: 'camera-root-relative', axisConvention: 'x-right-y-up-z-forward', unit: 'model-relative', rootJoint: 'mid-hip',
          quality: {coverageRatio: 0, usableFrameRatio: 0, identityLockedRatio: 0, meanJointConfidence: 0, warnings: ['unvalidated_research_policy', ...(windows.some(w => w.viewGate === 'limited') ? ['view_limited'] : [])]},
          ...(blocked ? {errorCode: 'view_blocked'} : {})};
        job = {id: jobId, analysisId: id, createdAt: new Date().toISOString(), attempts: 0, result, config: c, configHash}; await save(job);
      }
      await store.put('pose3d-selections', id, {jobId}); return job;
    });
    res.status(busy(job) ? 202 : 200).json(reply(job)); void drain().catch(console.error);
  }));
  app.post('/api/analyses/:id/pose3d/retry', wrap(async (req, res) => {
    const id = String(req.params.id);
    const job = await locked(id, async () => {
      await record(id); if (!enabled) throw poseError('feature_disabled', 'AI 3D 尚未启用', 503);
      const job = await selected(id);
      if (!job) throw poseError('pose3d_not_found', '尚无 3D 子任务', 404);
      if (!['failed', 'rejected_quality'].includes(job.result.status)) throw poseError('not_retryable', '只有失败或质量拒绝的 3D 子任务可以重试', 409);
      if (childJobId === job.id || cancelled.has(job.id)) throw poseError('already_running', '正在回收 3D 进程，请稍后重试', 409);
      if (job.result.windows.every(w => w.viewGate === 'blocked')) throw poseError('view_blocked', '视角不确定，无法进行 3D 推理', 422);
      job.result.status = 'queued'; delete job.result.errorCode; job.error = undefined; await save(job); return job;
    }); res.status(202).json(reply(job)); void drain().catch(console.error);
  }));
  app.post('/api/analyses/:id/pose3d/cancel', wrap(async (req, res) => {
    const id = String(req.params.id);
    const job = await locked(id, async () => {
      await record(id); const job = await selected(id);
      if (!job) throw poseError('pose3d_not_found', '尚无 3D 子任务', 404);
      if (!busy(job)) throw poseError('not_cancellable', '3D 子任务已经结束', 409);
      if (job.result.status === 'running') cancelled.add(job.id);
      if (childJobId === job.id) child?.kill('SIGKILL');
      fail(job, 'cancelled', '3D 子任务已取消'); await save(job); return job;
    }); res.json(reply(job));
  }));
  app.get('/data/results/:id/pose3d/:jobId/pose3d.f32', wrap(async (req, res) => {
    const job = await store.get<Job>('pose3d-jobs', String(req.params.jobId));
    if (!job || job.analysisId !== req.params.id || job.result.status !== 'succeeded') throw poseError('artifact_not_found', '3D 结果不可用', 404);
    const file = await containedFile(directory(job), 'pose3d.f32', store.root);
    const bytes = await readFile(file); validatePose3DBuffer(bytes, job.result);
    res.type('application/octet-stream').send(bytes);
  }));
  // Do not expose request JSON, failed files, raw files, or symlinks via generic static serving.
  app.use('/data/results/:id/pose3d', (_req, res) => {res.status(404).json({error: '3D 产物不可访问', errorCode: 'artifact_not_found'});});
  // express.static decodes URLs after route matching. Protect encoded aliases
  // (e.g. %70ose3d and encoded separators) before the generic static middleware.
  app.use('/data/results', (req, res, next) => {
    let decoded: string;
    try {decoded = decodeURIComponent(req.path);} catch {res.status(400).json({error: '无效的文件路径', errorCode: 'invalid_artifact_path'}); return;}
    if (decoded.split(/[\\/]/).some(segment => segment.toLowerCase() === 'pose3d')) {
      res.status(404).json({error: '3D 产物不可访问', errorCode: 'artifact_not_found'}); return;
    }
    next();
  });
  void drain().catch(console.error);
  return {attach, close: () => {closed = true; clearInterval(timer); child?.kill('SIGKILL');}};
}
