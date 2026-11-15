/** Frozen P0a transport contract. Breaking changes require a new schemaVersion. */
export const SCHEMA_VERSION = '0.1.0' as const;
export type ViewAngle = 'side' | 'back' | 'front' | 'oblique' | 'unknown';
export type SwingType = 'smash' | 'clear' | 'drop' | 'drive' | 'net' | 'unknown';
export type PoseSource = 'mono-2d' | 'mono-3d' | 'multi-view' | 'imu-fused';
export interface VideoAsset {
  id: string; storageKey: string; proxyKey?: string; width: number; height: number;
  captureFps: number; frameCount: number; durationSec: number; hasAudio: boolean;
  timeRemapped: boolean; device?: {make?: string; model?: string; os?: string};
}
export interface Analysis {
  id: string; videoId: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'rejected_quality';
  schemaVersion: string; pipelineVersion: string;
  userInput: {swingType?: SwingType; targetPoint?: [number, number]};
  target?: {trackId: number; lockedBy: 'auto' | 'user'};
  quality?: QualityReport; skeleton?: SkeletonRef;
  swings: SwingEvent[]; metrics: MetricResult[]; diagnoses: Diagnosis[];
  stageTimingsMs: Record<string, number>;
}
export interface QualityReport {
  fpsTier: 'high' | 'mid' | 'low'; view: ViewAngle; viewConfidence: number;
  bodyVisibleRatio: number; lighting: 'ok' | 'dark' | 'backlit'; multiPerson: boolean;
  issues: {code: string; severity: 'info' | 'warn' | 'block'}[]; usable: boolean;
}
export interface SkeletonRef {
  spec: 'cwb-65'; source: PoseSource; coordSpace: 'image-normalized'; frameCount: number;
  timestampsKey: string; keypointsKey: string;
  smoothing: {method: 'one-euro'; params: Record<string, number>}; keypoints3dKey?: string;
}
export interface SwingEvent {
  id: string; type: SwingType; typeConfidence: number;
  handedness: 'right' | 'left' | 'unknown';
  impact: {frame: number; tSec: number; confidence: number; evidence: ('audio' | 'wrist-peak' | 'manual')[]};
  phases: {backswingStartFrame?: number; impactFrame: number; followThroughEndFrame?: number};
}
export type MetricId = 'elbow_angle_impact' | 'impact_height_ratio' | 'impact_forward_offset' | 'accel_duration' | 'followthrough_duration';
export interface MetricResult {
  id: MetricId; swingId: string; value: number | null; unit: 'deg' | 'ratio' | 'sec'; confidence: number;
  suppressed?: {reason: 'low_fps' | 'bad_view' | 'low_conf'};
  seriesKey?: string; definitionVersion: string;
}
export interface Diagnosis {
  code: string; swingId: string; severity: 'info' | 'warn' | 'major'; evidenceMetricIds: string[]; confidence: number;
}

/** COCO WholeBody: body 0..16, feet 17..22, face omitted, hands 23..64. */
export const CWB_65 = {
  nose: 0, leftEye: 1, rightEye: 2, leftEar: 3, rightEar: 4,
  leftShoulder: 5, rightShoulder: 6, leftElbow: 7, rightElbow: 8,
  leftWrist: 9, rightWrist: 10, leftHip: 11, rightHip: 12,
  leftKnee: 13, rightKnee: 14, leftAnkle: 15, rightAnkle: 16,
  leftBigToe: 17, leftSmallToe: 18, leftHeel: 19,
  rightBigToe: 20, rightSmallToe: 21, rightHeel: 22,
  leftHandStart: 23, rightHandStart: 44, count: 65, stride: 3,
} as const;
export const KEYPOINT_COUNT = 65;
export const KEYPOINT_STRIDE = 3;

type Obj = Record<string, unknown>;
function fail(path: string, expected: string): never { throw new TypeError(`${path}: expected ${expected}`); }
function obj(v: unknown, path: string): Obj {
  if (!v || typeof v !== 'object' || Array.isArray(v)) fail(path, 'object');
  return v as Obj;
}
function str(v: unknown, path: string) { if (typeof v !== 'string') fail(path, 'string'); }
function num(v: unknown, path: string, min = -Infinity, max = Infinity, integer = false) {
  if (typeof v !== 'number' || !Number.isFinite(v) || v < min || v > max || (integer && !Number.isInteger(v))) fail(path, `finite ${integer ? 'integer' : 'number'} in [${min}, ${max}]`);
}
function bool(v: unknown, path: string) { if (typeof v !== 'boolean') fail(path, 'boolean'); }
function one(v: unknown, values: readonly string[], path: string) { if (typeof v !== 'string' || !values.includes(v)) fail(path, values.join(' | ')); }
function array(v: unknown, path: string, check: (x: unknown, p: string) => void) {
  if (!Array.isArray(v)) fail(path, 'array');
  v.forEach((x, i) => check(x, `${path}[${i}]`));
}
function optional(o: Obj, key: string, path: string, check: (v: unknown, p: string) => void) { if (o[key] !== undefined) check(o[key], `${path}.${key}`); }
function numbers(v: unknown, path: string, min = -Infinity) { Object.entries(obj(v,path)).forEach(([key,value]) => num(value,`${path}.${key}`,min)); }
const swingTypes = ['smash','clear','drop','drive','net','unknown'];
const conf = (v: unknown, p: string) => num(v,p,0,1);
const frame = (v: unknown, p: string) => num(v,p,0,Infinity,true);

export function parseVideoAsset(value: unknown): VideoAsset {
  const v = obj(value,'video');
  str(v.id,'video.id'); str(v.storageKey,'video.storageKey'); optional(v,'proxyKey','video',str);
  ['width','height','frameCount'].forEach(k => num(v[k],`video.${k}`,1,Infinity,true));
  num(v.captureFps,'video.captureFps',Number.EPSILON); num(v.durationSec,'video.durationSec',Number.EPSILON);
  bool(v.hasAudio,'video.hasAudio'); bool(v.timeRemapped,'video.timeRemapped');
  optional(v,'device','video',(d,p) => { const device=obj(d,p); ['make','model','os'].forEach(k=>optional(device,k,p,str)); });
  return value as VideoAsset;
}
export function validateVideoAsset(value: unknown): value is VideoAsset { try {parseVideoAsset(value); return true;} catch {return false;} }

export function parseAnalysis(value: unknown): Analysis {
  const a = obj(value,'analysis');
  ['id','videoId','pipelineVersion'].forEach(k=>str(a[k],`analysis.${k}`));
  one(a.schemaVersion,[SCHEMA_VERSION],'analysis.schemaVersion');
  one(a.status,['queued','running','succeeded','failed','rejected_quality'],'analysis.status');
  const input = obj(a.userInput,'analysis.userInput');
  optional(input,'swingType','analysis.userInput',(v,p)=>one(v,swingTypes,p));
  optional(input,'targetPoint','analysis.userInput',(v,p)=>{if (!Array.isArray(v)||v.length!==2) fail(p,'[x, y]'); v.forEach((n,i)=>conf(n,`${p}[${i}]`));});
  optional(a,'target','analysis',(v,p)=>{const t=obj(v,p); frame(t.trackId,`${p}.trackId`); one(t.lockedBy,['auto','user'],`${p}.lockedBy`);});
  optional(a,'quality','analysis',(v,p)=>{
    const q=obj(v,p); one(q.fpsTier,['high','mid','low'],`${p}.fpsTier`); one(q.view,['side','back','front','oblique','unknown'],`${p}.view`);
    conf(q.viewConfidence,`${p}.viewConfidence`); conf(q.bodyVisibleRatio,`${p}.bodyVisibleRatio`);
    one(q.lighting,['ok','dark','backlit'],`${p}.lighting`); bool(q.multiPerson,`${p}.multiPerson`); bool(q.usable,`${p}.usable`);
    array(q.issues,`${p}.issues`,(v,p)=>{const issue=obj(v,p); str(issue.code,`${p}.code`); one(issue.severity,['info','warn','block'],`${p}.severity`);});
    if(q.usable && (q.issues as Obj[]).some(i=>i.severity==='block')) fail(`${p}.usable`,'false when a blocking issue exists');
  });
  optional(a,'skeleton','analysis',(v,p)=>{
    const s=obj(v,p); one(s.spec,['cwb-65'],`${p}.spec`); one(s.source,['mono-2d','mono-3d','multi-view','imu-fused'],`${p}.source`);
    one(s.coordSpace,['image-normalized'],`${p}.coordSpace`); num(s.frameCount,`${p}.frameCount`,1,Infinity,true);
    str(s.timestampsKey,`${p}.timestampsKey`); str(s.keypointsKey,`${p}.keypointsKey`); optional(s,'keypoints3dKey',p,str);
    const smooth=obj(s.smoothing,`${p}.smoothing`); one(smooth.method,['one-euro'],`${p}.smoothing.method`); numbers(smooth.params,`${p}.smoothing.params`);
  });
  const ids=new Set<string>();
  array(a.swings,'analysis.swings',(v,p)=>{
    const s=obj(v,p); str(s.id,`${p}.id`); if(ids.has(s.id as string)) fail(`${p}.id`,'unique swing ID'); ids.add(s.id as string);
    one(s.type,swingTypes,`${p}.type`); conf(s.typeConfidence,`${p}.typeConfidence`); one(s.handedness,['right','left','unknown'],`${p}.handedness`);
    const i=obj(s.impact,`${p}.impact`); frame(i.frame,`${p}.impact.frame`); num(i.tSec,`${p}.impact.tSec`,0); conf(i.confidence,`${p}.impact.confidence`);
    array(i.evidence,`${p}.impact.evidence`,(v,p)=>one(v,['audio','wrist-peak','manual'],p));
    const ph=obj(s.phases,`${p}.phases`); frame(ph.impactFrame,`${p}.phases.impactFrame`);
    if(ph.impactFrame!==i.frame) fail(`${p}.phases.impactFrame`,'same frame as impact.frame');
    optional(ph,'backswingStartFrame',`${p}.phases`,(v,p)=>num(v,p,0,i.frame as number,true));
    optional(ph,'followThroughEndFrame',`${p}.phases`,(v,p)=>num(v,p,i.frame as number,Infinity,true));
    if(a.skeleton && [i.frame,ph.backswingStartFrame,ph.followThroughEndFrame].some(f=>f!==undefined&&(f as number)>=(a.skeleton as SkeletonRef).frameCount)) fail(p,'frames inside skeleton');
  });
  array(a.metrics,'analysis.metrics',(v,p)=>{
    const m=obj(v,p); one(m.id,['elbow_angle_impact','impact_height_ratio','impact_forward_offset','accel_duration','followthrough_duration'],`${p}.id`);
    str(m.swingId,`${p}.swingId`); if(!ids.has(m.swingId as string)) fail(`${p}.swingId`,'existing swing ID');
    if(m.value!==null) num(m.value,`${p}.value`);
    one(m.unit,['deg','ratio','sec'],`${p}.unit`); conf(m.confidence,`${p}.confidence`); str(m.definitionVersion,`${p}.definitionVersion`);
    optional(m,'seriesKey',p,str); optional(m,'suppressed',p,(v,p)=>{const s=obj(v,p); one(s.reason,['low_fps','bad_view','low_conf'],`${p}.reason`); if(m.value!==null) fail(p,'value null for suppressed metric');});
  });
  array(a.diagnoses,'analysis.diagnoses',(v,p)=>{const d=obj(v,p); str(d.code,`${p}.code`); str(d.swingId,`${p}.swingId`); if(!ids.has(d.swingId as string)) fail(`${p}.swingId`,'existing swing ID'); one(d.severity,['info','warn','major'],`${p}.severity`); array(d.evidenceMetricIds,`${p}.evidenceMetricIds`,str); conf(d.confidence,`${p}.confidence`);});
  numbers(a.stageTimingsMs,'analysis.stageTimingsMs',0);
  return value as Analysis;
}
export function validateAnalysis(value: unknown): value is Analysis { try {parseAnalysis(value); return true;} catch {return false;} }

/** Reject corrupt files at the language boundary; absent points must be NaN,NaN,0. */
export function validateSkeletonData(timestamps: Float32Array, keypoints: Float32Array, frameCount: number): void {
  if(timestamps.length!==frameCount || keypoints.length!==frameCount*65*3) throw new RangeError('Skeleton binary size does not match frameCount');
  for(let f=0;f<frameCount;f++) {
    const t=timestamps[f]; if(!Number.isFinite(t)||t<0||(f>0&&t<=timestamps[f-1])) throw new RangeError(`Timestamp ${f} must increase in capture time`);
  }
  for(let n=0;n<keypoints.length;n+=3) {
    const x=keypoints[n],y=keypoints[n+1],c=keypoints[n+2];
    if(!Number.isFinite(c)||c<0||c>1) throw new RangeError(`Invalid confidence at point ${n/3}`);
    if(c===0 && Number.isNaN(x) && Number.isNaN(y)) continue;
    if(!Number.isFinite(x)||!Number.isFinite(y)||x<0||x>1||y<0||y>1) throw new RangeError(`Invalid normalized point ${n/3}`);
  }
}
