/** Independent camera-relative prediction contract; never changes 2D metric semantics. */
export type Pose3DStatus = 'disabled' | 'queued' | 'running' | 'succeeded' | 'failed' | 'rejected_quality';
export type Pose3DScope = 'swing-windows' | 'manual' | 'full';
export type ViewClass = 'side' | 'oblique' | 'front' | 'rear' | 'unknown';
export type ViewGate = 'pass' | 'limited' | 'blocked';
export type Pose3DModelId = 'rtmw3d-rtmlib-x' | 'rtmw3d-mmpose-l' | 'motionbert-experimental';
export type Pose3DRuntime = 'rtmlib-onnx' | 'mmpose-pytorch' | 'motionbert-pytorch';
export interface Pose3DWindow {
  startFrame: number; endFrame: number;
  origin: 'auto-swing-event' | 'manual-annotation' | 'full-clip';
  viewClass: ViewClass; viewGate: ViewGate; usableFrameRatio: number;
}
export interface Pose3DResult {
  status: Pose3DStatus;
  source: 'ai-monocular-predicted';
  modelId: Pose3DModelId; runtime: Pose3DRuntime;
  pipelineVersion: string; modelVersion: string; weightsSha256: string;
  viewGatePolicyVersion: string;
  scope: Pose3DScope; windows: Pose3DWindow[]; frameCount: number;
  pointSpec: 'cwb-65'; coordinateSpace: 'camera-root-relative';
  axisConvention: 'x-right-y-up-z-forward'; unit: 'model-relative'; rootJoint: 'mid-hip';
  keypointsKey?: string;
  /** Optional immutable JSON sidecar; gateway must constrain paths before reading. */
  provenanceKey?: string;
  quality: {
    coverageRatio: number; usableFrameRatio: number; identityLockedRatio: number;
    meanJointConfidence: number; warnings: string[];
  };
  errorCode?: string;
}

export const POSE3D_POINT_COUNT = 65;
export const POSE3D_STRIDE = 4;
export const POSE3D_FRAME_BYTES = POSE3D_POINT_COUNT * POSE3D_STRIDE * 4;
export const POSE3D_MODEL_RUNTIMES: Readonly<Record<Pose3DModelId, Pose3DRuntime>> = {
  'rtmw3d-rtmlib-x': 'rtmlib-onnx',
  'rtmw3d-mmpose-l': 'mmpose-pytorch',
  'motionbert-experimental': 'motionbert-pytorch',
};

type Obj = Record<string, unknown>;
function fail(path: string, expected: string): never { throw new TypeError(`${path}: expected ${expected}`); }
function object(value: unknown, path: string): Obj {
  if(!value || typeof value!=='object' || Array.isArray(value)) fail(path,'object');
  return value as Obj;
}
function text(value: unknown, path: string): void {
  if(typeof value!=='string' || !value.trim()) fail(path,'nonempty string');
}
function member(value: unknown, values: readonly string[], path: string): void {
  if(typeof value!=='string' || !values.includes(value)) fail(path,values.join(' | '));
}
function integer(value: unknown, min: number, max: number, path: string): void {
  if(typeof value!=='number'||!Number.isSafeInteger(value)||value<min||value>max) fail(path,`safe integer in [${min}, ${max}]`);
}
function ratio(value: unknown, path: string): void {
  if(typeof value!=='number'||!Number.isFinite(value)||value<0||value>1) fail(path,'finite ratio in [0, 1]');
}

export function parsePose3DResult(value: unknown): Pose3DResult {
  const p=object(value,'pose3d');
  member(p.status,['disabled','queued','running','succeeded','failed','rejected_quality'],'pose3d.status');
  member(p.source,['ai-monocular-predicted'],'pose3d.source');
  member(p.modelId,Object.keys(POSE3D_MODEL_RUNTIMES),'pose3d.modelId');
  member(p.runtime,[POSE3D_MODEL_RUNTIMES[p.modelId as Pose3DModelId]],'pose3d.runtime');
  for(const key of ['pipelineVersion','modelVersion','viewGatePolicyVersion']) text(p[key],`pose3d.${key}`);
  if(typeof p.weightsSha256!=='string'||!/^[a-fA-F0-9]{64}$/.test(p.weightsSha256)) fail('pose3d.weightsSha256','SHA-256 hex digest');
  member(p.scope,['swing-windows','manual','full'],'pose3d.scope');
  integer(p.frameCount,1,Math.floor(Number.MAX_SAFE_INTEGER/POSE3D_FRAME_BYTES),'pose3d.frameCount');
  member(p.pointSpec,['cwb-65'],'pose3d.pointSpec');
  member(p.coordinateSpace,['camera-root-relative'],'pose3d.coordinateSpace');
  member(p.axisConvention,['x-right-y-up-z-forward'],'pose3d.axisConvention');
  member(p.unit,['model-relative'],'pose3d.unit');
  member(p.rootJoint,['mid-hip'],'pose3d.rootJoint');
  if(p.keypointsKey!==undefined) {
    text(p.keypointsKey,'pose3d.keypointsKey');
    if(p.status!=='succeeded') fail('pose3d.keypointsKey','absent until succeeded');
  } else if(p.status==='succeeded') fail('pose3d.keypointsKey','binary reference for succeeded result');
  if(p.provenanceKey!==undefined) text(p.provenanceKey,'pose3d.provenanceKey');
  if(p.errorCode!==undefined) {
    text(p.errorCode,'pose3d.errorCode');
    if(p.status==='succeeded') fail('pose3d.errorCode','absent for succeeded result');
  }
  if(['failed','rejected_quality'].includes(p.status as string)&&p.errorCode===undefined) fail('pose3d.errorCode','failure reason');
  if(!Array.isArray(p.windows)) fail('pose3d.windows','array');
  let previousEnd=-1;
  let eligibleFrames=0;
  for(const [i,value] of p.windows.entries()) {
    const path=`pose3d.windows[${i}]`, w=object(value,path);
    integer(w.startFrame,0,(p.frameCount as number)-1,`${path}.startFrame`);
    integer(w.endFrame,w.startFrame as number,(p.frameCount as number)-1,`${path}.endFrame`);
    if((w.startFrame as number)<=previousEnd) fail(path,'sorted nonoverlapping frame windows');
    previousEnd=w.endFrame as number;
    const origin=p.scope==='full'?'full-clip':p.scope==='manual'?'manual-annotation':'auto-swing-event';
    member(w.origin,[origin],`${path}.origin`);
    member(w.viewClass,['side','oblique','front','rear','unknown'],`${path}.viewClass`);
    member(w.viewGate,['pass','limited','blocked'],`${path}.viewGate`);
    if(w.viewClass==='unknown'&&w.viewGate!=='blocked') fail(`${path}.viewGate`,'blocked for unknown view');
    ratio(w.usableFrameRatio,`${path}.usableFrameRatio`);
    if(w.viewGate==='blocked'&&w.usableFrameRatio!==0) fail(`${path}.usableFrameRatio`,'0 for blocked windows');
    if(w.viewGate!=='blocked') eligibleFrames+=(w.endFrame as number)-(w.startFrame as number)+1;
  }
  if(p.scope==='full'&&p.windows.length>0) {
    const windows=p.windows as Pose3DWindow[];
    if(windows.length!==1||windows[0].startFrame!==0||windows[0].endFrame!==(p.frameCount as number)-1) fail('pose3d.windows','one full-clip window');
  }
  const q=object(p.quality,'pose3d.quality');
  for(const key of ['coverageRatio','usableFrameRatio','identityLockedRatio','meanJointConfidence']) ratio(q[key],`pose3d.quality.${key}`);
  if(!Array.isArray(q.warnings)) fail('pose3d.quality.warnings','array');
  q.warnings.forEach((v,i)=>text(v,`pose3d.quality.warnings[${i}]`));
  if((q.coverageRatio as number)>eligibleFrames/(p.frameCount as number)+1e-6) fail('pose3d.quality.coverageRatio','no greater than eligible window coverage');
  const coveredWindowRatio=eligibleFrames?(q.coverageRatio as number)*(p.frameCount as number)/eligibleFrames:0;
  if((q.usableFrameRatio as number)>coveredWindowRatio+1e-6) fail('pose3d.quality.usableFrameRatio','no greater than coverage within unblocked windows');
  if(p.status==='succeeded'&&(eligibleFrames===0||q.coverageRatio===0||q.usableFrameRatio===0)) fail('pose3d.status','usable data in at least one unblocked window before succeeded');
  return value as Pose3DResult;
}

export function validatePose3DResult(value: unknown): value is Pose3DResult {
  try {parsePose3DResult(value);return true;} catch {return false;}
}

/** Validate the decoded dense [T,65,4] tensor, including every gap/blocked frame. */
export function validatePose3DData(points: Float32Array, result: Pose3DResult): void {
  parsePose3DResult(result);
  if(points.length!==result.frameCount*POSE3D_POINT_COUNT*POSE3D_STRIDE) throw new RangeError('Pose3D binary size does not match frameCount');
  let windowIndex=0,coveredFrames=0;
  const windowCovered=new Array<number>(result.windows.length).fill(0);
  for(let f=0;f<result.frameCount;f++) {
    while(windowIndex<result.windows.length&&f>result.windows[windowIndex].endFrame) windowIndex++;
    const window=result.windows[windowIndex];
    const eligible=window&&f>=window.startFrame&&window.viewGate!=='blocked';
    let present=false;
    for(let point=0;point<POSE3D_POINT_COUNT;point++) {
      const n=(f*POSE3D_POINT_COUNT+point)*POSE3D_STRIDE;
      const [x,y,z,c]=[points[n],points[n+1],points[n+2],points[n+3]];
      if(!Number.isFinite(c)||c<0||c>1) throw new RangeError(`Invalid Pose3D confidence at frame ${f}, point ${point}`);
      if(c===0) {
        if(!Number.isNaN(x)||!Number.isNaN(y)||!Number.isNaN(z)) throw new RangeError(`Missing Pose3D point must be [NaN, NaN, NaN, 0] at frame ${f}, point ${point}`);
      } else {
        if(!Number.isFinite(x)||!Number.isFinite(y)||!Number.isFinite(z)) throw new RangeError(`Invalid Pose3D coordinate at frame ${f}, point ${point}`);
        if(!eligible) throw new RangeError(`Pose3D data outside an unblocked window at frame ${f}`);
        present=true;
      }
    }
    if(present) {
      // Match the Python boundary: root-relative frames require both hips and
      // their XYZ midpoint at zero, tolerating only Float32 rounding.
      const leftHip=(f*POSE3D_POINT_COUNT+11)*POSE3D_STRIDE;
      const rightHip=(f*POSE3D_POINT_COUNT+12)*POSE3D_STRIDE;
      if(points[leftHip+3]===0||points[rightHip+3]===0) throw new RangeError(`Pose3D root missing at frame ${f}`);
      for(let axis=0;axis<3;axis++) {
        if(Math.abs((points[leftHip+axis]+points[rightHip+axis])/2)>1e-5) throw new RangeError(`Pose3D root not centered at frame ${f}`);
      }
      coveredFrames++;windowCovered[windowIndex]++;
    }
  }
  if(Math.abs(coveredFrames/result.frameCount-result.quality.coverageRatio)>1e-6) throw new RangeError('Pose3D coverageRatio does not match binary data');
  for(const [i,window] of result.windows.entries()) {
    if(window.usableFrameRatio>windowCovered[i]/(window.endFrame-window.startFrame+1)+1e-6) throw new RangeError(`Pose3D usableFrameRatio exceeds frames with data in window ${i}`);
  }
}

/** Decode explicitly as little-endian, including unaligned Node Buffer subviews. */
export function validatePose3DBuffer(buffer: ArrayBuffer | ArrayBufferView, result: Pose3DResult): Float32Array {
  parsePose3DResult(result);
  if(buffer.byteLength!==result.frameCount*POSE3D_FRAME_BYTES) throw new RangeError('Pose3D binary size does not match frameCount');
  const view=ArrayBuffer.isView(buffer)
    ?new DataView(buffer.buffer,buffer.byteOffset,buffer.byteLength)
    :new DataView(buffer);
  const points=new Float32Array(view.byteLength/4);
  for(let i=0;i<points.length;i++) points[i]=view.getFloat32(i*4,true);
  validatePose3DData(points,result);
  return points;
}
