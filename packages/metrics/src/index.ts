import { CWB_65 as K, validateSkeletonData, type Analysis, type MetricId, type MetricResult, type SwingEvent, type VideoAsset } from '../../schema/src/index.ts';
export * from './biomechanics.ts';

/** Increment whenever a formula, confidence policy, or phase definition changes. */
export const DEFINITION_VERSION = '0.1.0';
type Point = { x: number; y: number; confidence: number };
type Reason = NonNullable<MetricResult['suppressed']>['reason'];
type Pose = (f: number, k: number) => Point;
const torso = [K.leftShoulder, K.rightShoulder, K.leftHip, K.rightHip];
const midpoint = (a: Point,b: Point): Point => ({x:(a.x+b.x)/2,y:(a.y+b.y)/2,confidence:Math.min(a.confidence,b.confidence)});
const distance = (a: Point,b: Point) => Math.hypot(a.x-b.x,a.y-b.y);
const clamp = (v: number) => Math.min(1,Math.max(0,v));
const valid = (p: Point) => Number.isFinite(p.x)&&Number.isFinite(p.y)&&p.confidence>=0.5;
function poseReader(video: VideoAsset, data: Float32Array): Pose {
  // Normalized image coordinates are anisotropic. Convert x into image-height units
  // before angle/distance calculations; otherwise landscape videos distort angles.
  return (f,k) => {
    const i=(f*K.count+k)*K.stride;
    return {x:data[i]*(video.width/video.height),y:data[i+1],confidence:data[i+2]};
  };
}
function arm(swing: SwingEvent): number[] {
  return swing.handedness==='left' ? [K.leftShoulder,K.leftElbow,K.leftWrist] : [K.rightShoulder,K.rightElbow,K.rightWrist];
}
function angle(a: Point,b: Point,c: Point): number | null {
  if(![a,b,c].every(valid)) return null;
  const ab=distance(a,b),bc=distance(b,c); if(ab<1e-6||bc<1e-6) return null;
  const cos=((a.x-b.x)*(c.x-b.x)+(a.y-b.y)*(c.y-b.y))/(ab*bc);
  return Math.acos(Math.min(1,Math.max(-1,cos)))*180/Math.PI;
}
function windowConfidence(timestamps: Float32Array, pose: Pose, impact: number, joints: number[]) {
  const t=timestamps[impact]; let sum=0,count=0;
  for(let f=0;f<timestamps.length;f++) {
    if(Math.abs(timestamps[f]-t)>0.100001) continue;
    for(const k of joints) {const p=pose(f,k); sum+=Number.isFinite(p.x)&&Number.isFinite(p.y) ? p.confidence : 0; count++;}
  }
  return count ? sum/count : 0;
}
/** Infer image-left/right facing only when visible foot/face evidence agrees.
 * Torso midpoints provide a stable center and scale; torso lean itself is not a heading.
 */
function facing(pose: Pose, f: number, scale: number, shoulders: Point, hips: Point): {direction: number; confidence: number} | null {
  const votes: {direction:number;confidence:number}[]=[];
  for(const [toe1,toe2,heel] of [[K.leftBigToe,K.leftSmallToe,K.leftHeel],[K.rightBigToe,K.rightSmallToe,K.rightHeel]]) {
    const a=pose(f,toe1), b=pose(f,toe2), c=pose(f,heel);
    if(![a,b,c].every(valid)) continue;
    const dx=(a.x+b.x)/2-c.x;
    if(Math.abs(dx)>scale*0.04) votes.push({direction:Math.sign(dx),confidence:Math.min(a.confidence,b.confidence,c.confidence)});
  }
  const nose=pose(f,K.nose);
  const centerX=shoulders.x*0.8+hips.x*0.2;
  if(valid(nose)&&Math.abs(nose.x-centerX)>scale*0.12) votes.push({direction:Math.sign(nose.x-centerX),confidence:nose.confidence});
  // Require two independent hints, so one obscured foot cannot reverse the sign.
  if(votes.length<2 || votes.some(v=>v.direction!==votes[0].direction)) return null;
  return {direction:votes[0].direction,confidence:Math.min(...votes.map(v=>v.confidence))};
}

export function calculateMetrics(video: VideoAsset, analysis: Analysis, timestamps: Float32Array, keypoints: Float32Array): MetricResult[] {
  validateSkeletonData(timestamps,keypoints,video.frameCount);
  if(analysis.skeleton && analysis.skeleton.frameCount!==video.frameCount) throw new RangeError('Video and skeleton frame counts differ');
  const pose=poseReader(video,keypoints), results: MetricResult[]=[];
  for(const swing of analysis.swings) {
    const f=swing.impact.frame;
    if(!Number.isInteger(f)||f<0||f>=timestamps.length) throw new RangeError('Impact frame outside skeleton');
    // Anchor all calculations in captured frame timestamps, never playback FPS or media time.
    const localInterval=f>0 ? timestamps[f]-timestamps[f-1] : timestamps.length>1 ? timestamps[1]-timestamps[0] : 1/video.captureFps;
    if(Math.abs(timestamps[f]-swing.impact.tSec)>Math.max(0.001,localInterval*0.6)) throw new RangeError('Impact tSec disagrees with capture timestamp');
    const joints=arm(swing), shoulder=pose(f,joints[0]),elbow=pose(f,joints[1]),wrist=pose(f,joints[2]);
    const sh=midpoint(pose(f,K.leftShoulder),pose(f,K.rightShoulder));
    const hip=midpoint(pose(f,K.leftHip),pose(f,K.rightHip));
    const L=distance(sh,hip), quality=analysis.quality;
    const emit=(id: MetricId,value: number|null,required: number[],allowedViews?: string[],extraConfidence=1, forced?: Reason) => {
      const temporal=id==='accel_duration'||id==='followthrough_duration';
      const average=windowConfidence(timestamps,pose,f,required);
      const available=required.every(k=>valid(pose(f,k)));
      const view=quality?.view ?? 'unknown';
      const viewBound=temporal ? 1 : (quality?.viewConfidence ?? 0)*(id==='elbow_angle_impact'&&['front','back'].includes(view) ? 0.85 : 1);
      // These scores are uncalibrated reliability bounds, not independent
      // probabilities. Multiplying them would always hide visual-only results.
      const confidence=clamp(Math.min(average,swing.impact.confidence,extraConfidence,viewBound));
      let reason=forced;
      // Nominal 60 FPS includes the common 60000/1001 capture rate.
      if(temporal && (video.captureFps<59 || quality?.fpsTier==='low')) reason='low_fps';
      else if(allowedViews && (!allowedViews.includes(view)||(quality?.viewConfidence??0)<0.5)) reason='bad_view';
      else if(!reason && (swing.handedness==='unknown'||average<0.5||!available||confidence<0.5||value===null||!Number.isFinite(value))) reason='low_conf';
      const metric: MetricResult={id,swingId:swing.id,value:reason ? null : value,unit:id==='elbow_angle_impact'?'deg':temporal?'sec':'ratio',confidence,definitionVersion:DEFINITION_VERSION};
      if(reason) metric.suppressed={reason};
      results.push(metric);
    };
    emit('elbow_angle_impact',angle(shoulder,elbow,wrist),joints,['side','oblique','front','back']);
    const bodyValid=valid(sh)&&valid(hip)&&L>1e-5;
    emit('impact_height_ratio',bodyValid ? (hip.y-wrist.y)/L : null,[...torso,joints[2]],['side','oblique','back']);
    const heading=bodyValid ? facing(pose,f,L,sh,hip) : null;
    const facingRequired=[...torso,joints[2],...[K.leftBigToe,K.leftSmallToe,K.leftHeel,K.rightBigToe,K.rightSmallToe,K.rightHeel].filter(k=>valid(pose(f,k)))];
    // Optional feet can be occluded; include only the visible points actually used.
    emit('impact_forward_offset',heading ? (wrist.x-shoulder.x)*heading.direction/L : null,facingRequired,['side','oblique'],heading?.confidence??0);
    const phaseDuration=(start: number|undefined,end: number|undefined): number|null => {
      if(start===undefined||end===undefined||!Number.isInteger(start)||!Number.isInteger(end)||start<0||end>=timestamps.length||end<start) return null;
      return timestamps[end]-timestamps[start];
    };
    emit('accel_duration',phaseDuration(swing.phases.backswingStartFrame,f),joints);
    emit('followthrough_duration',phaseDuration(f,swing.phases.followThroughEndFrame),joints);
  }
  return results;
}

export interface AngleSample {frame: number; tSec: number; value: number|null; confidence: number}
export function elbowAngleSeries(video: VideoAsset, swing: SwingEvent, timestamps: Float32Array, keypoints: Float32Array): AngleSample[] {
  validateSkeletonData(timestamps,keypoints,video.frameCount);
  const pose=poseReader(video,keypoints),joints=arm(swing);
  return Array.from(timestamps,(tSec,frame)=>{
    const points=joints.map(k=>pose(frame,k));
    return {frame,tSec,value:swing.handedness==='unknown'?null:angle(points[0],points[1],points[2]),confidence:swing.handedness==='unknown'?0:Math.min(...points.map(p=>p.confidence))};
  });
}
