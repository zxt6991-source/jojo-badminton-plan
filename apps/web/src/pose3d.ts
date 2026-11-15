import {CWB_65 as K, type VideoAsset} from '@badminton/schema';

export type Vec3 = [number, number, number];
export interface Pose3DPoint {
  position: Vec3;
  confidence: number;
  visible: boolean;
}
export interface Pose3DFrame {
  joints: Pose3DPoint[];
  /** A display conversion of 2D observations, never a measured 3D reconstruction. */
  mode: 'schematic-2d';
  /** The clip's fixed reference hip midpoint maps to this display-space origin. */
  origin: Vec3;
  /** Display units per source image-height unit. Constant for the entire clip. */
  scale: number;
  visibleCount: number;
}
export interface Pose3DSequence {
  frame(index: number): Pose3DFrame;
  floorY: number;
  center: Vec3;
  /** Approximate display envelope for camera framing, not stature in meters. */
  height: number;
}

const MIN_CONFIDENCE = 0.5;
const DISPLAY_TORSO_LENGTH = 0.56;
const DEFAULT_SCALE = 2.2;
const DEPTH_SEPARATION = 0.1;
const FOOT_CLEARANCE = 0.08;
const torsoIndices = [K.leftShoulder, K.rightShoulder, K.leftHip, K.rightHip];
const leftIndices = new Set([K.leftEye, K.leftEar, K.leftShoulder, K.leftElbow,
  K.leftWrist, K.leftHip, K.leftKnee, K.leftAnkle, K.leftBigToe, K.leftSmallToe,
  K.leftHeel, ...Array.from({length:21}, (_, index) => K.leftHandStart + index)]);
const rightIndices = new Set([K.rightEye, K.rightEar, K.rightShoulder, K.rightElbow,
  K.rightWrist, K.rightHip, K.rightKnee, K.rightAnkle, K.rightBigToe, K.rightSmallToe,
  K.rightHeel, ...Array.from({length:21}, (_, index) => K.rightHandStart + index)]);

type Point2D = {x:number; y:number; confidence:number; visible:boolean};
function quantile(values:number[], fraction:number):number {
  const sorted = [...values].sort((a,b) => a-b);
  const at = (sorted.length-1)*fraction, lower = Math.floor(at), upper = Math.ceil(at);
  return sorted[lower] + (sorted[upper]-sorted[lower])*(at-lower);
}
function median(values:number[], fallback:number):number {
  return values.length ? quantile(values, 0.5) : fallback;
}
function depthFor(index:number):number {
  // Fixed display spacing separates corresponding limbs while orbiting. These
  // depths do not depend on the pose, visibility, phase, or inferred body turns.
  const depth = index < K.leftShoulder ? DEPTH_SEPARATION*0.4 : DEPTH_SEPARATION;
  return leftIndices.has(index) ? -depth : rightIndices.has(index) ? depth : 0;
}

/**
 * Build an explicitly schematic 3D display sequence from cwb-65 image points.
 *
 * XY retains the video's projected geometry after aspect correction and y-axis
 * inversion. A bounded scale and fixed hip reference are estimated from at most
 * 100 evenly distributed frames. The reference never follows individual frames,
 * so changing pose or temporary occlusion cannot make the body resize or recenter.
 * Z is a constant illustration offset for anatomical left/right, not recovered
 * depth. None of these coordinates may be used to calculate physical 3D metrics.
 *
 * Missing and low-confidence joints have visible=false and NaN coordinates;
 * consumers must check visible instead of drawing them at the origin or keeping
 * a previous frame's position. Source arrays are read only and remain untouched.
 */
export function createPose3DSequence(video:VideoAsset, points:Float32Array):Pose3DSequence {
  const {width,height:videoHeight,frameCount} = video;
  if (!Number.isFinite(width) || !Number.isFinite(videoHeight) || width<=0 || videoHeight<=0)
    throw new RangeError('3D display requires positive finite video dimensions');
  if (!Number.isSafeInteger(frameCount) || frameCount<=0 || points.length!==frameCount*K.count*K.stride)
    throw new RangeError('3D display binary length does not match cwb-65 frame count');
  const aspect = width/videoHeight;
  const read = (frame:number,index:number):Point2D => {
    const at = (frame*K.count+index)*K.stride;
    const x=points[at], y=points[at+1], c=points[at+2];
    const confidence = Number.isFinite(c)&&c>=0&&c<=1 ? c : 0;
    const visible = confidence>=MIN_CONFIDENCE && Number.isFinite(x) && Number.isFinite(y)
      && x>=0 && x<=1 && y>=0 && y<=1;
    return {x:x*aspect,y,confidence,visible};
  };

  const sampleCount=Math.min(100,frameCount);
  const sampledFrames=Array.from({length:sampleCount},(_,i)=>sampleCount===1 ? 0 : Math.round(i*(frameCount-1)/(sampleCount-1)));
  const hipXs:number[]=[],hipYs:number[]=[],torsoLengths:number[]=[],ankleYs:number[]=[];
  const observedBody:Point2D[]=[];
  for (const index of sampledFrames) {
    const p=torsoIndices.map(joint=>read(index,joint));
    if (p[2].visible&&p[3].visible) {
      hipXs.push((p[2].x+p[3].x)/2);hipYs.push((p[2].y+p[3].y)/2);
    }
    if (p.every(joint=>joint.visible)) {
      const dx=(p[0].x+p[1].x-p[2].x-p[3].x)/2;
      const dy=(p[0].y+p[1].y-p[2].y-p[3].y)/2;
      const length=Math.hypot(dx,dy);
      // Near-coincident shoulder/hip centers are not a usable body-size ruler.
      if (length>=0.025) torsoLengths.push(length);
    }
    for (const joint of [K.leftAnkle,K.rightAnkle]) {
      const ankle=read(index,joint);if(ankle.visible)ankleYs.push(ankle.y);
    }
    for(let joint=0;joint<K.leftHandStart;joint++) {
      const point=read(index,joint);if(point.visible)observedBody.push(point);
    }
  }

  const referenceX=median(hipXs,aspect*0.5),referenceY=median(hipYs,0.55);
  const scale=torsoLengths.length ? Math.max(0.5,Math.min(8,DISPLAY_TORSO_LENGTH/median(torsoLengths,1))) : DEFAULT_SCALE;
  const floorY=ankleYs.length ? (referenceY-quantile(ankleYs,0.9))*scale-FOOT_CLEARANCE : -0.85;
  // The camera envelope is fixed per sequence. Include projected hands/wrists in
  // body17 and feet, but do not let many duplicate hand landmarks bias centering.
  const xs=observedBody.map(p=>(p.x-referenceX)*scale);
  const ys=observedBody.map(p=>(referenceY-p.y)*scale);
  const minX=xs.length?Math.min(...xs):-0.5,maxX=xs.length?Math.max(...xs):0.5;
  const top=ys.length?Math.max(...ys)+0.16:0.95;
  const bottom=ys.length?Math.min(floorY,Math.min(...ys)-0.06):floorY;
  const envelopeHeight=Math.max(1.2,top-bottom,(maxX-minX)*0.85);
  const center:Vec3=[(minX+maxX)/2,(top+bottom)/2,0];

  return {
    floorY,center,height:envelopeHeight,
    frame(index:number):Pose3DFrame {
      if(!Number.isInteger(index)||index<0||index>=frameCount)
        throw new RangeError('3D display frame index is outside the sequence');
      let visibleCount=0;
      const joints=Array.from({length:K.count},(_,joint):Pose3DPoint=>{
        const p=read(index,joint);
        if(!p.visible)return{position:[NaN,NaN,NaN],confidence:p.confidence,visible:false};
        visibleCount++;
        return {position:[(p.x-referenceX)*scale,(referenceY-p.y)*scale,depthFor(joint)],confidence:p.confidence,visible:true};
      });
      return{joints,mode:'schematic-2d',origin:[0,0,0],scale,visibleCount};
    },
  };
}
