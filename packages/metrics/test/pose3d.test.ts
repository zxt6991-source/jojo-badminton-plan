import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createPose3DSequence,createReplaySampler} from '../../../apps/web/src/pose3d.ts';
import {CWB_65 as K,type VideoAsset} from '../../schema/src/index.ts';

function fixture(frameCount=3) {
  const video:VideoAsset={id:'test',storageKey:'',width:200,height:100,captureFps:120,
    frameCount,durationSec:frameCount/120,hasAudio:false,timeRemapped:false};
  const points=new Float32Array(frameCount*65*3);
  for(let i=0;i<points.length;i+=3){points[i]=NaN;points[i+1]=NaN;}
  const pose:Record<number,number[]>={
    0:[.5,.2],5:[.45,.3],6:[.55,.3],7:[.38,.4],8:[.62,.4],9:[.32,.5],10:[.68,.5],
    11:[.45,.6],12:[.55,.6],13:[.44,.75],14:[.56,.75],15:[.43,.9],16:[.57,.9],
    17:[.45,.92],18:[.46,.92],19:[.42,.92],20:[.6,.92],21:[.61,.92],22:[.56,.92],
    23:[.32,.5],44:[.68,.5],64:[.7,.52],
  };
  const set=(frame:number,joint:number,x:number,y:number,c=.9)=>points.set([x,y,c],frame*195+joint*3);
  for(let f=0;f<frameCount;f++)for(const [joint,p] of Object.entries(pose))set(f,Number(joint),p[0],p[1]);
  return{video,points,set};
}
const near=(a:number,b:number,tolerance=1e-6)=>assert.ok(Math.abs(a-b)<tolerance,`${a} != ${b}`);

test('schematic display preserves aspect-corrected XY, reverses y and uses stable display depth',()=>{
  const f=fixture(),sequence=createPose3DSequence(f.video,f.points),frame=sequence.frame(0);
  const shoulder=frame.joints[6].position,elbow=frame.joints[8].position;
  near((elbow[0]-shoulder[0])/(elbow[1]-shoulder[1]),-1.4);
  assert.ok(shoulder[1]>frame.joints[12].position[1]);
  near(frame.joints[5].position[2],-.1);near(frame.joints[6].position[2],.1);
  near(frame.joints[23].position[2],-.1);near(frame.joints[44].position[2],.1);
  assert.equal(frame.joints.length,65);assert.equal(frame.mode,'schematic-2d');
  assert.deepEqual(frame.origin,[0,0,0]);
  assert.ok(sequence.height>1.2&&sequence.height<2.5);
});

test('pose changes retain one clip-wide scale, hip reference, floor and camera center',()=>{
  const f=fixture();
  for(const k of [5,6,11,12]){const at=195+k*3;f.set(1,k,f.points[at]+.1,f.points[at+1]-.05);}
  f.set(1,10,.9,.1); // raise hand without changing body-size calibration
  const sequence=createPose3DSequence(f.video,f.points),a=sequence.frame(0),b=sequence.frame(1);
  assert.equal(a.scale,b.scale);
  near(b.joints[12].position[0]-a.joints[12].position[0],.2*a.scale);
  near(b.joints[12].position[1]-a.joints[12].position[1],.05*a.scale);
  const floor=sequence.floorY,center=[...sequence.center];sequence.frame(2);
  assert.equal(sequence.floorY,floor);assert.deepEqual(sequence.center,center);
  assert.deepEqual(createPose3DSequence(f.video,f.points).frame(1),b);
});

test('occluded and missing points are hidden rather than moved to zero or retained from an earlier frame',()=>{
  const f=fixture();f.set(1,10,.68,.5,.49);f.set(1,9,NaN,NaN,0);
  const sequence=createPose3DSequence(f.video,f.points);
  assert.equal(sequence.frame(0).joints[10].visible,true);
  for(const k of [9,10]){
    const p=sequence.frame(1).joints[k];assert.equal(p.visible,false);assert.ok(p.position.every(Number.isNaN));
  }
  assert.equal(sequence.frame(2).joints[10].visible,true);
  assert.equal(sequence.frame(1).visibleCount,sequence.frame(0).visibleCount-2);
});

test('mapping does not mutate input keypoints or use frame-specific depth',()=>{
  const f=fixture(),before=new Float32Array(f.points);f.set(1,10,.4,.1);const copy=new Float32Array(f.points);
  const sequence=createPose3DSequence(f.video,f.points);
  assert.notEqual(before[195+10*3],copy[195+10*3]);
  assert.deepEqual(sequence.frame(0).joints.map(j=>j.position[2]),sequence.frame(1).joints.map(j=>j.position[2]));
  sequence.frame(2);assert.deepEqual(f.points,copy);
});

test('degenerate and all-missing poses remain finite at sequence level with no exploding scale',()=>{
  const f=fixture();
  for(let frame=0;frame<3;frame++)for(const k of [5,6,11,12])f.set(frame,k,.5,.5);
  const degenerate=createPose3DSequence(f.video,f.points);assert.ok(degenerate.frame(0).scale<=8);
  f.points.fill(0);for(let i=0;i<f.points.length;i+=3){f.points[i]=NaN;f.points[i+1]=NaN;}
  const empty=createPose3DSequence(f.video,f.points);
  assert.ok([empty.floorY,empty.height,...empty.center].every(Number.isFinite));
  assert.equal(empty.frame(0).visibleCount,0);assert.ok(empty.frame(0).joints.every(j=>!j.visible));
});

test('an isolated ankle outlier does not move the display floor',()=>{
  const f=fixture(100);f.set(50,K.leftAnkle,.43,1);f.set(50,K.rightAnkle,.57,1);
  const floor=createPose3DSequence(f.video,f.points).floorY;
  f.set(50,K.leftAnkle,.43,.9);f.set(50,K.rightAnkle,.57,.9);
  near(floor,createPose3DSequence(f.video,f.points).floorY);
});

test('rejects malformed dimensions, binary sizes and invalid frame indices',()=>{
  const f=fixture();
  assert.throws(()=>createPose3DSequence({...f.video,width:0},f.points),/dimensions/);
  assert.throws(()=>createPose3DSequence({...f.video,height:Infinity},f.points),/dimensions/);
  assert.throws(()=>createPose3DSequence(f.video,f.points.subarray(1)),/length/);
  assert.throws(()=>createPose3DSequence({...f.video,frameCount:0},new Float32Array()),/length/);
  const sequence=createPose3DSequence(f.video,f.points);
  for(const frame of [-1,3,1.5,NaN,Infinity])assert.throws(()=>sequence.frame(frame),/index/);
});

test('invalid normalized coordinates and confidence values cannot enter the 3D scene',()=>{
  const f=fixture();f.set(0,7,2,.5);f.set(0,8,.4,.5,NaN);f.set(0,9,.4,.5,1.5);f.set(0,10,.4,.5,.5);
  const frame=createPose3DSequence(f.video,f.points).frame(0);
  for(const k of [7,8,9])assert.equal(frame.joints[k].visible,false);
  assert.equal(frame.joints[10].visible,true);
});

test('slow motion follows actual capture times between samples without crossing missing data', () => {
  const f = fixture(5), times = Float32Array.of(0, .02, .05, .25, .28);
  for (let i=0;i<5;i++) f.set(i,10,.1+times[i],.4);
  f.set(2,9,.32,.5,.2);
  const seq = createPose3DSequence(f.video,f.points);
  const sampler = createReplaySampler(seq,times,30,'balanced');
  const mid = sampler.atTime((times[1]+times[2])/2);
  near(mid.pose.joints[10].position[0], (seq.frame(1).joints[10].position[0]+seq.frame(2).joints[10].position[0])/2);
  assert.equal(mid.pose.joints[9].visible,false);
  assert.equal(sampler.atTime(.12).pose.visibleCount,0);
  assert.equal(sampler.atTime(times[3]).pose.joints[9].visible,true);
  assert.deepEqual(sampler.atTime((times[1]+times[2])/2),mid);
});

test('profile strength limits single-frame display corrections and preserves source samples', () => {
  const f = fixture(3), times=Float32Array.of(0,.02,.04);
  f.set(1,10,.69,.5);
  const seq = createPose3DSequence(f.video,f.points), raw = seq.frame(1);
  const x = raw.joints[10].position[0];
  const adjustment = (profile:'responsive'|'balanced'|'steady') =>
    x-createReplaySampler(seq,times,50,profile).frame(1).joints[10].position[0];
  assert.ok(0<adjustment('responsive')&&adjustment('responsive')<adjustment('balanced'));
  assert.ok(adjustment('balanced')<adjustment('steady'));
  f.set(1,10,.99,.1);
  const peakSeq = createPose3DSequence(f.video,f.points), peak=peakSeq.frame(1).joints[10].position;
  const displayed=createReplaySampler(peakSeq,times,50,'steady').frame(1).joints[10].position;
  assert.ok(Math.hypot(peak[0]-displayed[0],peak[1]-displayed[1])<=.050001);
  near(f.points[195+10*3],.99);
});
