import {test} from 'node:test';
import assert from 'node:assert/strict';
import {calculateMetrics,elbowAngleSeries} from '../src/index.ts';
import {type Analysis,type VideoAsset,CWB_65 as K} from '../../schema/src/index.ts';

function fixture() {
  const timestamps=new Float32Array([0,.015,.035,.06,.10]);
  const keypoints=new Float32Array(5*65*3);
  const pose:Record<number,number[]>={
    [K.nose]:[.58,.2],[K.leftShoulder]:[.45,.3],[K.rightShoulder]:[.5,.3],
    [K.rightElbow]:[.5,.4],[K.rightWrist]:[.6,.5],[K.leftHip]:[.45,.6],[K.rightHip]:[.5,.6],
    [K.leftBigToe]:[.48,.9],[K.leftSmallToe]:[.48,.9],[K.leftHeel]:[.4,.9],
    [K.rightBigToe]:[.68,.9],[K.rightSmallToe]:[.68,.9],[K.rightHeel]:[.6,.9],
  };
  for(let f=0;f<5;f++) for(let k=0;k<65;k++) {const at=(f*65+k)*3; const p=pose[k]??[.5,.5];keypoints[at]=p[0];keypoints[at+1]=p[1];keypoints[at+2]=.95;}
  const video:VideoAsset={id:'test',storageKey:'original.mp4',width:200,height:100,captureFps:120,frameCount:5,durationSec:.12,hasAudio:false,timeRemapped:false};
  const analysis:Analysis={id:'a',videoId:'test',status:'succeeded',schemaVersion:'0.1.0',pipelineVersion:'test',userInput:{},swings:[{id:'s',type:'smash',typeConfidence:1,handedness:'right',impact:{frame:2,tSec:timestamps[2],confidence:.95,evidence:['wrist-peak']},phases:{backswingStartFrame:0,impactFrame:2,followThroughEndFrame:4}}],metrics:[],diagnoses:[],stageTimingsMs:{},quality:{fpsTier:'high',view:'side',viewConfidence:1,bodyVisibleRatio:1,lighting:'ok',multiPerson:false,issues:[],usable:true}};
  const calc=()=>calculateMetrics(video,analysis,timestamps,keypoints);
  const set=(key:number,x:number,y:number,c=.95)=>{for(let f=0;f<5;f++){const i=(f*65+key)*3;keypoints.set([x,y,c],i);}};
  return {video,analysis,timestamps,keypoints,calc,set};
}
const near=(a:number|null,b:number)=>assert.ok(a!==null&&Math.abs(a-b)<1e-5,`${a} != ${b}`);
test('geometry corrects image aspect ratio and uses torso length',()=>{
  const {calc}=fixture();const m=calc();
  near(m[0].value,116.565051177);near(m[1].value,1/3);near(m[2].value,2/3);
  assert.ok(m.every(x=>x.confidence>=.5));
});
test('durations use irregular capture timestamps, never playback or nominal FPS',()=>{
  const f=fixture(),m=f.calc();near(m[3].value,.035);near(m[4].value,.065);
  f.video.captureFps=60;assert.deepEqual(f.calc().map(m=>m.value),m.map(m=>m.value));
});
test('30 and 45 FPS suppress temporal metrics while keeping spatial values',()=>{
  for(const fps of [30,45]) {const f=fixture();f.video.captureFps=fps;const m=f.calc();assert.notEqual(m[0].value,null);assert.equal(m[3].suppressed?.reason,'low_fps');assert.equal(m[4].value,null);}
});
test('view suppression follows each metric contract; frontal elbow confidence is discounted',()=>{
  const f=fixture();f.analysis.quality!.view='front';const m=f.calc();
  assert.notEqual(m[0].value,null);assert.ok(m[0].confidence<.95);assert.equal(m[1].suppressed?.reason,'bad_view');assert.equal(m[2].suppressed?.reason,'bad_view');assert.notEqual(m[3].value,null);
});
test('low confidence or missing joints are suppressed, never coerced into zeros',()=>{
  const f=fixture();f.set(K.rightWrist,NaN,NaN,0);const m=f.calc();
  assert.ok(m.every(x=>x.value===null&&x.suppressed?.reason==='low_conf'));
  assert.equal(elbowAngleSeries(f.video,f.analysis.swings[0],f.timestamps,f.keypoints)[0].value,null);
});
test('one reliable impact frame cannot hide an unreliable surrounding window',()=>{
  const f=fixture();for(const k of [K.rightShoulder,K.rightElbow,K.rightWrist]) for(let frame=0;frame<5;frame++) f.keypoints[(frame*65+k)*3+2]=frame===2?.95:.1;
  assert.equal(f.calc()[0].suppressed?.reason,'low_conf');
});
test('ambiguous body facing suppresses signed offset instead of guessing',()=>{
  const f=fixture();f.set(K.leftBigToe,.3,.9);f.set(K.leftSmallToe,.3,.9);
  assert.equal(f.calc()[2].suppressed?.reason,'low_conf');assert.notEqual(f.calc()[1].value,null);
});
test('unknown handedness and missing phase boundaries are explicit suppression',()=>{
  const f=fixture();f.analysis.swings[0].handedness='unknown';assert.ok(f.calc().every(m=>m.value===null));
  f.analysis.swings[0].handedness='right';delete f.analysis.swings[0].phases.backswingStartFrame;
  assert.equal(f.calc()[3].suppressed?.reason,'low_conf');assert.notEqual(f.calc()[4].value,null);
});
test('corrupt binary data and disagreement with captured impact time fail loudly',()=>{
  const f=fixture();f.analysis.swings[0].impact.tSec=.08;assert.throws(f.calc,/timestamp/);
  f.analysis.swings[0].impact.tSec=f.timestamps[2];f.timestamps[3]=.02;assert.throws(f.calc,/increase/);
});
test('nominal fractional 60/120 FPS rates retain temporal metrics',()=>{
  for(const fps of [60000/1001,120000/1001]){const f=fixture();f.video.captureFps=fps;assert.notEqual(f.calc()[3].value,null);}
});
test('real worker visual-only confidence and oblique view remain usable without multiplying scores',()=>{
  const f=fixture();f.analysis.swings[0].impact.confidence=.62;f.analysis.quality!.view='oblique';f.analysis.quality!.viewConfidence=.55;
  const m=f.calc();assert.notEqual(m[0].value,null);near(m[0].confidence,.55);assert.notEqual(m[1].value,null);
});
test('single-frame skeletons still check capture time agreement',()=>{
  const f=fixture();f.video.frameCount=1;f.analysis.swings[0].impact.frame=0;f.analysis.swings[0].impact.tSec=1;
  assert.throws(()=>calculateMetrics(f.video,f.analysis,new Float32Array([0]),f.keypoints.slice(0,65*3)),/timestamp/);
});
