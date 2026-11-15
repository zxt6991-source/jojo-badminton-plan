import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parseAnalysis,parseVideoAsset,validateAnalysis,validateVideoAsset,validateSkeletonData,CWB_65} from '../src/index.ts';
const fixture=JSON.parse(readFileSync(new URL('../../../fixtures/demo-analysis.json',import.meta.url),'utf8'));
const video=JSON.parse(readFileSync(new URL('../../../fixtures/demo-video.json',import.meta.url),'utf8'));

test('synthetic fixtures satisfy the frozen schema',()=>{
  assert.equal(parseAnalysis(fixture).schemaVersion,'0.1.0');
  assert.equal(parseVideoAsset(video).frameCount,720);
  assert.equal(CWB_65.rightHandStart+21,CWB_65.count);
});
test('runtime boundary rejects malformed confidence, target and unsupported versions',()=>{
  for(const mutate of [
    (x:any)=>x.swings[0].impact.confidence=1.2,
    (x:any)=>x.userInput.targetPoint=[1.01,.5],
    (x:any)=>x.schemaVersion='1.0.0',
    (x:any)=>x.quality.view='diagonal',
    (x:any)=>x.stageTimingsMs.pose=-10,
  ]) {const x=structuredClone(fixture);mutate(x);assert.equal(validateAnalysis(x),false);}
  assert.equal(validateVideoAsset({...video,captureFps:NaN}),false);
  assert.equal(validateVideoAsset({...video,frameCount:2.5}),false);
});
test('event/metric cross references and suppression invariants are checked',()=>{
  for(const mutate of [
    (x:any)=>x.metrics[0].swingId='missing',
    (x:any)=>x.metrics[0].suppressed={reason:'low_conf'},
    (x:any)=>x.swings[0].phases.impactFrame++,
    (x:any)=>x.swings[0].phases.backswingStartFrame=200,
    (x:any)=>x.swings[0].phases.followThroughEndFrame=720,
    (x:any)=>x.swings[1].id=x.swings[0].id,
    (x:any)=>x.quality.issues.push({code:'error',severity:'block'}),
  ]) {const x=structuredClone(fixture);mutate(x);assert.equal(validateAnalysis(x),false);}
});
test('binary validation enforces timestamps, lengths, finite confidence and missing point representation',()=>{
  const times=new Float32Array([0,.01]);const points=new Float32Array(2*65*3);
  for(let i=0;i<points.length;i+=3) {points[i]=NaN;points[i+1]=NaN;}
  assert.doesNotThrow(()=>validateSkeletonData(times,points,2));
  assert.throws(()=>validateSkeletonData(times,points,3),/size/);
  assert.throws(()=>validateSkeletonData(new Float32Array([0,0]),points,2),/increase/);
  points[0]=1.1; points[1]=.2; points[2]=.8;
  assert.throws(()=>validateSkeletonData(times,points,2),/normalized/);
  points[0]=.2;points[2]=NaN;
  assert.throws(()=>validateSkeletonData(times,points,2),/confidence/);
});
