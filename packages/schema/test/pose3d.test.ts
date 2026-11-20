import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {
  SCHEMA_VERSION, parseAnalysis, parseLegacyAnalysis, parsePose3DResult,
  validateAnalysis, validatePose3DResult, validatePose3DData, validatePose3DBuffer,
  POSE3D_POINT_COUNT, POSE3D_STRIDE, POSE3D_FRAME_BYTES, type Pose3DResult,
} from '../src/index.ts';

const legacy=JSON.parse(readFileSync(new URL('../../../fixtures/demo-analysis.json',import.meta.url),'utf8'));

function metadata(): Pose3DResult {
  return {
    status:'succeeded',source:'ai-monocular-predicted',modelId:'rtmw3d-rtmlib-x',runtime:'rtmlib-onnx',
    pipelineVersion:'pose3d-test-1',modelVersion:'contract-fixture-only',weightsSha256:'ab'.repeat(32),
    viewGatePolicyVersion:'test-1',scope:'manual',frameCount:5,
    windows:[{startFrame:1,endFrame:2,origin:'manual-annotation',viewClass:'side',viewGate:'pass',usableFrameRatio:1}],
    pointSpec:'cwb-65',coordinateSpace:'camera-root-relative',axisConvention:'x-right-y-up-z-forward',unit:'model-relative',rootJoint:'mid-hip',
    keypointsKey:'pose3d.f32',provenanceKey:'provenance.json',
    quality:{coverageRatio:2/5,usableFrameRatio:1,identityLockedRatio:1,meanJointConfidence:.75,warnings:[]},
  };
}

/** Synthetic contract bytes only, never a model or accuracy fixture. */
function points(): Float32Array {
  const p=new Float32Array(5*POSE3D_POINT_COUNT*POSE3D_STRIDE);
  for(let i=0;i<p.length;i+=4) p.set([NaN,NaN,NaN,0],i);
  p.set([-.3,.4,.7,.75],(1*65+9)*4);
  p.set([.3,.2,-.9,.75],(2*65+10)*4);
  for(const f of [1,2]) {
    p.set([-.2,.1,-.3,.75],(f*65+11)*4);
    p.set([.2,-.1,.3,.75],(f*65+12)*4);
  }
  return p;
}
function binary(p: Float32Array): Buffer {
  const bytes=Buffer.alloc(p.byteLength);
  p.forEach((v,i)=>bytes.writeFloatLE(v,i*4));
  return bytes;
}

test('v0.2 is additive and validated legacy records keep their version and 2D semantics',()=>{
  assert.equal(SCHEMA_VERSION,'0.2.0');
  const old=parseLegacyAnalysis(structuredClone(legacy));
  assert.equal(old.schemaVersion,'0.1.0');
  assert.equal(old.skeleton?.coordSpace,'image-normalized');
  const next=parseAnalysis({...old,schemaVersion:SCHEMA_VERSION});
  assert.deepEqual(next.metrics,old.metrics);
  assert.deepEqual(next.skeleton,old.skeleton);
  assert.throws(()=>parseLegacyAnalysis(next),/schemaVersion/);
  assert.throws(()=>parseLegacyAnalysis({...old,metrics:[{}]}));
});

test('3D analysis metadata must use v0.2, preserve successful 2D, and align frameCount',()=>{
  const pose=metadata();
  pose.frameCount=legacy.skeleton.frameCount;
  pose.quality.coverageRatio=2/pose.frameCount;
  const next={...structuredClone(legacy),schemaVersion:SCHEMA_VERSION,pose3d:pose};
  assert.equal(parseAnalysis(next).pose3d?.frameCount,720);
  for(const mutate of [
    (a:any)=>a.schemaVersion='0.1.0',
    (a:any)=>a.status='failed',
    (a:any)=>delete a.skeleton,
    (a:any)=>a.pose3d.frameCount++,
  ]) {const a=structuredClone(next);mutate(a);assert.equal(validateAnalysis(a),false);}
});

test('3D source, model runtime, hash, axes and units cannot silently change meaning',()=>{
  assert.equal(parsePose3DResult(metadata()).modelId,'rtmw3d-rtmlib-x');
  for(const mutate of [
    (p:any)=>p.source='schematic-2d',
    (p:any)=>p.modelId='unverified-model',
    (p:any)=>p.runtime='mmpose-pytorch',
    (p:any)=>p.weightsSha256='not-a-hash',
    (p:any)=>p.modelVersion='',
    (p:any)=>p.pipelineVersion='',
    (p:any)=>p.viewGatePolicyVersion='',
    (p:any)=>p.unit='meters',
    (p:any)=>p.coordinateSpace='world',
    (p:any)=>p.axisConvention='x-right-y-down-z-back',
    (p:any)=>p.rootJoint='nose',
    (p:any)=>p.pointSpec='coco-133',
    (p:any)=>p.frameCount=2.5,
    (p:any)=>p.frameCount=Number.MAX_SAFE_INTEGER,
    (p:any)=>p.quality.meanJointConfidence=NaN,
    (p:any)=>p.quality.identityLockedRatio=1.1,
    (p:any)=>p.quality.warnings=[null],
  ]) {const p=metadata();mutate(p);assert.equal(validatePose3DResult(p),false);}
});

test('3D file references and failure reasons obey independent task states',()=>{
  for(const status of ['disabled','queued','running','failed','rejected_quality'] as const) {
    const p=metadata();p.status=status;
    assert.equal(validatePose3DResult(p),false,'non-success cannot publish binary');
    delete p.keypointsKey;
    if(status==='failed'||status==='rejected_quality') {
      assert.equal(validatePose3DResult(p),false);
      p.errorCode='model_unavailable';
    }
    assert.equal(validatePose3DResult(p),true);
  }
  const p=metadata();delete p.keypointsKey;
  assert.equal(validatePose3DResult(p),false);
  p.keypointsKey='pose3d.f32';p.errorCode='model_unavailable';
  assert.equal(validatePose3DResult(p),false);
});

test('3D windows enforce bounds, order, scope, blocked views and usable coverage',()=>{
  for(const mutate of [
    (p:any)=>p.windows[0].startFrame=-1,
    (p:any)=>p.windows[0].endFrame=5,
    (p:any)=>p.windows[0].startFrame=3,
    (p:any)=>p.windows[0].endFrame=2.1,
    (p:any)=>p.windows.push({...p.windows[0]}),
    (p:any)=>p.windows[0].origin='auto-swing-event',
    (p:any)=>p.windows[0].viewClass='unknown',
    (p:any)=>p.windows[0].viewGate='blocked',
    (p:any)=>p.windows[0].usableFrameRatio=1.1,
    (p:any)=>p.quality.coverageRatio=1,
    (p:any)=>p.quality.usableFrameRatio=0,
    (p:any)=>p.windows=[],
  ]) {const p=metadata();mutate(p);assert.equal(validatePose3DResult(p),false);}
  const p=metadata();p.scope='full';p.windows[0].origin='full-clip';
  assert.equal(validatePose3DResult(p),false);
  p.windows[0].startFrame=0;p.windows[0].endFrame=4;
  p.quality.usableFrameRatio=.4;
  assert.equal(validatePose3DResult(p),true);
});

test('all-blocked and no-window rejections remain explicit and contain no artifact',()=>{
  const p=metadata();p.status='rejected_quality';p.errorCode='view_blocked';delete p.keypointsKey;
  p.windows[0].viewClass='unknown';p.windows[0].viewGate='blocked';p.windows[0].usableFrameRatio=0;
  p.quality.coverageRatio=0;p.quality.usableFrameRatio=0;
  assert.equal(validatePose3DResult(p),true);
  p.windows=[];p.errorCode='no_swing_windows';
  assert.equal(validatePose3DResult(p),true);
});

test('3D binary uses true XYZ and explicit little-endian decoding with Buffer offsets',()=>{
  const p=points();const bytes=binary(p);
  assert.equal(bytes.byteLength,5*POSE3D_FRAME_BYTES);
  const unaligned=Buffer.concat([Buffer.from([123]),bytes,Buffer.from([231])]).subarray(1,1+bytes.length);
  const decoded=validatePose3DBuffer(unaligned,metadata());
  assert.deepEqual(decoded,p);
  assert.ok(decoded[(1*65+9)*4+2]>0);
  assert.ok(decoded[(2*65+10)*4+2]<0);
  assert.deepEqual(validatePose3DBuffer(new Uint8Array(bytes).buffer,metadata()),p);
  assert.throws(()=>validatePose3DBuffer(bytes.subarray(0,-1),metadata()),/size/);
  assert.throws(()=>validatePose3DBuffer(Buffer.concat([bytes,Buffer.alloc(4)]),metadata()),/size/);
});

test('3D binary rejects every invalid missing pattern and nonfinite value',()=>{
  for(const values of [
    [0,0,0,0], [NaN,NaN,0,0], [NaN,NaN,Infinity,0],
    [.1,.2,.3,-.1], [.1,.2,.3,1.1], [.1,.2,.3,NaN],
    [NaN,.2,.3,.8], [.1,Infinity,.3,.8], [.1,.2,-Infinity,.8],
  ]) {
    const p=points();p.set(values,(1*65+9)*4);
    assert.throws(()=>validatePose3DData(p,metadata()));
  }
});

test('3D binary never publishes gap frames or blocked windows',()=>{
  for(const f of [0,3,4]) {
    const p=points();p.set([1,2,3,.5],f*65*4);
    assert.throws(()=>validatePose3DData(p,metadata()),/outside/);
  }
  const m=metadata();m.windows.push({startFrame:3,endFrame:4,origin:'manual-annotation',viewClass:'unknown',viewGate:'blocked',usableFrameRatio:0});
  const p=points();p.set([1,2,3,.5],3*65*4);
  assert.throws(()=>validatePose3DData(p,m),/outside/);
});

test('3D metadata cannot overstate binary coverage or usable frames',()=>{
  const p=points(),m=metadata();
  for(let point=0;point<65;point++) p.set([NaN,NaN,NaN,0],(2*65+point)*4);
  assert.throws(()=>validatePose3DData(p,m),/coverageRatio/);
  m.quality.coverageRatio=1/5;
  assert.throws(()=>validatePose3DData(p,m),/usableFrameRatio/);
  m.windows[0].usableFrameRatio=.5;m.quality.usableFrameRatio=.5;
  assert.doesNotThrow(()=>validatePose3DData(p,m));
});

test('populated 3D frames require both hips with midpoint at the root in all axes',()=>{
  for(const hip of [11,12]) {
    const p=points();p.set([NaN,NaN,NaN,0],(1*65+hip)*4);
    assert.throws(()=>validatePose3DData(p,metadata()),/root missing/);
  }
  for(let axis=0;axis<3;axis++) {
    const p=points();p[(1*65+11)*4+axis]+=.001;
    assert.throws(()=>validatePose3DData(p,metadata()),/root not centered/);
  }
  const p=points();p[(1*65+11)*4]+=1e-5;
  assert.doesNotThrow(()=>validatePose3DData(p,metadata()),'midpoint within 1e-5 tolerates rounding');
});

test('overall usability cannot exceed coverage inside unblocked windows',()=>{
  const p=metadata();p.quality.coverageRatio=.2;
  assert.throws(()=>parsePose3DResult(p),/usableFrameRatio/);
  p.quality.usableFrameRatio=.5;
  assert.doesNotThrow(()=>parsePose3DResult(p));
  // Blocked ranges never inflate the denominator used by Python quality.summarize.
  p.windows.push({startFrame:3,endFrame:4,origin:'manual-annotation',viewClass:'unknown',viewGate:'blocked',usableFrameRatio:0});
  assert.doesNotThrow(()=>parsePose3DResult(p));
  p.status='queued';delete p.keypointsKey;p.windows=[];p.quality.coverageRatio=0;
  assert.throws(()=>parsePose3DResult(p),/usableFrameRatio/);
  p.quality.usableFrameRatio=0;
  assert.doesNotThrow(()=>parsePose3DResult(p));
});
