import {test} from 'node:test';
import assert from 'node:assert/strict';
import {computeMovementAnalysis} from '../src/biomechanics.ts';
import type {Analysis,VideoAsset} from '../../schema/src/index.ts';

function fixture(moving=true){
 const n=41,times=new Float32Array(n);for(let f=1;f<n;f++)times[f]=times[f-1]+(f%2?.006:.011);
 const points=new Float32Array(n*195);
 for(let f=0;f<n;f++){
  const angle=(20+(moving?times[f]*90:0))*Math.PI/180;
  const pose:Record<number,number[]>={5:[.43,.35],6:[.5,.35],11:[.43,.62],12:[.5,.62],13:[.43,.77],14:[.56,.76],15:[.43,.92],16:[.5,.92]};
  pose[8]=[.5+Math.cos(angle)*.1,.35+Math.sin(angle)*.2];pose[10]=[pose[8][0]+Math.cos(angle*2)*.09,pose[8][1]+Math.sin(angle*2)*.18];
  for(let k=0;k<65;k++)points.set([...(pose[k]??[.4,.4]),.96],f*195+k*3);
 }
 const video:VideoAsset={id:'v',storageKey:'',width:200,height:100,captureFps:120,frameCount:n,durationSec:.35,hasAudio:false,timeRemapped:false};
 const analysis:Analysis={id:'a',videoId:'v',status:'succeeded',schemaVersion:'0.1.0',pipelineVersion:'test',userInput:{},swings:[{id:'s',type:'smash',typeConfidence:1,handedness:'right',impact:{frame:20,tSec:times[20],confidence:.9,evidence:['wrist-peak']},phases:{backswingStartFrame:3,impactFrame:20,followThroughEndFrame:37}}],metrics:[],diagnoses:[],stageTimingsMs:{},quality:{fpsTier:'high',view:'side',viewConfidence:.9,bodyVisibleRatio:1,lighting:'ok',multiPerson:false,issues:[],usable:true}};
 return{video,analysis,times,points,calc:()=>computeMovementAnalysis(video,analysis,analysis.swings[0],times,points)};
}

test('body angular speed uses capture times and correct image aspect',()=>{
 const f=fixture(),data=f.calc();const upper=data.segments.find(s=>s.id==='upperArm')!;
 assert.ok(upper.available);assert.ok(Math.abs(upper.samples.find(s=>s.frame===20)!.value!-90)<.01);
 f.video.captureFps=60;assert.deepEqual(f.calc().segments.map(s=>s.samples),data.segments.map(s=>s.samples));
});
test('static regions retain valid zeros without inventing motion peaks',()=>{
 const data=fixture(false).calc();assert.ok(data.segments.every(s=>s.available));
 assert.ok(data.segments.every(s=>s.peak===null&&s.samples.every(x=>x.value===null||x.value===0)));
});
test('losing one static knee never creates a false lower-limb peak',()=>{
 const f=fixture(false);for(let frame=18;frame<=23;frame++)for(const k of [11,13,15])f.points[frame*195+k*3+2]=.1;
 const legs=f.calc().segments.find(s=>s.id==='legs')!;assert.equal(legs.peak,null);assert.ok(legs.samples.every(s=>s.value===null||s.value===0));
});
test('low fps, bad view and uncertain impact suppress motion rather than fabricate dynamics',()=>{
 for(const apply of [(f:ReturnType<typeof fixture>)=>f.video.captureFps=30,(f:ReturnType<typeof fixture>)=>f.analysis.quality!.view='front',(f:ReturnType<typeof fixture>)=>f.analysis.swings[0].impact.confidence=.2]){
  const f=fixture();apply(f);const d=f.calc();assert.ok(d.segments.every(s=>!s.available&&s.peak===null&&s.samples.every(x=>x.value===null)));
 }
 const f=fixture();f.video.captureFps=30;assert.match(f.calc().measurements.find(m=>m.id==='wrist_speed')!.reason!,/帧率/);
});
test('unknown handedness hides arms but leaves observed torso and knees',()=>{
 const f=fixture();f.analysis.swings[0].handedness='unknown';const d=f.calc();
 assert.ok(d.segments.filter(s=>['upperArm','forearm'].includes(s.id)).every(s=>!s.available));
 assert.ok(d.segments.filter(s=>['legs','trunk'].includes(s.id)).every(s=>s.available));
 assert.equal(d.measurements.find(m=>m.id==='elbow_flexion')!.value,null);
});
test('missing phase boundaries remain absent; phase points preserve known frames',()=>{
 const f=fixture();delete f.analysis.swings[0].phases.backswingStartFrame;delete f.analysis.swings[0].phases.followThroughEndFrame;
 assert.deepEqual(f.calc().phases.map(p=>p.id),['impact']);assert.ok(f.calc().observations.some(o=>o.title.includes('边界')));
});
test('derivatives never bridge long missing-time intervals',()=>{
 const f=fixture();for(let i=21;i<f.times.length;i++)f.times[i]+=.2;
 const arm=f.calc().segments.find(s=>s.id==='upperArm')!;assert.equal(arm.samples.find(s=>s.frame===20)!.value,null);assert.equal(arm.samples.find(s=>s.frame===21)!.value,null);
});
test('confidence gaps remain null in the plotted data',()=>{
 const f=fixture();for(let frame=19;frame<=21;frame++)f.points[frame*195+8*3+2]=.1;
 const d=f.calc();assert.equal(d.segments.find(s=>s.id==='upperArm')!.samples.find(s=>s.frame===20)!.value,null);assert.equal(d.measurements.find(m=>m.id==='elbow_flexion')!.value,null);
});
