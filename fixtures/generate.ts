/** Deterministic synthetic contract fixture. These are not model predictions. */
import { writeFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { SCHEMA_VERSION, parseAnalysis, parseVideoAsset, validateSkeletonData, type Analysis, type VideoAsset } from '../packages/schema/src/index.ts';
import { calculateMetrics } from '../packages/metrics/src/index.ts';

const dir=fileURLToPath(new URL('.',import.meta.url));
const frames=720,fps=120,impacts=[164,380,596];
const timestamps=Float32Array.from({length:frames},(_,i)=>i/fps);
const points=new Float32Array(frames*65*3);
const base:number[][]=[
  [0.535,0.30],[0.53,0.285],[0.54,0.286],[0.50,0.30],[0.51,0.30],
  [0.475,0.41],[0.50,0.42],[0.44,0.51],[0.54,0.31],[0.40,0.42],[0.57,0.23],
  [0.465,0.62],[0.49,0.62],[0.44,0.76],[0.53,0.77],[0.42,0.90],[0.55,0.90],
  [0.455,0.93],[0.45,0.935],[0.415,0.93],[0.59,0.93],[0.585,0.935],[0.545,0.93],
];
// Relative frame / elbow x,y / wrist x,y. Coordinates interpolate smoothly.
const poses=[[-100,.54,.50,.57,.58],[-48,.44,.33,.39,.39],[-18,.51,.27,.48,.20],[0,.565,.265,.645,.13],[16,.63,.39,.73,.34],[50,.57,.56,.63,.69],[100,.54,.50,.57,.58]];
for(let f=0;f<frames;f++) {
  const impact=impacts.reduce((a,b)=>Math.abs(f-a)<Math.abs(f-b)?a:b),relative=f-impact;
  let a=poses[0],b=poses[0];
  for(let j=1;j<poses.length;j++) {a=poses[j-1];b=poses[j]; if(relative<=b[0]) break;}
  const u=Math.max(0,Math.min(1,(relative-a[0])/(b[0]-a[0]))),smooth=(1-Math.cos(u*Math.PI))/2;
  const coords=base.map(p=>[...p]);
  // Illustrative whole-body motion, not a learned ideal sequence or force model.
  // Feet stay on the drawn court; hip height, knee bend and torso lean vary.
  const bend=Math.exp(-Math.pow((relative+38)/36,2));
  const rise=Math.exp(-Math.pow((relative+3)/25,2));
  const follow=Math.exp(-Math.pow((relative-40)/30,2));
  const hipShift=.025*bend-.014*rise+.008*follow;
  const lean=-.035*bend+.015*rise+.045*follow;
  coords[11][1]+=hipShift;coords[12][1]+=hipShift;
  for(const k of [0,1,2,3,4,5,6,7,9]){coords[k][0]+=lean;coords[k][1]+=hipShift;}
  coords[13][0]-=.035*bend;coords[13][1]+=.013*bend;
  coords[14][0]+=.032*bend;coords[14][1]+=.013*bend;
  coords[8]=[a[1]+(b[1]-a[1])*smooth,a[2]+(b[2]-a[2])*smooth];
  coords[10]=[a[3]+(b[3]-a[3])*smooth,a[4]+(b[4]-a[4])*smooth];
  for(const k of [8,10]){coords[k][0]+=lean;coords[k][1]+=hipShift;}
  for(let hand=0;hand<2;hand++) {
    const wrist=coords[hand===0?9:10];
    for(let k=0;k<21;k++) coords.push([wrist[0]+Math.sin(k*1.4)*.006,wrist[1]+Math.cos(k*1.4)*.009]);
  }
  for(let k=0;k<65;k++) {
    const at=(f*65+k)*3;
    points[at]=coords[k][0];points[at+1]=coords[k][1];points[at+2]=k<23?0.96:0.82;
    // Third swing deliberately includes an occlusion to exercise suppression.
    if(impact===596&&Math.abs(relative)<=12&&(k===8||k===10)) points[at+2]=0.22;
  }
}
const video:VideoAsset={id:'demo',storageKey:'',width:1920,height:1080,captureFps:fps,frameCount:frames,durationSec:6,hasAudio:true,timeRemapped:false};
const analysis:Analysis={
  id:'demo',videoId:'demo',status:'succeeded',schemaVersion:SCHEMA_VERSION,pipelineVersion:'demo-synthetic-0.2.0',
  userInput:{swingType:'smash'},target:{trackId:1,lockedBy:'auto'},
  quality:{fpsTier:'high',view:'side',viewConfidence:.94,bodyVisibleRatio:.96,lighting:'ok',multiPerson:false,issues:[{code:'synthetic_demo',severity:'info'},{code:'partial_occlusion',severity:'warn'}],usable:true},
  skeleton:{spec:'cwb-65',source:'mono-2d',coordSpace:'image-normalized',frameCount:frames,timestampsKey:'/api/fixtures/timestamps.f32',keypointsKey:'/api/fixtures/keypoints.f32',smoothing:{method:'one-euro',params:{minCutoff:1,beta:.04,dCutoff:1}}},
  swings:impacts.map((frame,i)=>({id:`demo-swing-${i+1}`,type:'smash',typeConfidence:1,handedness:'right',impact:{frame,tSec:timestamps[frame],confidence:i===1?.79:.94,evidence:i===1?['wrist-peak']:['audio','wrist-peak']},phases:{backswingStartFrame:frame-48,impactFrame:frame,followThroughEndFrame:frame+50}})),
  metrics:[],diagnoses:[],stageTimingsMs:{},
};
analysis.metrics=calculateMetrics(video,analysis,timestamps,points);
parseVideoAsset(video);parseAnalysis(analysis);validateSkeletonData(timestamps,points,frames);
const binary=(data:Float32Array)=>{const buffer=Buffer.alloc(data.byteLength);data.forEach((v,i)=>buffer.writeFloatLE(v,i*4));return buffer;};
await mkdir(dir,{recursive:true});
await Promise.all([
  writeFile(`${dir}demo-video.json`,JSON.stringify(video,null,2)+'\n'),
  writeFile(`${dir}demo-analysis.json`,JSON.stringify(analysis,null,2)+'\n'),
  writeFile(`${dir}timestamps.f32`,binary(timestamps)),writeFile(`${dir}keypoints.f32`,binary(points)),
]);
console.log(`Wrote synthetic demo fixture: ${frames} frames, ${impacts.length} swings, ${analysis.metrics.length} metrics.`);
