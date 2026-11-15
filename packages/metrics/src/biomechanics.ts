import {validateSkeletonData,type Analysis,type SwingEvent,type VideoAsset} from '../../schema/src/index.ts';

export type BodyRegion='legs'|'trunk'|'upperArm'|'forearm';
export type MovementPhaseId='prepare'|'accelerate'|'impact'|'follow';
export const REGION_COLORS:Record<BodyRegion,string>={legs:'#68c9e8',trunk:'#efbe72',upperArm:'#ba9bff',forearm:'#c5ef8a'};
export interface MotionSample {frame:number;tSec:number;value:number|null;confidence:number}
export interface MovementPhase {id:MovementPhaseId;label:string;startFrame:number;endFrame:number;confidence:number;estimated:boolean}
export interface MovementSegment {id:BodyRegion;label:string;color:string;samples:MotionSample[];peak:MotionSample|null;coverage:number;available:boolean;reason?:string}
export interface MovementMeasurement {id:string;label:string;value:number|null;unit:'deg'|'torso/s';explanation:string;confidence:number;reason?:string}
export interface MovementAnalysis {definitionVersion:string;window:{startFrame:number;endFrame:number};phases:MovementPhase[];segments:MovementSegment[];measurements:MovementMeasurement[];observations:{title:string;text:string;tone:'neutral'|'limited'}[];limits:string[]}
type P={x:number;y:number;c:number};
const good=(p:P)=>p.c>=.5&&Number.isFinite(p.x)&&Number.isFinite(p.y);
const mid=(a:P,b:P):P=>({x:(a.x+b.x)/2,y:(a.y+b.y)/2,c:Math.min(a.c,b.c)});
function flex(a:P,b:P,c:P):number|null {if(![a,b,c].every(good))return null;const ax=a.x-b.x,ay=a.y-b.y,cx=c.x-b.x,cy=c.y-b.y,den=Math.hypot(ax,ay)*Math.hypot(cx,cy);return den>1e-8?180-Math.acos(Math.max(-1,Math.min(1,(ax*cx+ay*cy)/den)))*180/Math.PI:null;}
const orientation=(a:P,b:P)=>[a,b].every(good)&&Math.hypot(a.x-b.x,a.y-b.y)>1e-8?Math.atan2(b.x-a.x,a.y-b.y)*180/Math.PI:null;
const deltaAngle=(a:number,b:number)=>((b-a+540)%360)-180;

/** Projected kinematics only. No forces, energy transfer, muscle activation or ideal-sequence score. */
export function computeMovementAnalysis(video:VideoAsset,analysis:Analysis,swing:SwingEvent,timestamps:Float32Array,keypoints:Float32Array):MovementAnalysis {
  validateSkeletonData(timestamps,keypoints,video.frameCount);
  if(analysis.skeleton&&analysis.skeleton.frameCount!==video.frameCount)throw new RangeError('Video and skeleton frame counts differ');
  const impact=swing.impact.frame;
  if(!Number.isInteger(impact)||impact<0||impact>=timestamps.length||Math.abs(timestamps[impact]-swing.impact.tSec)>.001)throw new RangeError('Movement impact must match capture timestamp');
  const pose=(f:number,k:number):P=>{const o=f*195+k*3;return{x:keypoints[o]*video.width/video.height,y:keypoints[o+1],c:keypoints[o+2]};};
  const left=swing.handedness==='left',arm=left?[5,7,9]:[6,8,10];
  const validFrame=(f:number|undefined)=>f!==undefined&&Number.isInteger(f)&&f>=0&&f<timestamps.length;
  const start=swing.phases.backswingStartFrame,end=swing.phases.followThroughEndFrame;
  const hasStart=validFrame(start)&&start! < impact,hasEnd=validFrame(end)&&end! > impact;
  const before=hasStart?timestamps[start!]-.15:timestamps[impact]-.65,after=hasEnd?timestamps[end!]+.1:timestamps[impact]+.65;
  let first=impact,last=impact;while(first>0&&timestamps[first-1]>=before)first--;while(last<timestamps.length-1&&timestamps[last+1]<=after)last++;
  const viewOk=['side','oblique'].includes(analysis.quality?.view??'')&&(analysis.quality?.viewConfidence??0)>=.5;
  const rateOk=video.captureFps>=59&&analysis.quality?.fpsTier!=='low';
  const quality=Math.min(analysis.quality?.viewConfidence??0,swing.impact.confidence);
  const baseReason=!viewOk?'当前机位无法可靠比较画面内的关节运动':quality<.5?'击球时刻或视角置信度不足':undefined;
  const frames=Array.from({length:last-first+1},(_,i)=>first+i);
  const raw=(id:BodyRegion,f:number):{v:number|null;c:number}=>{
    if(id==='legs'){
      const sides=[[11,13,15],[12,14,16]].map(ids=>{const p=ids.map(k=>pose(f,k));return {v:flex(p[0],p[1],p[2]),c:Math.min(...p.map(x=>x.c))};}).filter(x=>x.v!==null);
      return sides.length?{v:sides.reduce((s,x)=>s+x.v!,0)/sides.length,c:Math.min(...sides.map(x=>x.c))}:{v:null,c:0};
    }
    if(id==='trunk'){const sh=mid(pose(f,5),pose(f,6)),hip=mid(pose(f,11),pose(f,12));return{v:orientation(hip,sh),c:Math.min(sh.c,hip.c)};}
    if(swing.handedness==='unknown')return{v:null,c:0};
    const a=pose(f,id==='upperArm'?arm[0]:arm[1]),b=pose(f,id==='upperArm'?arm[1]:arm[2]);return{v:orientation(a,b),c:Math.min(a.c,b.c)};
  };
  const labels:Record<BodyRegion,string>={legs:'下肢',trunk:'躯干',upperArm:'持拍上臂',forearm:'持拍前臂'};
  const segments=(Object.keys(labels) as BodyRegion[]).map(id=>{
    const samples=frames.map(frame=>{
      const a=frame-1,b=frame+1;
      if(a<first||b>last||timestamps[b]-timestamps[a]>.050001)return{frame,tSec:timestamps[frame],value:null,confidence:0};
      if(id==='legs'){
        // Differentiate each knee with a consistent set of joints. Changing the
        // membership of a per-frame average creates false peaks under occlusion.
        const sides=[[11,13,15],[12,14,16]].map(ids=>{
          const poses=[a,frame,b].map(f=>ids.map(k=>pose(f,k))),angles=poses.map(p=>flex(p[0],p[1],p[2]));
          const confidence=Math.min(quality,...poses.flat().map(p=>p.c));
          return {value:angles.some(v=>v===null)||confidence<.5?null:Math.abs(angles[2]!-angles[0]!)/(timestamps[b]-timestamps[a]),confidence};
        }).filter(s=>s.value!==null);
        return{frame,tSec:timestamps[frame],value:sides.length?sides.reduce((v,s)=>v+s.value!,0)/sides.length:null,confidence:sides.length?Math.min(...sides.map(s=>s.confidence)):0};
      }
      const pa=raw(id,a),pb=raw(id,b),pc=raw(id,frame),confidence=Math.min(pa.c,pb.c,pc.c,quality);
      return{frame,tSec:timestamps[frame],value:pa.v===null||pb.v===null||pc.v===null||confidence<.5?null:Math.abs(deltaAngle(pa.v,pb.v))/(timestamps[b]-timestamps[a]),confidence};
    });
    const coverage=samples.filter(s=>s.value!==null).length/Math.max(1,samples.length);
    const reason=baseReason??(!rateOk?'帧率不足，无法可靠比较变化速度':coverage<.7?'这段动作的可用关键点不足 70%':undefined);
    if(reason)samples.forEach(s=>s.value=null);
    const available=!reason;
    const candidate=samples.reduce<MotionSample|null>((best,s)=>s.value!==null&&(!best||s.value>best.value!)?s:best,null);
    const peak=candidate&&candidate.value!>.001?candidate:null;
    return{id,label:labels[id],color:REGION_COLORS[id],samples,peak,coverage,available,reason};
  });
  const phase=(id:MovementPhaseId,label:string,a:number,b:number):MovementPhase=>({id,label,startFrame:a,endFrame:b,confidence:swing.impact.confidence,estimated:!swing.impact.evidence.includes('manual')||id!=='impact'});
  const phases:MovementPhase[]=[];
  if(hasStart){phases.push(phase('prepare','引拍起点',start!,start!));phases.push(phase('accelerate','挥拍过程',start!,impact));}
  phases.push(phase('impact','击球',impact,impact));if(hasEnd)phases.push(phase('follow','随挥',impact,end!));
  const measure=(id:string,label:string,value:number|null,confidence:number,explanation:string,unit:'deg'|'torso/s'='deg'):MovementMeasurement=>{
    const reason=baseReason??(confidence<.5||value===null?'关键点不足，暂不显示':undefined);
    return{id,label,value:reason?null:value,unit,explanation,confidence:Math.min(quality,confidence),reason};
  };
  const k=raw('legs',impact),tr=raw('trunk',impact),up=raw('upperArm',impact),ap=arm.map(j=>pose(impact,j));
  const sh=mid(pose(impact,5),pose(impact,6)),hip=mid(pose(impact,11),pose(impact,12)),L=Math.hypot(sh.x-hip.x,sh.y-hip.y);
  let speed:number|null=null,speedConf=0;
  if(impact>first&&impact<last&&timestamps[impact+1]-timestamps[impact-1]<=.050001&&rateOk&&swing.handedness!=='unknown'){
    const a=pose(impact-1,arm[2]),b=pose(impact+1,arm[2]);speedConf=Math.min(a.c,b.c,ap[2].c,sh.c,hip.c);
    if([a,b,ap[2],sh,hip].every(good)&&L>1e-5)speed=Math.hypot(b.x-a.x,b.y-a.y)/(timestamps[impact+1]-timestamps[impact-1])/L;
  }
  const measurements=[
    measure('knee_flexion','膝弯曲',k.v,k.c,'可见双腿膝弯曲角的平均值；0° 表示伸直。'),
    measure('trunk_lean','躯干倾斜',tr.v===null?null:Math.abs(tr.v),tr.c,'肩中点到髋中点连线与画面竖直方向的夹角，不是躯干转体角。'),
    measure('upper_arm_angle','上臂朝向',up.v===null?null:Math.abs(up.v),up.c,'肩到肘方向与画面竖直向上的夹角；不表示肩内旋角。'),
    measure('elbow_flexion','肘弯曲',swing.handedness==='unknown'?null:flex(ap[0],ap[1],ap[2]),Math.min(...ap.map(x=>x.c)),'0° 表示肘伸直；与基础指标中的肩—肘—腕夹角互补。'),
    {...measure('wrist_speed','手腕移动速度',speed,speedConf,'击球附近的画面内速度，以击球帧躯干长度为单位；不是球速或球拍速度。','torso/s'),...(!rateOk?{reason:'帧率不足，无法可靠计算手腕速度'}:{})},
  ];
  const peaks=segments.filter(s=>s.peak).sort((a,b)=>a.peak!.tSec-b.peak!.tSec);
  const observations:MovementAnalysis['observations']=[{title:'先看动作如何衔接',text:'切换观察位置，再分别高亮下肢、躯干和持拍臂。曲线显示各部位在画面内变化的速度，帮助定位值得回看的片段。',tone:'neutral'}];
  if(peaks.length>=2)observations.push({title:'本拍的可见变化',text:peaks.map(s=>`${s.label}的变化峰值在击球${s.peak!.tSec<timestamps[impact]?'前':'后'} ${Math.abs(s.peak!.tSec-timestamps[impact]).toFixed(2)} 秒`).join('；')+'。峰值先后不等于力量传递顺序。',tone:'neutral'});
  if(segments.some(s=>!s.available||s.coverage<.95))observations.push({title:'部分身体运动暂不可判断',text:'遮挡、机位或采集帧率限制了变化曲线。空白或隐藏区域不代表该部位没有运动。',tone:'limited'});
  if(!hasStart||!hasEnd)observations.push({title:'阶段边界尚不完整',text:'未识别到的引拍或随挥边界不会自动补齐；仍可逐帧查看击球附近。',tone:'limited'});
  return{definitionVersion:'movement-2d-0.1.0',window:{startFrame:first,endFrame:last},phases,segments,measurements,observations,limits:['这是二维运动学：描述身体怎样移动，不测量力、关节力矩或肌肉激活。','上臂内旋、前臂旋前与骨盆—胸廓转体需要额外的三维信息；本页不以平面角度代替。','不同身体区域采用不同角度定义。曲线可对齐时间，但不能据其高度比较力量贡献。']};
}
