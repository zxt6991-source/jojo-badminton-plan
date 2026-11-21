import {lazy,Suspense,useMemo,useState,type CSSProperties,type ReactNode} from 'react';
import type {Analysis,ReplayProfile,SwingEvent,VideoAsset} from '@badminton/schema';
import {computeMovementAnalysis,REGION_COLORS,type BodyRegion,type MovementPhaseId} from '@badminton/metrics';
import Icon from './Icon';
import Pose3DNotice from './Pose3DNotice';
import ReplayTuning from './ReplayTuning';
import './movement.css';
const Body3DViewer=lazy(()=>import('./Body3DViewer'));


export type BodyFocus=BodyRegion|'all';
interface Props {video:VideoAsset;analysis:Analysis;event:SwingEvent;points:Float32Array;timestamps:Float32Array;frame:number;onSeek:(frame:number)=>void;onFrame:(frame:number)=>void;seekToken:number;demo:boolean;focus:BodyFocus;onFocus:(region:BodyFocus)=>void;children?:ReactNode}
const labels:Record<BodyRegion,string>={legs:'下肢',trunk:'躯干',upperArm:'上臂',forearm:'前臂'};
const definitions:Record<BodyRegion,string>={legs:'双膝弯曲角变化',trunk:'躯干在画面内的倾斜变化',upperArm:'肩到肘方向的变化',forearm:'肘到腕方向的变化'};
const movementDescriptions:Record<BodyRegion,string>={
  legs:'慢放时看髋、膝与踝怎样移动，骨骼和周围肌肉会一起跟随动作。下方曲线对应双膝弯曲角的变化。',
  trunk:'观察胸廓、脊柱与骨盆在挥拍中的姿态变化，再对照下方的躯干倾斜曲线。',
  upperArm:'观察持拍侧从肩到肘的摆动，结合肌肉外形看清上臂方向。下方曲线对应肩到肘方向的变化。',
  forearm:'跟随肘、腕与手腕轨迹，看前臂在击球前后怎样延续动作。下方曲线对应肘到腕方向的变化。',
};
const prompts:Record<MovementPhaseId,{title:string;text:string;look:string}>={
  prepare:{title:'动作从怎样的姿态开始？',text:'先看膝的弯曲、躯干在画面中的倾斜，再看持拍臂的位置。这个时点用于建立后续动作的参照。',look:'高亮下肢与躯干，回看身体姿态。'},
  accelerate:{title:'身体与手臂怎样衔接？',text:'慢放引拍至击球的过程。分别观察躯干、上臂与前臂的变化，再用下方曲线定位各部位变化较快的片段。',look:'依次高亮躯干、上臂、前臂，对照同一时间。'},
  impact:{title:'击球瞬间，身体处在什么位置？',text:'看手腕相对肩的位置、肘是否接近伸直，以及躯干与下肢的姿态。这些是画面中的几何关系。',look:'黄色圆环标记手腕参考位置，不是已追踪的球拍触球点。'},
  follow:{title:'击球以后，动作怎样延续？',text:'跟随手腕轨迹，看前臂、上臂和躯干继续向哪里移动。片段结束不等于身体受力归零。',look:'观察轨迹方向与关节变化，不根据随挥长短判断好坏。'},
};
const relative=(time:number,impact:number)=>Math.abs(time-impact)<.005?'击球时':`击球${time<impact?'前':'后'} ${Math.abs(time-impact).toFixed(2)} s`;

export default function MovementInsight({video,analysis,event,points,timestamps,frame,onSeek,onFrame,seekToken,demo,focus,onFocus,children}:Props){
  const [spatialView,setSpatialView]=useState<'3d'|'video'>('3d');
  const [replayProfile,setReplayProfile]=useState<ReplayProfile>('balanced');
  const motion=useMemo(()=>computeMovementAnalysis(video,analysis,event,timestamps,points),[video,analysis,event,timestamps,points]);
  const currentTime=timestamps[frame]??event.impact.tSec;
  const start=event.phases.backswingStartFrame,end=event.phases.followThroughEndFrame;
  const phase:MovementPhaseId|undefined=Math.abs(frame-event.impact.frame)<=1?'impact':start!==undefined&&frame===start?'prepare':start!==undefined&&frame>start&&frame<event.impact.frame?'accelerate':end!==undefined&&frame>event.impact.frame&&frame<=end?'follow':undefined;
  const prompt=phase?prompts[phase]:{title:'逐帧观察这段动作',text:'当前画面位于已标记阶段之外。可点击上方观察位置，回到这一拍的引拍、挥拍过程或随挥。',look:'阶段边界缺失时，我们保留空白，不自动补齐。'};
  const t0=timestamps[motion.window.startFrame],t1=timestamps[motion.window.endFrame],span=Math.max(.001,t1-t0);
  const maxSpeed=Math.max(1,...motion.segments.flatMap(s=>s.samples.map(p=>p.value??0)));
  const plot=(samples:typeof motion.segments[number]['samples'])=>{let d='',connected=false;for(const s of samples){if(s.value===null){connected=false;continue;}d+=`${connected?'L':'M'}${(12+(s.tSec-t0)/span*576).toFixed(2)},${(56-s.value/maxSpeed*42).toFixed(2)} `;connected=true;}return d;};
  return <section className="movement-insight" aria-labelledby="movement-title">
    {!demo&&<Pose3DNotice analysisId={analysis.id} initial={analysis.pose3d}/>}
    <div className="movement-heading"><div><span className="eyebrow">READ THE WHOLE MOVEMENT</span><h2 id="movement-title">身体如何完成这一拍</h2><p>让骨骼与肌肉跟随挥拍，慢放看清身体各部位怎样衔接。</p></div><span className="movement-scope"><Icon name="eye" size={15}/>3D 动作 · 2D 测量</span></div>
    <div className="movement-phases" aria-label="动作观察位置">{(['prepare','accelerate','impact','follow'] as MovementPhaseId[]).map((id,i)=>{const p=motion.phases.find(x=>x.id===id);return <button key={id} className={phase===id?'selected':''} disabled={!p} aria-pressed={phase===id} onClick={()=>p&&onSeek(id==='accelerate'?Math.round((p.startFrame+p.endFrame)/2):id==='follow'?Math.round((p.startFrame+p.endFrame)/2):p.startFrame)}><span className="phase-index">0{i+1}</span><span><strong>{id==='accelerate'?'挥拍过程':{prepare:'引拍起点',impact:'击球',follow:'随挥'}[id]}</strong><small>{p?(p.startFrame===p.endFrame?relative(timestamps[p.startFrame],event.impact.tSec):`${(timestamps[p.endFrame]-timestamps[p.startFrame]).toFixed(2)} s 观察段`):'边界未识别'}</small></span><Icon name="chevron" size={15}/></button>;})}</div>
    <div className="movement-stage">
      <div className="movement-video">
        <div className="movement-view-switch" aria-label="身体展示方式">
          <button className={spatialView==='3d'?'selected':''} aria-pressed={spatialView==='3d'} onClick={()=>setSpatialView('3d')}>3D 动作回放</button>
          <button className={spatialView==='video'?'selected':''} aria-pressed={spatialView==='video'} onClick={()=>setSpatialView('video')}>{demo?'2D 骨架对照':'视频对照'}</button>
          <span>共用同一时间轴</span>
        </div>
        <div hidden={spatialView!=='3d'}><ReplayTuning analysisId={analysis.id} profile={replayProfile} onChange={setReplayProfile}/></div>
        {spatialView==='3d'?<Suspense fallback={<div className="body3d-import-loading" role="status"><span className="spinner"/>正在加载 3D 动作人体…</div>}><Body3DViewer video={video} analysis={analysis} event={event} points={points} timestamps={timestamps} frame={frame} onFrame={onFrame} onSeek={onSeek} seekToken={seekToken} focus={focus} onFocus={onFocus} demo={demo} replayProfile={replayProfile}/></Suspense>:children}
        {spatialView==='video'&&<div className="movement-color-key"><span>按部位看</span><button className={focus==='all'?'active':''} aria-pressed={focus==='all'} onClick={()=>onFocus('all')}>全身</button>{(Object.keys(labels) as BodyRegion[]).map(id=><button key={id} style={{'--region-color':REGION_COLORS[id]} as CSSProperties} className={focus===id?'active':''} aria-pressed={focus===id} onClick={()=>onFocus(focus===id?'all':id)}><i/>{labels[id]}</button>)}</div>}
      </div>
      <aside className="movement-reader">
        <div className="reader-label"><span><i/>{phase?{prepare:'引拍参考帧',accelerate:'挥拍过程',impact:'击球参考帧',follow:'随挥过程'}[phase]:'当前画面'}</span><span>{currentTime.toFixed(3)} s</span></div>
        <h3>{prompt.title}</h3>
        <p>{prompt.text}</p>
        <div className="reader-body reader-body-3d"><div className="reader-focus">
          <span className="eyebrow">观察重点</span>
          <strong>{focus==='all'?(spatialView==='3d'?'慢放这一拍，转动视角看动作':'逐帧观察身体变化'):labels[focus]}</strong>
          <p>{focus==='all'?(spatialView==='3d'?'点击播放，骨骼与肌肉会随姿态一起运动。拖动可换角度，点击身体区域可查看下肢、躯干或持拍臂。':'慢放或逐帧查看当前动作，选择身体区域，对照关键点与下方曲线。'):movementDescriptions[focus]}</p>
          <button onClick={()=>onSeek(event.impact.frame)}><Icon name="target" size={14}/>回到击球帧</button>
        </div></div>
        <div className="reader-footnote"><Icon name="info" size={15}/><span>3D 动画由二维姿态驱动，前后深度为示意。肌肉外形与颜色不表示本次挥拍的受力大小。</span></div>
      </aside>
    </div>
    <div className="movement-section-title"><div><span className="eyebrow">01 / BODY REGIONS</span><h3>这一刻，哪些部位在变化？</h3></div><p>点击卡片，聚焦 3D 人体中的对应区域</p></div>
    <div className="region-cards">{motion.segments.map(s=>{const sample=s.samples.find(x=>x.frame===frame);return <button key={s.id} className={`region-card ${focus===s.id?'selected':''}`} style={{'--region-color':s.color} as CSSProperties} aria-pressed={focus===s.id} onClick={()=>onFocus(focus===s.id?'all':s.id)}><div><i/><strong>{s.label}</strong><Icon name="arrow" size={15}/></div><p>{definitions[s.id]}</p><span className="region-value">{sample?.value!==null&&sample?.value!==undefined?Math.round(sample.value):'—'}<small>°/s</small></span><span className="region-caption">{!s.available?s.reason:sample?.value===null||sample===undefined?'当前帧缺少连续数据':'当前画面的角度变化速度'}</span><span className="region-coverage"><span>连续样本覆盖</span><span>{Math.round(s.coverage*100)}%</span><i><b style={{width:`${s.coverage*100}%`}}/></i></span></button>;})}</div>
    <section className="movement-chart"><div className="movement-section-title"><div><span className="eyebrow">02 / TIMING TOGETHER</span><h3>把各部位放在同一条时间线上</h3></div><span className="chart-help">参考虚线：击球时刻</span></div><p className="movement-chart-intro">以下数值来自二维关键点，转动 3D 视角不会改变测量。曲线越高，表示该角度变化越快。点击右侧峰值，人体回放会定位到对应帧。峰值先后不代表力量传递顺序。</p>
    <div className="motion-chart-rows">{motion.segments.map(s=><div key={s.id} className={`motion-chart-row ${focus!=='all'&&focus!==s.id?'dimmed':''}`}><span className="motion-row-label" style={{color:s.color}}>{s.label}<small>{definitions[s.id]}</small></span><svg viewBox="0 0 600 70" preserveAspectRatio="none" role="img" aria-label={`${s.label}角度变化速度曲线`}><line x1="12" x2="588" y1="56" y2="56" stroke="#2d3a32"/><line x1={12+(event.impact.tSec-t0)/span*576} x2={12+(event.impact.tSec-t0)/span*576} y1="4" y2="62" stroke="#73856e" strokeDasharray="3 4"/><path d={plot(s.samples)} fill="none" stroke={s.color} strokeWidth="2" vectorEffect="non-scaling-stroke"/>{currentTime>=t0&&currentTime<=t1&&<line x1={12+(currentTime-t0)/span*576} x2={12+(currentTime-t0)/span*576} y1="4" y2="62" stroke="#eff4e7" opacity=".8"/>}{!s.available&&<text x="300" y="35" textAnchor="middle" fill="#89988e" fontSize="11">数据不足，曲线隐藏</text>}</svg><button disabled={!s.peak} onClick={()=>s.peak&&onSeek(s.peak.frame)} className="peak-button"><strong>{s.peak?relative(s.peak.tSec,event.impact.tSec):s.available?'未见明显变化':'暂不可判断'}</strong><span>{s.peak?`${Math.round(s.peak.value!)} °/s · 变化峰值`:'保留不确定性'}{s.peak&&<Icon name="arrow" size={12}/>}</span></button></div>)}</div>
    <div className="movement-chart-time"><span>{relative(t0,event.impact.tSec)}</span><span>统一纵轴：0–{Math.ceil(maxSpeed)} °/s</span><span>{relative(t1,event.impact.tSec)}</span></div><input className="movement-time-slider" type="range" min={motion.window.startFrame} max={motion.window.endFrame} value={Math.max(motion.window.startFrame,Math.min(motion.window.endFrame,frame))} aria-label="身体协同时间轴" onChange={e=>onSeek(Number(e.target.value))}/></section>
    <div className="movement-observations">{motion.observations.slice(1).map((o,i)=><article key={i}><Icon name={o.tone==='limited'?'info':'activity'} size={19}/><div><h4>{o.title}</h4><p>{o.text}</p></div></article>)}</div>
    <details className="movement-details"><summary>理解数值与测量范围 <Icon name="chevron" size={15}/></summary><div className="movement-detail-content"><h4>击球帧的姿态与速度</h4><div className="movement-measurements">{motion.measurements.map(m=><div key={m.id}><strong>{m.label}<span>{m.value===null?'未显示':m.value.toFixed(1)} {m.value===null?'':m.unit==='deg'?'°':'躯干长度/s'}</span></strong><p>{m.reason??m.explanation}</p></div>)}</div>{motion.limits.map((line,i)=><p key={i}>{line}</p>)}<p>引拍与随挥边界为粗略事件估计；“挥拍过程”包含引拍到击球，并非已验证的加速阶段分割。{demo?'本例为合成演示，不代表真实人体的发力模式。':''}</p><p>方法参考：<a href="https://www.stuartmcnaylor.com/publication/badtech1/King_et_al_2020.pdf" target="_blank" rel="noreferrer">羽毛球跳杀三维运动学研究</a> · <a href="https://doi.org/10.1371/journal.pcbi.1011462" target="_blank" rel="noreferrer">OpenCap 运动学与动力学方法</a></p></div></details>
  </section>;
}
