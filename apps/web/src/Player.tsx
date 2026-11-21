import { useCallback, useEffect, useRef, useState } from 'react';
import type { Analysis, VideoAsset } from '@badminton/schema';
import {REGION_COLORS,type BodyRegion} from '@badminton/metrics';
import {closestFrame} from './pose3d';
import Icon from './Icon';

export {closestFrame} from './pose3d';

export const bodyEdges = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4],[15,17],[17,18],[15,19],[16,20],[20,21],[16,22]];
const allEdges = [...bodyEdges];
for (const start of [23,44]) for (let finger = 0; finger < 5; finger++) {
  const base = start + 1 + finger * 4;
  allEdges.push([start,base],[base,base+1],[base+1,base+2],[base+2,base+3]);
}

function court(ctx: CanvasRenderingContext2D, width: number, height: number) {
  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, '#121c1a'); bg.addColorStop(.6, '#1c3028'); bg.addColorStop(1, '#2c4436');
  ctx.fillStyle = bg; ctx.fillRect(0,0,width,height);
  ctx.fillStyle = '#20332a'; ctx.beginPath(); ctx.moveTo(width*.16,height*.4); ctx.lineTo(width*.84,height*.4); ctx.lineTo(width*1.04,height); ctx.lineTo(-width*.04,height); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = 'rgba(196,216,184,.2)'; ctx.lineWidth = 1.5;
  const ys = [.4,.53,.7,.95];
  const bound = (y:number) => .16 - ((y-.4)/.6)*.20;
  ys.forEach(y => {ctx.beginPath(); ctx.moveTo(width*bound(y),height*y); ctx.lineTo(width*(1-bound(y)),height*y);ctx.stroke();});
  [.16,.23,.5,.77,.84].forEach((x,i)=> {ctx.beginPath();ctx.moveTo(width*x,height*.4);ctx.lineTo(width*[-.04,.07,.5,.93,1.04][i],height);ctx.stroke();});
  // Quiet architectural details keep the synthetic scene recognizably illustrative.
  ctx.strokeStyle = 'rgba(215,230,212,.07)'; ctx.lineWidth = 1;
  for(let x=0;x<width;x+=width/12){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,height*.36);ctx.stroke();}
  ctx.beginPath(); ctx.moveTo(0,height*.34);ctx.lineTo(width,height*.34);ctx.stroke();
  const light = ctx.createRadialGradient(width*.53,height*.3,10,width*.53,height*.3,width*.5);
  light.addColorStop(0,'rgba(187,229,155,.07)');light.addColorStop(1,'rgba(187,229,155,0)');
  ctx.fillStyle=light;ctx.fillRect(0,0,width,height);
}

interface Props {
  video: VideoAsset; analysis: Analysis; points: Float32Array; timestamps: Float32Array;
  frame: number; onFrame: (frame: number) => void; demo: boolean; onMark: () => void; marking: boolean;
  seekToken: number;
  colorByRegion?:boolean; focus?:BodyRegion|'all'; handedness?:'left'|'right'|'unknown';
}

export default function Player({ video, analysis, points, timestamps, frame, onFrame, demo, onMark, marking, seekToken,colorByRegion=false,focus='all',handedness='unknown' }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null), videoRef = useRef<HTMLVideoElement>(null);
  const [playing,setPlaying] = useState(false), [speed,setSpeed] = useState(.5), [overlay,setOverlay] = useState(true);
  const [videoError,setVideoError] = useState(false);
  const frameRef = useRef(frame); frameRef.current = frame;
  const time = timestamps[frame] || 0;
  const duration = timestamps[timestamps.length-1] || video.durationSec;

  const draw = useCallback(() => {
    const canvas = canvasRef.current; if(!canvas) return;
    const rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
    canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
    const ctx=canvas.getContext('2d');if(!ctx) return;
    ctx.scale(dpr,dpr);const w=rect.width,h=rect.height;
    if(demo)court(ctx,w,h);else ctx.clearRect(0,0,w,h);
    if(!overlay)return;
    const scale=Math.min(w/video.width,h/video.height), iw=video.width*scale,ih=video.height*scale;
    const ox=(w-iw)/2,oy=(h-ih)/2, offset=frame*65*3;
    const point=(id:number)=>[ox+points[offset+id*3]*iw,oy+points[offset+id*3+1]*ih,points[offset+id*3+2]];
    const valid=(p:number[])=>p[2]>=(colorByRegion ? .5 : .35)&&Number.isFinite(p[0])&&Number.isFinite(p[1]);
    const holdingArm=handedness==='left'?[5,7,9]:handedness==='right'?[6,8,10]:[];
    const edgeRegion=(a:number,b:number):BodyRegion|undefined=>{
      if([5,6,11,12].includes(a)&&[5,6,11,12].includes(b))return 'trunk';
      if(a>=11&&a<=22&&b>=13&&b<=22)return 'legs';
      if(a===holdingArm[0]&&b===holdingArm[1])return 'upperArm';
      if(a===holdingArm[1]&&b===holdingArm[2])return 'forearm';
      if(handedness==='left'&&a>=23&&a<44||handedness==='right'&&a>=44)return 'forearm';
    };
    // The demo is a drawn mannequin, not a processed video.
    if(demo){
      ctx.lineCap='round';ctx.strokeStyle='rgba(218,227,211,.22)';ctx.lineWidth=Math.max(9,w*.021);
      bodyEdges.slice(0,12).forEach(([a,b])=>{const p=point(a),q=point(b);if(!valid(p)||!valid(q))return;ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();});
      const nose=point(0); if(valid(nose)){ctx.fillStyle='rgba(218,227,211,.25)';ctx.beginPath();ctx.ellipse(nose[0],nose[1],w*.019,w*.024,0,0,Math.PI*2);ctx.fill();}
      const left=point(15),right=point(16);
      if(valid(left)&&valid(right)){ctx.fillStyle='rgba(4,11,6,.23)';ctx.beginPath();ctx.ellipse((left[0]+right[0])/2,Math.max(left[1],right[1])+8,w*.1,h*.025,0,0,Math.PI*2);ctx.fill();}
    }
    ctx.lineCap='round'; ctx.lineJoin='round';
    allEdges.forEach(([a,b])=>{
      const p=point(a),q=point(b);if(!valid(p)||!valid(q))return;
      const region=edgeRegion(a,b),emphasized=region&&(focus==='all'||focus===region);
      ctx.lineWidth=colorByRegion?(emphasized?(a>=23?1.3:3.5):1.5):(a>=23?1.15:2);
      ctx.strokeStyle=colorByRegion?(region&&emphasized?REGION_COLORS[region]:'#54635c'):(a>=23?'rgba(194,238,120,.55)':'#c0ed7c');
      ctx.globalAlpha=colorByRegion && !emphasized ? .35 : 1;
      ctx.shadowColor=ctx.strokeStyle;ctx.shadowBlur=demo&&(!colorByRegion||emphasized)?5:0;ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();
    });
    ctx.shadowBlur=0;ctx.globalAlpha=1;
    for(let i=0;i<65;i++){const p=point(i);if(!valid(p))continue;const region=allEdges.map(([a,b])=>a===i||b===i?edgeRegion(a,b):undefined).find(r=>r&&(focus==='all'||focus===r));ctx.fillStyle=colorByRegion?(region?REGION_COLORS[region]:'#75857a'):(i<17?'#e0ffb8':'#b9e67d');ctx.beginPath();ctx.arc(p[0],p[1],i<17?3:1.4,0,Math.PI*2);ctx.fill();}
    if(colorByRegion&&holdingArm.length&&(focus==='all'||focus==='forearm')){
      ctx.strokeStyle=REGION_COLORS.forearm;ctx.globalAlpha=.4;ctx.lineWidth=1.5;ctx.setLineDash([4,4]);ctx.beginPath();let connected=false;
      let trailStart=frame;while(trailStart>0&&timestamps[frame]-timestamps[trailStart-1]<=.15)trailStart--;for(let f=trailStart;f<=frame;f++){const o=f*195+holdingArm[2]*3;const x=points[o],y=points[o+1];if(points[o+2]<.5||!Number.isFinite(x)||!Number.isFinite(y)||(f>0&&timestamps[f]-timestamps[f-1]>.05)){connected=false;continue;}if(connected)ctx.lineTo(ox+x*iw,oy+y*ih);else ctx.moveTo(ox+x*iw,oy+y*ih);connected=true;}ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
    }
    const event=analysis.swings.find(s=>Math.abs(s.impact.frame-frame)<=4);
    if(event&&event.handedness!=='unknown'){const wrist=point(event.handedness==='left'?9:10);if(valid(wrist)){ctx.strokeStyle='#f2db8c';ctx.lineWidth=1;ctx.beginPath();ctx.arc(wrist[0],wrist[1],17,0,Math.PI*2);ctx.stroke();ctx.fillStyle='#f2db8c';ctx.font='11px system-ui';ctx.fillText('击球时刻',wrist[0]+24,wrist[1]+4);}}
  },[video.width,video.height,video.captureFps,points,frame,overlay,demo,analysis.swings,colorByRegion,focus,handedness,timestamps]);

  useEffect(()=>{draw();const observer=new ResizeObserver(draw);if(canvasRef.current)observer.observe(canvasRef.current);return()=>observer.disconnect();},[draw]);
  useEffect(()=>{setPlaying(false);setVideoError(false);},[analysis.id]);
  useEffect(()=>{setPlaying(false);},[seekToken]);
  useEffect(()=>{
    if(!demo||!playing)return;
    let raf=0,last=performance.now(),current=timestamps[frameRef.current]||0;
    const tick=(now:number)=>{current+=(now-last)/1000*speed;last=now;if(current>=duration){onFrame(timestamps.length-1);setPlaying(false);return;}onFrame(closestFrame(timestamps,current));raf=requestAnimationFrame(tick);};
    raf=requestAnimationFrame(tick);return()=>cancelAnimationFrame(raf);
  },[playing,speed,demo,duration,timestamps,onFrame]);
  useEffect(()=>{
    const el=videoRef.current;if(demo||!el)return;
    el.playbackRate=speed;
    if(playing)void el.play().catch(()=>setPlaying(false));else el.pause();
  },[playing,speed,demo]);
  useEffect(()=>{
    const el=videoRef.current;if(demo||!el)return;
    let handle=0,raf=0,disposed=false;
    const update=(mediaTime:number)=>{if(!disposed&&playing)onFrame(closestFrame(timestamps,mediaTime));};
    if(typeof el.requestVideoFrameCallback === 'function'){
      const tick=(_now:number,metadata:VideoFrameCallbackMetadata)=>{update(metadata.mediaTime);handle=el.requestVideoFrameCallback(tick);};
      handle=el.requestVideoFrameCallback(tick);
    }else {const tick=()=>{update(el.currentTime);raf=requestAnimationFrame(tick);};raf=requestAnimationFrame(tick);}
    return()=>{disposed=true;if(handle)el.cancelVideoFrameCallback(handle);cancelAnimationFrame(raf);};
  },[demo,playing,timestamps,onFrame]);
  useEffect(()=>{
    const el=videoRef.current;if(!demo&&el&&!playing&&Math.abs(el.currentTime-time)>.001)el.currentTime=time;
  },[frame,time,playing,demo]);

  function seek(next:number){setPlaying(false);onFrame(Math.max(0,Math.min(timestamps.length-1,next)));}
  function togglePlay(){if(frame>=timestamps.length-1){onFrame(0);if(videoRef.current)videoRef.current.currentTime=0;}setPlaying(v=>!v);}
  return <section className="player-card">
    <div className="panel-heading"><span><Icon name="film" size={17}/>动作回放</span><span className="subtle">{video.width} × {video.height}<span className="dot-separator">·</span>{video.captureFps} fps</span></div>
    <div className="video-stage">
      {!demo&&<video ref={videoRef} src={video.proxyKey||video.storageKey} playsInline preload="auto" onEnded={()=>setPlaying(false)} onError={()=>setVideoError(true)}/>}
      <canvas ref={canvasRef} aria-label={demo?'合成演示骨架':'视频同步骨架叠加'}/>
      <div className="video-top"><span className="video-tag"><span className="live-dot"/>{demo?'合成演示':'目标球员已锁定'}</span><span className="video-tag monospace">{String(frame).padStart(4,'0')} FRAME</span></div>
      <div className="video-bottom"><span className="scene-caption">{demo?'合成骨架 · 非真实分析结果':'2D 全身骨架 · 采集时间同步'}</span><button className={`overlay-button ${overlay?'on':''}`} onClick={()=>setOverlay(v=>!v)} aria-pressed={overlay}><Icon name="bone" size={16}/>骨架叠加<span className="switch"><i/></span></button></div>
      {videoError&&<div className="video-error"><Icon name="info"/><p>视频无法播放，请检查代理视频是否生成。</p></div>}
    </div>
    <div className="player-controls">
      <div className="controls-left"><button className="play-button" aria-label={playing?'暂停':'播放'} onClick={togglePlay}><Icon name={playing?'pause':'play'} size={18}/></button><button className="icon-button" aria-label="上一帧" onClick={()=>seek(frame-1)}><Icon name="prev" size={17}/></button><button className="icon-button" aria-label="下一帧" onClick={()=>seek(frame+1)}><Icon name="next" size={17}/></button><span className="time-code">{time.toFixed(2)}<span> / {duration.toFixed(2)} s</span></span></div>
      <div className="controls-right"><select value={speed} onChange={e=>setSpeed(Number(e.target.value))} aria-label="播放速度"><option value={.25}>0.25× 慢放</option><option value={.5}>0.5× 慢放</option><option value={1}>1× 原速</option></select><button className="mark-button" onClick={()=>{setPlaying(false);onMark();}} disabled={marking}><Icon name="target" size={15}/><span>{marking?'保存中':'标记击球帧'}</span></button></div>
    </div>
    <div className="scrubber"><input type="range" min={0} max={Math.max(0,timestamps.length-1)} value={frame} onChange={e=>seek(Number(e.target.value))} aria-label="逐帧时间轴" style={{'--progress':`${frame/Math.max(1,timestamps.length-1)*100}%`} as React.CSSProperties}/></div>
  </section>;
}
