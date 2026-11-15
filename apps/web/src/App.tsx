import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Analysis, MetricResult, SwingEvent } from '@badminton/schema';
import { api, json, type HistoryItem, type Result } from './api';
import Icon from './Icon';
import Player from './Player';
import MovementInsight, {type BodyFocus} from './MovementInsight';
import { useDialog } from './useDialog';
import Upload from './Upload';

const names: Record<string,string>={elbow_angle_impact:'击球肘角',impact_height_ratio:'击球点高度',impact_forward_offset:'击球点前后位置',accel_duration:'引拍至击球',followthrough_duration:'随挥时长'};
const reasons: Record<string,string>={low_fps:'采集帧率不足 60 fps，暂不显示时序指标',bad_view:'当前视角不适合可靠测量此项指标',low_conf:'相关关键点或击球事件置信度不足'};
const views: Record<string,string>={side:'侧面',back:'背面',front:'正面',oblique:'斜侧',unknown:'未知视角'};
const statuses: Record<string,string>={queued:'排队中',running:'分析中',succeeded:'分析完成',failed:'分析失败',rejected_quality:'未通过质检'};
const issueLabels: Record<string,string>={synthetic_demo:'当前为合成演示数据',partial_occlusion:'第三次挥拍模拟局部遮挡，部分指标已隐藏',no_audio:'未检测到音轨，使用视觉击球检测',low_fps:'帧率较低，时序指标将隐藏',low_visibility:'身体遮挡较多，部分指标不可靠',multi_person:'检测到多人，请确认目标球员',dark:'光线偏暗，建议在明亮环境重新拍摄',time_remapped:'视频可能经过慢动作时间重映射',capture_time_uncertain:'无法确认真实采集时间，请提供原始视频',no_swing:'未检测到可靠的挥拍事件',model_unavailable:'姿态模型尚未配置，请完成 Worker 模型设置'};

function Confidence({value}:{value:number}){return <span className={`confidence ${value<.7?'muted-confidence':''}`}><i/>{Math.round(value*100)}% 置信度</span>;}

function MetricCard({metric,id}:{metric?:MetricResult;id:string}){
  const suppressed=!metric||metric.value===null;
  const description:Record<string,string>={elbow_angle_impact:'肩 — 肘 — 腕的二维夹角',impact_height_ratio:'以自身躯干长度为参考',impact_forward_offset:'正值表示位于身体前方',accel_duration:'引拍起点 → 击球时刻',followthrough_duration:'击球时刻 → 随挥结束'};
  return <div className={`metric-card ${suppressed?'suppressed':''}`}><div className="metric-title"><span>{names[id]}</span><span title={description[id]}><Icon name="info" size={14}/></span></div><div className="metric-value">{suppressed?'—':metric.value!.toFixed(id==='elbow_angle_impact'?1:2)}<span>{suppressed?'未显示':metric.unit==='deg'?'°':metric.unit==='sec'?'s':'× 躯干'}</span></div>{metric&&!suppressed?<Confidence value={metric.confidence}/>:<span className="suppressed-label"><Icon name="eye" size={13}/>{metric?.suppressed?.reason==='bad_view'?'视角受限':metric?.suppressed?.reason==='low_fps'?'帧率不足':'数据不足'}</span>}<p>{suppressed?(metric?.suppressed?reasons[metric.suppressed.reason]:'尚无可靠指标'):description[id]}</p></div>;
}

function ElbowChart({points,timestamps,event,frame,width,height}:{points:Float32Array;timestamps:Float32Array;event?:SwingEvent;frame:number;width:number;height:number}){
  const chart=useMemo(()=>{
    if(!event||event.handedness==='unknown')return {path:'',start:0,end:1};
    const start=Math.max(0,event.impact.tSec-.65),end=Math.min(timestamps[timestamps.length-1],event.impact.tSec+.65);
    const side=event.handedness==='left'?[5,7,9]:[6,8,10];
    let path='',connected=false;
    for(let f=0;f<timestamps.length;f++){
      const time=timestamps[f];if(time<start||time>end)continue;
      const p=side.map(k=>({x:points[f*195+k*3]*width,y:points[f*195+k*3+1]*height,c:points[f*195+k*3+2]}));
      if(p.some(v=>!Number.isFinite(v.x)||!Number.isFinite(v.y)||v.c<.5)){connected=false;continue;}
      const ax=p[0].x-p[1].x,ay=p[0].y-p[1].y,bx=p[2].x-p[1].x,by=p[2].y-p[1].y;
      const denominator=Math.hypot(ax,ay)*Math.hypot(bx,by);if(denominator<1e-6){connected=false;continue;}
      const angle=Math.acos(Math.max(-1,Math.min(1,(ax*bx+ay*by)/denominator)))*180/Math.PI;
      const x=38+(time-start)/(end-start)*610,y=128-angle/180*100;
      path+=`${connected?'L':'M'}${x.toFixed(2)},${y.toFixed(2)} `;connected=true;
    }
    return {path,start,end};
  },[event,points,timestamps,width,height]);
  const x=(time:number)=>38+(time-chart.start)/(chart.end-chart.start)*610;
  const current=timestamps[frame]||0;
  return <div className="chart-panel"><div className="panel-heading"><span>肘角变化<span className="small-subtitle">击球前后 650 ms</span></span><span className="chart-legend"><i/>持拍臂肘角</span></div><svg className="angle-chart" viewBox="0 0 680 162" role="img" aria-label="击球附近持拍臂肘角变化曲线"><defs><linearGradient id="chartFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#c5ef8a" stopOpacity=".13"/><stop offset="100%" stopColor="#c5ef8a" stopOpacity="0"/></linearGradient></defs>{[60,120,180].map(v=><g key={v}><line x1="38" y1={128-v/180*100} x2="648" y2={128-v/180*100} stroke="#2a322d" strokeDasharray="3 5"/><text x="0" y={132-v/180*100} fill="#77847b" fontSize="10">{v}°</text></g>)}{event&&<><rect x={x(event.impact.tSec)-20} y="20" width="40" height="108" fill="#c5ef8a" opacity=".05"/><line x1={x(event.impact.tSec)} y1="16" x2={x(event.impact.tSec)} y2="130" stroke="#a4b18b" strokeDasharray="3 4"/><text x={x(event.impact.tSec)+7} y="13" fill="#b6c79b" fontSize="10">击球</text></>}<path d={chart.path} fill="none" stroke="#c5ef8a" strokeWidth="2" strokeLinejoin="round"/>{current>=chart.start&&current<=chart.end&&<line x1={x(current)} x2={x(current)} y1="22" y2="128" stroke="#e7ebe0" opacity=".5"/>}{[chart.start,(chart.start+chart.end)/2,chart.end].map((v,i)=><text key={i} x={38+i*305} y="151" textAnchor={i===0?'start':i===2?'end':'middle'} fill="#77847b" fontSize="10">{v.toFixed(2)} s</text>)}{!chart.path&&<text x="340" y="85" textAnchor="middle" fill="#8e998f" fontSize="12">暂无可靠角度曲线</text>}</svg></div>;
}

function CaptureGuide({onClose}:{onClose:()=>void}){const dialog=useDialog(onClose);return <div className="modal-backdrop" onMouseDown={e=>{if(e.target===e.currentTarget)onClose();}}><section ref={dialog} className="modal guide-modal" role="dialog" aria-modal="true" aria-labelledby="guide-title"><div className="modal-header"><div><span className="eyebrow">CAPTURE GUIDE</span><h2 id="guide-title">拍得清楚，分析更可靠</h2></div><button className="icon-button" aria-label="关闭拍摄指南" onClick={onClose}><Icon name="close"/></button></div><div className="guide-grid">{[{n:'01',title:'从侧面或斜侧拍摄',text:'固定手机，保证完整身体、手臂和球拍活动范围入镜。多人练习时，上传后点选目标球员。',icon:'camera'},{n:'02',title:'优先选择 120 fps',text:'使用原始视频，避免编辑软件再次变速。60 fps 可显示时序指标，低帧率将隐藏不可靠的时长。',icon:'film'},{n:'03',title:'明亮场地，保留声音',text:'声音能帮助定位击球窗口。无音轨也可分析，但击球置信度会降低。',icon:'eye'},{n:'04',title:'每次保留完整挥拍',text:'建议上传 5–15 秒片段，包含引拍、击球和随挥。目前支持杀球与高远球。',icon:'activity'}].map(item=><div key={item.n}><span className="guide-number">{item.n}</span><Icon name={item.icon} size={27}/><h3>{item.title}</h3><p>{item.text}</p></div>)}</div><div className="modal-footer"><span>当前版本仅展示二维数值与趋势，不评判动作好坏。</span><button className="primary-button" onClick={onClose}>了解了<Icon name="check" size={17}/></button></div></section></div>;}

export default function App(){
  const [result,setResult]=useState<Result|null>(null),[timestamps,setTimestamps]=useState(new Float32Array()),[points,setPoints]=useState(new Float32Array());
  const [seekToken,setSeekToken]=useState(0);
  const [analysisView,setAnalysisView]=useState<'movement'|'metrics'>('movement');
  const [bodyFocus,setBodyFocus]=useState<BodyFocus>('all');
  const [frame,setFrame]=useState(0),[selected,setSelected]=useState(''),[loading,setLoading]=useState(true),[error,setError]=useState('');
  const [page,setPage]=useState<'analysis'|'history'>('analysis'),[upload,setUpload]=useState(false),[guide,setGuide]=useState(false),[history,setHistory]=useState<HistoryItem[]>([]);
  const [jobId,setJobId]=useState<string|null>(null),[marking,setMarking]=useState(false),[toast,setToast]=useState(''),[menu,setMenu]=useState(false),[demo,setDemo]=useState(true);
  const generation=useRef(0);
  const navRef=useRef<HTMLElement>(null);
  useEffect(()=>{
    const media=window.matchMedia('(max-width: 720px)');
    const resize=()=>setMenu(false);media.addEventListener('change',resize);
    return()=>media.removeEventListener('change',resize);
  },[]);
  useEffect(()=>{
    if(!menu)return;const previous=document.activeElement as HTMLElement|null;const oldOverflow=document.body.style.overflow;document.body.style.overflow='hidden';
    navRef.current?.querySelector<HTMLButtonElement>('.mobile-nav-close')?.focus();
    const keydown=(e:KeyboardEvent)=>{if(e.key==='Escape'){e.preventDefault();setMenu(false);}if(e.key==='Tab'){const elements=[...(navRef.current?.querySelectorAll<HTMLElement>('a[href],button:not(:disabled)')||[])].filter(el=>el.getClientRects().length>0);const first=elements[0],last=elements[elements.length-1];if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus();}}};
    document.addEventListener('keydown',keydown);return()=>{document.body.style.overflow=oldOverflow;document.removeEventListener('keydown',keydown);previous?.focus();};
  },[menu]);
  const setCurrentFrame=useCallback((f:number)=>setFrame(f),[]);
  const loadResult=useCallback(async(data:Result,isDemo:boolean)=>{
    const current=++generation.current;setLoading(true);setResult(data);setDemo(isDemo);setError('');setPoints(new Float32Array());setTimestamps(new Float32Array());
    try{
      const ref=data.analysis.skeleton;
      if(ref){
        const buffers=await Promise.all([ref.timestampsKey,ref.keypointsKey].map(async key=>{const response=await fetch(key);if(!response.ok)throw new Error('骨架数据读取失败，请检查分析产物。');return response.arrayBuffer();}));
        if(generation.current!==current)return;
        const ts=new Float32Array(buffers[0]),kp=new Float32Array(buffers[1]);
        if(ts.length!==ref.frameCount||kp.length!==ref.frameCount*195)throw new Error('骨架文件长度与清单不匹配。');
        setTimestamps(ts);setPoints(kp);
      }
      if(generation.current!==current)return;
      setSelected(data.analysis.swings[0]?.id||'');setFrame(data.analysis.swings[0]?.impact.frame||0);
    }catch(e){if(generation.current===current)setError((e as Error).message);}finally{if(generation.current===current)setLoading(false);}
  },[]);
  const loadDemo=useCallback(async()=>{const request=++generation.current;setJobId(null);setLoading(true);setPage('analysis');try{const data=await api<Result>('/api/demo');if(request===generation.current)await loadResult(data,true);}catch(e){if(request===generation.current){setError(`无法加载示例：${(e as Error).message}`);setLoading(false);}}},[loadResult]);
  useEffect(()=>{void loadDemo();},[loadDemo]);
  useEffect(()=>{if(!toast)return;const timeout=setTimeout(()=>setToast(''),4000);return()=>clearTimeout(timeout);},[toast]);
  useEffect(()=>{if(page==='history')void api<HistoryItem[]>('/api/analyses').then(setHistory).catch(e=>setToast(e.message));},[page]);
  useEffect(()=>{
    if(!jobId)return;let disposed=false,timer:ReturnType<typeof setTimeout>;
    const poll=async()=>{try{
      const data=await api<Result>(`/api/analyses/${jobId}`);if(disposed)return;
      setResult(data);setDemo(false);setError('');
      if(['succeeded','failed','rejected_quality'].includes(data.analysis.status)){setJobId(null);await loadResult(data,false);return;}
      timer=setTimeout(poll,1200);
    }catch(e){if(!disposed){setError((e as Error).message);timer=setTimeout(poll,3000);}}};
    void poll();return()=>{disposed=true;clearTimeout(timer);};
  },[jobId,loadResult]);
  const analysis=result?.analysis,video=result?.video;
  const event=analysis?.swings.find(s=>s.id===selected)||analysis?.swings[0];
  const selectedIndex=analysis?.swings.findIndex(s=>s.id===event?.id)||0;
  const metrics=analysis?.metrics.filter(m=>m.swingId===event?.id)||[];
  const isProcessing=!!jobId||analysis?.status==='queued'||analysis?.status==='running';
  const isFailed=analysis?.status==='failed'||analysis?.status==='rejected_quality';
  const usable=!!(analysis&&video&&points.length&&timestamps.length&&!isProcessing&&!isFailed);
  function selectEvent(s:SwingEvent){setSelected(s.id);setFrame(s.impact.frame);setSeekToken(n=>n+1);}
  function seekMovement(next:number){setFrame(next);setSeekToken(n=>n+1);}
  async function mark(){
    if(!analysis)return;
    if(demo){setToast(`演示标记：第 ${frame} 帧（${(timestamps[frame]||0).toFixed(3)} s），未保存。上传视频后可保存真实标注。`);return;}
    setMarking(true);try{const data=await api<Result>(`/api/analyses/${analysis.id}/annotations`,json({frame}));await loadResult(data,false);setToast('击球帧标注已保存，指标已重新计算。');}catch(e){setToast((e as Error).message);}finally{setMarking(false);}
  }
  async function openHistory(item:HistoryItem){const request=++generation.current;setJobId(null);setPage('analysis');setLoading(true);setError('');if(item.status==='queued'||item.status==='running'){setJobId(item.id);return;}try{const data=await api<Result>(`/api/analyses/${item.id}`);if(request===generation.current)await loadResult(data,false);}catch(e){if(request===generation.current){setError((e as Error).message);setLoading(false);}}}
  async function retry(){if(!analysis)return;try{await api(`/api/analyses/${analysis.id}/retry`,json({}));setError('');setJobId(analysis.id);}catch(e){setToast((e as Error).message);}}
  return <div className="app-shell">
    <button className={`nav-scrim ${menu?'visible':''}`} aria-label="关闭导航" tabIndex={-1} onClick={()=>setMenu(false)}/>
    <aside id="main-navigation" ref={navRef} className={`sidebar ${menu?'mobile-open':''}`} aria-label="主导航">
      <button className="icon-button mobile-nav-close" aria-label="关闭导航菜单" onClick={()=>setMenu(false)}><Icon name="close"/></button>
      <a className="brand" href="#" onClick={e=>{e.preventDefault();setPage('analysis');setMenu(false);}}><span className="brand-mark"><svg width="29" height="29" viewBox="0 0 32 32" fill="none"><path d="m8 5 13 5-6 13L8 5Z" fill="currentColor"/><path d="m14 3 13 7-8 14M5 9l8 18m2-4-4 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg></span><span>羽析<span>SHUTTLE LAB</span></span></a>
      <div className="nav-group-label">训练工作台</div><nav><button className={page==='analysis'?'active':''} onClick={()=>{setPage('analysis');setMenu(false);}}><Icon name="grid" size={18}/><span>动作分析</span><span className="nav-active-dot"/></button><button className={page==='history'?'active':''} onClick={()=>{setPage('history');setMenu(false);}}><Icon name="folder" size={18}/><span>我的视频</span></button><button onClick={()=>{setGuide(true);setMenu(false);}}><Icon name="book" size={18}/><span>拍摄指南</span><Icon name="chevron" size={13}/></button></nav>
      <div className="sidebar-note"><span className="note-orbit"><Icon name="activity" size={24}/></span><strong>每一次挥拍，都有迹可循。</strong><p>让动作变得可见，<br/>让训练更加清晰。</p><button onClick={()=>{setMenu(false);setGuide(true);}}>开始之前，了解拍摄建议<Icon name="arrow" size={14}/></button></div>
      <div className="sidebar-footer"><span className="local-indicator"/><span>本地分析工作台</span><span className="version">P0a</span></div>
    </aside>
    <main className="main">
      <header className="topbar"><div className="breadcrumb"><button className="icon-button mobile-menu" aria-label={menu?'收起导航':'展开导航'} aria-expanded={menu} aria-controls="main-navigation" onClick={()=>setMenu(v=>!v)}><Icon name="menu"/></button><span>训练工作台</span><Icon name="chevron" size={13}/><strong>{page==='history'?'我的视频':'动作分析'}</strong></div><div className="topbar-right"><span className="beta-tag">P0a · Preview</span><span className="avatar">羽</span></div></header>
      <div className="workspace"><div className="page-heading"><div><div className="eyebrow">{page==='analysis'?'SEE YOUR SWING. UNDERSTAND YOUR GAME.':'YOUR TRAINING LIBRARY'}</div><h1>{page==='analysis'?'羽毛球动作分析':'我的视频'}</h1><p>{page==='analysis'?'从每一帧，发现动作的细节。':'保存每一次分析，回看每一次挥拍。'}</p></div><button className="primary-button" onClick={()=>setUpload(true)}><Icon name="upload" size={17}/>上传新视频</button></div>
      {page==='history'?<section className="history-panel"><div className="panel-heading"><span>分析记录</span><button className="text-button" onClick={()=>void api<HistoryItem[]>('/api/analyses').then(setHistory).catch(e=>setToast(e.message))}><Icon name="refresh" size={15}/>刷新</button></div>{history.length?history.map(item=><button className="history-row" key={item.id} onClick={()=>void openHistory(item)}><span className="history-icon"><Icon name="film" size={23}/></span><span><strong>{item.fileName||`分析 ${item.id.slice(0,8)}`}</strong><small>{item.createdAt?new Date(item.createdAt).toLocaleString('zh-CN'):item.id}</small></span><span className={`history-status ${item.status}`}>{statuses[item.status]}</span><Icon name="chevron" size={17}/></button>):<div className="empty-state"><Icon name="folder" size={36}/><h3>还没有分析记录</h3><p>上传你的第一段挥拍视频，开始积累训练记录。</p><button className="primary-button" onClick={()=>setUpload(true)}>上传视频<Icon name="arrow" size={16}/></button></div>}</section>:<>
      {demo&&!isProcessing&&<div className="demo-banner"><span className="demo-badge"><Icon name="bolt" size={13}/>演示数据</span><p>先探索一份分析示例。骨架与指标为合成数据，不代表真实运动表现。</p><button onClick={()=>setUpload(true)}>分析我的视频<Icon name="arrow" size={15}/></button></div>}
      {error&&<div className="error-message" role="alert"><Icon name="info"/>{error}<button className="text-button" onClick={()=>void loadDemo()}>重新加载示例</button></div>}
      {isProcessing?<div className="processing-panel"><span className="processing-orbit"><Icon name="activity" size={39}/></span><span className="eyebrow">ANALYSIS IN PROGRESS</span><h2>{analysis?.status==='queued'?'视频已就绪，等待分析':'正在解析你的挥拍动作'}</h2><p>从视频中提取真实时间、追踪球员、计算姿态与击球事件。</p><div className="processing-steps">{['摄取视频','目标追踪','姿态提取','击球检测','指标计算'].map((name,i)=><span key={name}><i>{i+1}</i>{name}</span>)}</div><div className="indeterminate-progress"/><small>任务在后台运行，你可以前往「我的视频」查看进度。</small></div>:isFailed?<div className="failure-panel"><Icon name="info" size={36}/><h2>{statuses[analysis!.status]}</h2><p>{result?.error||'当前视频无法生成可靠结果，请根据质检提示调整后重试。'}</p>{analysis?.quality?.issues.map((issue,i)=><div className="failure-issue" key={`${issue.code}-${i}`}>{issueLabels[issue.code]||issue.code}</div>)}<div className="failure-actions"><button className="secondary-button" onClick={()=>void loadDemo()}>查看演示</button>{analysis?.status==='failed'&&<button className="secondary-button" onClick={()=>void retry()}><Icon name="refresh" size={16}/>重试分析</button>}<button className="primary-button" onClick={()=>setUpload(true)}>重新上传</button></div></div>:loading?<div className="loading-state"><span className="spinner"/>正在加载分析数据…</div>:usable&&analysis&&video?<>
      <div className="session-meta"><div className="session-name"><span className="session-icon"><Icon name="film" size={16}/></span><strong>{demo?'杀球 · 侧面机位示例':`${analysis.userInput.swingType==='clear'?'高远球':'杀球'}动作分析`}</strong><span className="session-complete"><Icon name="check" size={12}/>分析完成</span></div><div className="session-stats"><span><Icon name="clock" size={14}/>{video.durationSec.toFixed(1)} 秒</span><span><Icon name="activity" size={14}/>{analysis.swings.length} 次击球</span><span>{views[analysis.quality?.view||'unknown']}机位</span></div></div>
      <div className="analysis-view-bar"><div className="analysis-view-tabs" aria-label="分析视图"><button className={analysisView==='movement'?'selected':''} aria-pressed={analysisView==='movement'} onClick={()=>setAnalysisView('movement')}>身体协同</button><button className={analysisView==='metrics'?'selected':''} aria-pressed={analysisView==='metrics'} onClick={()=>setAnalysisView('metrics')}>基础指标</button></div><label className="movement-event-selector">当前挥拍<select aria-label="选择挥拍" value={event?.id??''} onChange={e=>{const s=analysis.swings.find(x=>x.id===e.target.value);if(s)selectEvent(s);}}>{analysis.swings.map((s,i)=><option key={s.id} value={s.id}>第 {i+1} 拍 · {s.impact.tSec.toFixed(2)} s</option>)}</select></label></div>
      {analysisView==='movement'&&event?<MovementInsight video={video} analysis={analysis} event={event} points={points} timestamps={timestamps} frame={frame} onSeek={seekMovement} onFrame={setCurrentFrame} seekToken={seekToken} demo={demo} focus={bodyFocus} onFocus={setBodyFocus}><Player key={analysis.id} video={video} analysis={analysis} points={points} timestamps={timestamps} frame={frame} onFrame={setCurrentFrame} seekToken={seekToken} demo={demo} onMark={()=>void mark()} marking={marking} colorByRegion focus={bodyFocus} handedness={event.handedness}/></MovementInsight>:<div className="analysis-grid"><div className="analysis-left"><Player key={analysis.id} video={video} analysis={analysis} points={points} timestamps={timestamps} frame={frame} onFrame={setCurrentFrame} seekToken={seekToken} demo={demo} onMark={()=>void mark()} marking={marking}/>
        <section className="timeline-panel"><div className="panel-heading"><span>击球时间轴<span className="count-pill">{analysis.swings.length}</span></span><span className="subtle">选择一次击球，查看动作指标</span></div><div className="event-track"><div className="event-track-line"/>{analysis.swings.map((s,i)=><button key={s.id} title={`击球 ${i+1} · ${s.impact.tSec.toFixed(2)} 秒`} className={`track-marker ${s.id===event?.id?'selected':''}`} style={{left:`${Math.max(3,Math.min(97,s.impact.tSec/video.durationSec*100))}%`}} onClick={()=>selectEvent(s)}><span>{String(i+1).padStart(2,'0')}</span><i/></button>)}<span className="track-start">0.00 s</span><span className="track-end">{video.durationSec.toFixed(2)} s</span></div><div className="swing-tabs">{analysis.swings.map((s,i)=><button key={s.id} onClick={()=>selectEvent(s)} className={s.id===event?.id?'selected':''}><span className="swing-number">{String(i+1).padStart(2,'0')}</span><span><strong>{s.type==='clear'?'高远球':'杀球'}</strong><small>{s.impact.tSec.toFixed(2)} s · 第 {s.impact.frame} 帧</small></span><Icon name="chevron" size={14}/></button>)}</div>{!analysis.swings.length&&<p className="no-swings">未检测到可靠击球事件，可使用「标记击球帧」添加标注。</p>}</section>
        <ElbowChart points={points} timestamps={timestamps} event={metrics.find(m=>m.id==='elbow_angle_impact')?.value===null?undefined:event} frame={frame} width={video.width} height={video.height}/>
      </div><aside className="metrics-panel"><div className="metrics-header"><div><span className="eyebrow">SWING {String(selectedIndex+1).padStart(2,'0')}</span><h2>这一拍，发生了什么</h2></div><span className="round-target"><Icon name="target" size={20}/></span></div><div className="impact-summary"><span><i/>击球时刻</span><strong>{event?.impact.tSec.toFixed(3)||'—'}<small> s</small></strong><div>{event&&<Confidence value={event.impact.confidence}/>}<span className="evidence-label">{event?.impact.evidence.includes('manual')?'人工标注':event?.impact.evidence.includes('audio')?'声音 + 视觉':'视觉检测'}</span></div></div><div className="metric-grid">{['elbow_angle_impact','impact_height_ratio','impact_forward_offset'].map(id=><MetricCard id={id} key={id} metric={metrics.find(m=>m.id===id)}/>)}</div><div className="timing-heading"><Icon name="clock" size={15}/>挥拍时序</div><div className="timing-grid">{['accel_duration','followthrough_duration'].map(id=><MetricCard id={id} key={id} metric={metrics.find(m=>m.id===id)}/>)}</div><div className="metric-note"><Icon name="shield" size={16}/><p>只展示可靠的测量结果。<br/>当前为二维指标，不代表动作好坏。</p></div></aside></div>}
      <div className="quality-strip"><span className="quality-title"><Icon name="shield" size={17}/>视频质量</span><span><i className={video.captureFps>=60?'ok-dot':'warn-dot'}/>{video.captureFps} fps 采集</span><span><i className={(analysis.quality?.bodyVisibleRatio||0)>=.9?'ok-dot':'warn-dot'}/>全身可见 {Math.round((analysis.quality?.bodyVisibleRatio||0)*100)}%</span><span><i className={video.hasAudio?'ok-dot':'warn-dot'}/>{video.hasAudio?'含音频':'无音频 · 视觉回退'}</span><span><i className={analysis.target?'ok-dot':'warn-dot'}/>{analysis.target?.lockedBy==='user'?'手动锁定球员':'自动追踪球员'}</span><button onClick={()=>setGuide(true)}>拍摄建议<Icon name="arrow" size={14}/></button></div>
      {!!analysis.quality?.issues.length&&<div className="quality-issues">{analysis.quality.issues.map((issue,i)=><span key={`${issue.code}-${i}`}><Icon name="info" size={13}/>{issueLabels[issue.code]||issue.code}</span>)}</div>}
      <footer className="workspace-footer"><span>以采集时间为基准 · 二维归一化坐标 · 置信度优先</span><span>Schema {analysis.schemaVersion}<span className="dot-separator">·</span>{analysis.pipelineVersion}</span></footer>
      </>:null}</>}
      </div>
    </main>
    {upload&&<Upload onClose={()=>setUpload(false)} onCreated={id=>{generation.current++;setUpload(false);setPage('analysis');setDemo(false);setError('');setLoading(false);setJobId(id);}}/>}
    {guide&&<CaptureGuide onClose={()=>setGuide(false)}/>}
    {toast&&<div className="toast" role="status"><Icon name="info" size={17}/><span>{toast}</span><button className="icon-button" aria-label="关闭提示" onClick={()=>setToast('')}><Icon name="close" size={15}/></button></div>}
  </div>;
}
