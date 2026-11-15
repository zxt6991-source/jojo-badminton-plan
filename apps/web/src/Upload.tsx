import { useEffect, useRef, useState } from 'react';
import Icon from './Icon';
import { api, json, uploadVideo } from './api';
import { useDialog } from './useDialog';

export default function Upload({ onClose, onCreated }: { onClose: () => void; onCreated: (id: string) => void }) {
  const [file,setFile]=useState<File|null>(null), [preview,setPreview]=useState('');
  const [swingType,setSwingType]=useState('smash'),[target,setTarget]=useState<[number,number]|undefined>();
  const [captureFps,setCaptureFps]=useState(''),[progress,setProgress]=useState<number|null>(null),[error,setError]=useState(''),[drag,setDrag]=useState(false);
  const input=useRef<HTMLInputElement>(null), video=useRef<HTMLVideoElement>(null), abort=useRef<AbortController|null>(null);
  const dialog=useDialog(()=>{abort.current?.abort();onClose();});
  const busy=progress!==null;
  useEffect(()=>{if(!file){setPreview('');return;}const url=URL.createObjectURL(file);setPreview(url);return()=>URL.revokeObjectURL(url);},[file]);
  useEffect(()=>()=>abort.current?.abort(),[]);
  function choose(candidate?:File){
    if(!candidate)return;
    if(!/\.(mp4|mov|webm|m4v)$/i.test(candidate.name)){setError('请选择 MP4、MOV、M4V 或 WebM 视频。');return;}
    if(candidate.size===0){setError('视频文件为空，请重新选择。');return;}
    if(candidate.size>500*1024*1024){setError('单个视频请控制在 500 MB 以内。');return;}
    setFile(candidate);setTarget(undefined);setError('');
  }
  async function start(){
    if(!file||busy)return;setError('');setProgress(0);abort.current=new AbortController();
    try{
      if(captureFps&&(!Number.isFinite(Number(captureFps))||Number(captureFps)<15||Number(captureFps)>480))throw new Error('真实采集帧率应在 15–480 fps 之间。');
      if(video.current&&Number.isFinite(video.current.duration)&&video.current.duration>60)throw new Error('视频最长支持 60 秒，请先裁剪为较短片段。');
      const uploaded=await uploadVideo(file,setProgress,abort.current.signal);
      const created=await api<{id:string}>('/api/analyses', {...json({ videoId:uploaded.videoId,swingType,targetPoint:target,...(captureFps?{captureFps:Number(captureFps)}:{}) }),signal:abort.current.signal});
      onCreated(created.id);
    }catch(e){if((e as Error).name!=='AbortError')setError((e as Error).message);setProgress(null);}
  }
  return <div className="modal-backdrop" onMouseDown={e=>{if(e.target===e.currentTarget&&!busy)onClose();}}>
    <section ref={dialog} className="modal upload-modal" role="dialog" aria-modal="true" aria-labelledby="upload-title">
      <div className="modal-header"><div><span className="eyebrow">NEW ANALYSIS</span><h2 id="upload-title">看清每一次挥拍</h2><p>上传一段视频，从击球时刻开始理解动作。</p></div><button className="icon-button" aria-label="关闭上传窗口" onClick={()=>{abort.current?.abort();onClose();}}><Icon name="close"/></button></div>
      <div className="upload-content">
        <input ref={input} type="file" accept=".mp4,.mov,.m4v,.webm,video/mp4,video/quicktime,video/webm" onChange={e=>choose(e.target.files?.[0])} hidden/>
        {!file?<button className={`drop-zone ${drag?'dragging':''}`} onClick={()=>input.current?.click()} onDragOver={e=>{e.preventDefault();setDrag(true);}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);choose(e.dataTransfer.files[0]);}}><span className="upload-symbol"><Icon name="upload" size={28}/></span><strong>拖放视频到这里，或点击选择</strong><span>MP4 / MOV / WebM · 最大 500 MB · 最长 60 秒</span><span className="upload-advice">建议 5–15 秒，完整拍到身体与挥拍范围</span></button>:<div className="preview-container">
          <div className="upload-preview" role="button" tabIndex={0} aria-label="选择目标球员，回车选择中心，方向键移动目标，删除键恢复自动选择" onKeyDown={e=>{if(busy)return;const key=e.key;if(['Enter',' ','ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Delete','Backspace'].includes(key)){e.preventDefault();if(key==='Delete'||key==='Backspace'){setTarget(undefined);return;}const [x,y]=target||[.5,.5];setTarget([Math.max(0,Math.min(1,x+(key==='ArrowLeft'?-.02:key==='ArrowRight'?.02:0))),Math.max(0,Math.min(1,y+(key==='ArrowUp'?-.02:key==='ArrowDown'?.02:0)))]);}}} onClick={e=>{
            if(busy||!video.current)return;const el=video.current;const rect=el.getBoundingClientRect();
            const scale=Math.min(rect.width/(el.videoWidth||1),rect.height/(el.videoHeight||1));
            const w=el.videoWidth*scale,h=el.videoHeight*scale,x=(e.clientX-rect.left-(rect.width-w)/2)/w,y=(e.clientY-rect.top-(rect.height-h)/2)/h;
            if(x>=0&&x<=1&&y>=0&&y<=1)setTarget([x,y]);
          }}>
            <video ref={video} src={preview} preload="metadata" playsInline onLoadedMetadata={()=>{if(video.current)video.current.currentTime=0;}}/>
            {target&&<div className="target-indicator" style={(()=>{const el=video.current;if(!el)return{};const scale=Math.min(el.clientWidth/el.videoWidth,el.clientHeight/el.videoHeight);return {left:(el.clientWidth-el.videoWidth*scale)/2+target[0]*el.videoWidth*scale,top:(el.clientHeight-el.videoHeight*scale)/2+target[1]*el.videoHeight*scale};})()}><Icon name="target" size={28}/></div>}
            <span className="preview-hint"><Icon name="target" size={14}/>{target?'已选择目标球员 · 点击可重新选择':'点击画面中的目标球员（可选）'}</span>
          </div>
          <div className="selected-file"><Icon name="film"/><div><strong>{file.name}</strong><span>{(file.size/1024/1024).toFixed(1)} MB{target?' · 手动锁定球员':' · 自动锁定主要球员'}</span></div><button className="text-button" onClick={()=>input.current?.click()} disabled={busy}>更换</button></div>
        </div>}
        <div className="upload-options"><fieldset disabled={busy}><legend>动作类型</legend><div className="choice-row"><button type="button" className={swingType==='smash'?'selected':''} onClick={()=>setSwingType('smash')}><Icon name="bolt" size={17}/>杀球{swingType==='smash'&&<Icon name="check" size={14}/>}</button><button type="button" className={swingType==='clear'?'selected':''} onClick={()=>setSwingType('clear')}><Icon name="arrow" size={17}/>高远球{swingType==='clear'&&<Icon name="check" size={14}/>}</button></div></fieldset><label className="fps-input">真实采集帧率 <span>可选</span><input type="number" min="15" max="480" placeholder="自动读取" value={captureFps} disabled={busy} onChange={e=>setCaptureFps(e.target.value)}/></label></div>
        <p className="field-note"><Icon name="info" size={15}/>仅在确认手机的原始拍摄帧率时填写。慢动作视频的播放帧率不等于采集帧率。</p>
        <div className="capture-tips"><span><Icon name="check" size={14}/>侧面或斜侧拍摄</span><span><Icon name="check" size={14}/>全身完整入镜</span><span><Icon name="check" size={14}/>推荐 120 fps</span></div>
        {error&&<div className="error-message" role="alert"><Icon name="info" size={17}/>{error}</div>}
        {busy&&<div className="upload-progress" role="status"><div><span>{progress===100?'视频已上传，正在创建分析…':'正在上传 · 中断后可重新选择同一文件续传'}</span><strong>{progress}%</strong></div><div className="progress-track"><i style={{width:`${progress}%`}}/></div></div>}
      </div>
      <div className="modal-footer"><span><Icon name="shield" size={15}/>请使用已获得拍摄者授权的视频</span><button className="primary-button" disabled={!file||busy} onClick={start}>{busy?<span className="spinner"/>:<Icon name="activity" size={17}/>}开始分析<Icon name="arrow" size={16}/></button></div>
    </section>
  </div>;
}
