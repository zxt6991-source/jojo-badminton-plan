import express from 'express';
import type {Request,Response,NextFunction} from 'express';
import {createHash,randomUUID} from 'node:crypto';
import {mkdir,readFile,writeFile,rename,stat,open} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawn} from 'node:child_process';
import type {Analysis,VideoAsset} from '@badminton/schema';
import {SCHEMA_VERSION,parseAnalysis,parseVideoAsset,validateSkeletonData} from '@badminton/schema';
import {calculateMetrics} from '@badminton/metrics';
import {Store} from './store.js';
import {resolveWorkerPython,workerFailureMessage} from './worker-runtime.js';

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../..');
const PIPELINE_VERSION=JSON.parse(await readFile(path.join(ROOT,'pipeline-version.json'),'utf8')).pipelineVersion as string;
const CHUNK_SIZE=4*1024*1024, MAX_SIZE=500*1024*1024;
type Upload={id:string;fileName:string;size:number;mimeType:string;fingerprint:string;chunkSize:number;uploadedChunks:number[];videoId?:string;createdAt:string};
type VideoRecord={id:string;fileName:string;inputPath:string;size:number};
type RecordData={id:string;status:Analysis['status'];createdAt:string;fileName:string;analysis:Analysis;video?:VideoAsset;error?:string;errorDetail?:string;errorCode?:string;attempts:number;captureFps?:number};
type Options={dataDir?:string;runWorker?:boolean;workerScript?:string};
const bad=(message:string,status=400)=>Object.assign(new Error(message),{status});
const wrap=(fn:(req:Request,res:Response)=>Promise<unknown>)=>(req:Request,res:Response,next:NextFunction)=>{Promise.resolve(fn(req,res)).catch(next);};
const asFloat=(b:Buffer)=>new Float32Array(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));

export async function createApp(options:Options={}) {
  const store=new Store(path.resolve(options.dataDir ?? process.env.DATA_DIR ?? path.join(ROOT,'data'))); await store.init();
  const app=express(); app.disable('x-powered-by');
  app.use((req,res,next)=>{res.setHeader('X-Content-Type-Options','nosniff'); next();});
  app.use(express.json({limit:'64kb'}));
  const locks=new Map<string,Promise<unknown>>();
  async function locked<T>(key:string,fn:()=>Promise<T>):Promise<T> {const prior=locks.get(key)??Promise.resolve(); const task=prior.catch(()=>{}).then(fn); locks.set(key,task); try{return await task;}finally{if(locks.get(key)===task)locks.delete(key);} }
  async function getUpload(id:string) {const value=await store.get<Upload>('uploads',id);if(!value)throw bad('上传会话不存在',404);return value;}
  async function getRecord(id:string) {const value=await store.get<RecordData>('analyses',id);if(!value)throw bad('分析任务不存在',404);return value;}
  async function skeletonData(analysis:Analysis) { if(!analysis.skeleton)throw bad('尚无骨架结果',409); const read=async(key:string)=>{if(!key.startsWith(`/data/results/${analysis.id}/`))throw bad('骨架路径无效',500); return asFloat(await readFile(path.join(store.root,key.slice('/data/'.length))));}; return {timestamps:await read(analysis.skeleton.timestampsKey),keypoints:await read(analysis.skeleton.keypointsKey)}; }
  let active=false,closed=false; let child:ReturnType<typeof spawn>|undefined;
  async function run(record:RecordData) {
    record.status='running'; record.analysis.status='running';record.attempts++;await store.put('analyses',record.id,record);
    const video=await store.get<VideoRecord>('videos',record.analysis.videoId); if(!video)throw new Error('原始视频不存在');
    const outputDir=path.join(store.root,'results',record.id);await mkdir(outputDir,{recursive:true});
    const args=[options.workerScript??path.join(ROOT,'worker/main.py'),'--input',video.inputPath,'--output-dir',outputDir,'--analysis-id',record.id,'--video-id',video.id,'--swing-type',record.analysis.userInput.swingType??'unknown'];
    if(record.analysis.userInput.targetPoint)args.push('--target-point',record.analysis.userInput.targetPoint.join(','));
    if(record.captureFps)args.push('--capture-fps',String(record.captureFps));
    const output=await new Promise<string>((resolve,reject)=>{
      const proc=spawn(resolveWorkerPython(ROOT),args,{cwd:ROOT,stdio:['ignore','pipe','pipe']});child=proc;
      let stdout='',stderr='',overflow=false,timedOut=false;
      const timer=setTimeout(()=>{timedOut=true;proc.kill('SIGKILL');},Number(process.env.WORKER_TIMEOUT_MS??600000));
      proc.stdout!.on('data',d=>{stdout+=d;if(stdout.length>4*1024*1024){overflow=true;proc.kill('SIGKILL');}});
      proc.stderr!.on('data',d=>{stderr=(stderr+d).slice(-8000);});
      proc.on('error',e=>{clearTimeout(timer);reject(e);});
      proc.on('close',code=>{clearTimeout(timer);if(child===proc)child=undefined;if(timedOut)reject(new Error('分析超时，请缩短视频后重试'));else if(overflow)reject(new Error('分析进程输出超过限制'));else if(code===0&&stdout.trim())resolve(stdout);else reject(new Error(`分析进程退出 (${code})：${stderr.slice(-1500)}`));});
    });
    let payload:{video?:VideoAsset;analysis?:Analysis;error?:string;errorCode?:string};try{payload=JSON.parse(output.trim().split('\n').filter(Boolean).at(-1)!);}catch{throw new Error('分析进程未返回有效的结果');}
    if(!payload.analysis)throw new Error(payload.error??'分析进程未返回结果');
    const analysis=parseAnalysis(payload.analysis);
    if(analysis.id!==record.id||analysis.videoId!==video.id||analysis.schemaVersion!==SCHEMA_VERSION||analysis.pipelineVersion!==PIPELINE_VERSION)throw new Error('分析结果的数据版本或任务 ID 不匹配');
    if(!['succeeded','failed','rejected_quality'].includes(analysis.status))throw new Error('分析进程未返回终态');
    const prefix=`/data/results/${record.id}/`;
    if(analysis.skeleton){for(const k of ['timestampsKey','keypointsKey'] as const){analysis.skeleton[k]=prefix+path.basename(analysis.skeleton[k]);}}
    record.analysis=analysis;record.status=analysis.status;record.video=payload.video?parseVideoAsset(payload.video):undefined;
    record.errorDetail=payload.error;record.errorCode=payload.errorCode;record.error=workerFailureMessage(payload.error,payload.errorCode);
    if(record.video){if(record.video.id!==video.id)throw new Error('视频结果 ID 不匹配');record.video.storageKey=`/data/videos/${video.id}/original${path.extname(video.inputPath)}`;if(record.video.proxyKey)record.video.proxyKey=prefix+path.basename(record.video.proxyKey);}
    if(analysis.status==='succeeded'){
      if(!record.video||!analysis.skeleton)throw new Error('成功结果缺少视频或骨架数据');
      if(!analysis.quality?.usable||analysis.swings.length===0)throw new Error('成功结果缺少可用质量报告或击球事件');
      const {timestamps,keypoints}=await skeletonData(analysis);
      validateSkeletonData(timestamps,keypoints,analysis.skeleton.frameCount);
      const start=performance.now();analysis.metrics=calculateMetrics(record.video,analysis,timestamps,keypoints);analysis.stageTimingsMs.metrics=performance.now()-start;
    }
    parseAnalysis(analysis);await store.put('analyses',record.id,record);
  }
  async function drain(){if(active||closed||options.runWorker===false)return;active=true;try{while(!closed){const next=(await store.list<RecordData>('analyses')).filter(x=>x.status==='queued').sort((a,b)=>a.createdAt.localeCompare(b.createdAt))[0];if(!next)break;try{await run(next);}catch(e){next.status='failed';next.analysis.status='failed';next.error=(e as Error).message;await store.put('analyses',next.id,next);}}}finally{active=false;}}
  // Recover interrupted jobs once; explicit failures require the retry endpoint.
  for(const r of await store.list<RecordData>('analyses')){if(r.status==='running'){r.status=r.attempts<2?'queued':'failed';r.analysis.status=r.status;r.error=r.status==='failed'?'任务多次被中断，请重试':undefined;await store.put('analyses',r.id,r);}}
  const timer=setInterval(()=>void drain().catch(console.error),1000);timer.unref();void drain();
  app.get('/api/health',(_req,res)=>res.json({ok:true,schemaVersion:SCHEMA_VERSION,pipelineVersion:PIPELINE_VERSION,queue:active?'running':'idle'}));
  app.use('/api/fixtures',express.static(path.join(ROOT,'fixtures'),{fallthrough:false}));
  app.get('/api/demo',wrap(async(_req,res)=>{const [video,analysis]=await Promise.all(['demo-video.json','demo-analysis.json'].map(f=>readFile(path.join(ROOT,'fixtures',f),'utf8').then(JSON.parse)));res.json({video,analysis,demo:true});}));
  app.post('/api/uploads',wrap(async(req,res)=>{
    const {fileName,size,mimeType='',fingerprint}=req.body??{};
    if(typeof fileName!=='string'||fileName.length>240||!['.mp4','.mov','.webm','.m4v'].includes(path.extname(fileName).toLowerCase()))throw bad('请选择 MP4、MOV、M4V 或 WebM 视频');
    if(!Number.isSafeInteger(size)||size<=0||size>MAX_SIZE)throw bad('视频大小需在 1 字节至 500 MB 之间');
    if(typeof fingerprint!=='string'||!fingerprint.length||fingerprint.length>200)throw bad('文件指纹无效');
    const key=createHash('sha256').update(`${fingerprint}:${size}:${fileName}`).digest('hex');
    const value=await locked(`upload:${key}`,async()=>{const old=await store.get<Upload>('uploads',key);if(old)return old;const u:Upload={id:key,fileName:path.basename(fileName),size,mimeType:String(mimeType).slice(0,100),fingerprint,chunkSize:CHUNK_SIZE,uploadedChunks:[],createdAt:new Date().toISOString()};await mkdir(path.join(store.root,'uploads',key),{recursive:true});await store.put('uploads',key,u);return u;});res.json(value);
  }));
  app.get('/api/uploads/:id',wrap(async(req,res)=>res.json(await getUpload(String(req.params.id)))));
  app.put('/api/uploads/:id/chunks/:index',express.raw({type:()=>true,limit:CHUNK_SIZE}),wrap(async(req,res)=>{
    const id=String(req.params.id);res.json(await locked(`upload:${id}`,async()=>{const u=await getUpload(id);if(u.videoId)return {ok:true,completed:true};const index=Number(req.params.index),count=Math.ceil(u.size/CHUNK_SIZE);if(!Number.isSafeInteger(index)||index<0||index>=count)throw bad('分片编号无效');const expected=index===count-1?u.size-index*CHUNK_SIZE:CHUNK_SIZE;if(!Buffer.isBuffer(req.body)||req.body.length!==expected)throw bad('分片长度不匹配');const dest=path.join(store.root,'uploads',id,`${index}.part`);await writeFile(`${dest}.tmp`,req.body);await rename(`${dest}.tmp`,dest);if(!u.uploadedChunks.includes(index))u.uploadedChunks.push(index);u.uploadedChunks.sort((a,b)=>a-b);await store.put('uploads',id,u);return {ok:true,uploadedChunks:u.uploadedChunks};}));
  }));
  app.post('/api/uploads/:id/complete',wrap(async(req,res)=>{const id=String(req.params.id);res.json(await locked(`upload:${id}`,async()=>{
    const u=await getUpload(id);if(u.videoId)return {videoId:u.videoId};const count=Math.ceil(u.size/CHUNK_SIZE);if(u.uploadedChunks.length!==count)throw bad('上传尚未完成，请继续上传缺少的分片',409);
    const videoId=randomUUID(),dir=path.join(store.root,'videos',videoId);await mkdir(dir,{recursive:true});const dest=path.join(dir,`original${path.extname(u.fileName).toLowerCase()}`),handle=await open(`${dest}.tmp`,'w');
    try{for(let i=0;i<count;i++){const bytes=await readFile(path.join(store.root,'uploads',id,`${i}.part`));const expected=i===count-1?u.size-i*CHUNK_SIZE:CHUNK_SIZE;if(bytes.length!==expected)throw bad('分片校验失败，请重新上传',409);let offset=0;while(offset<bytes.length){const {bytesWritten}=await handle.write(bytes,offset,bytes.length-offset);if(bytesWritten===0)throw new Error('写入视频失败');offset+=bytesWritten;}}if((await handle.stat()).size!==u.size)throw new Error('合并后的文件大小不匹配');}finally{await handle.close();}
    await rename(`${dest}.tmp`,dest);await store.put('videos',videoId,{id:videoId,fileName:u.fileName,inputPath:dest,size:u.size});u.videoId=videoId;await store.put('uploads',id,u);return {videoId};
  }));}));
  app.post('/api/analyses',wrap(async(req,res)=>{
    const {videoId,swingType='smash',targetPoint,captureFps}=req.body??{};
    if(typeof videoId!=='string'||!['smash','clear'].includes(swingType))throw bad('视频或动作类型无效');
    if(targetPoint!==undefined&&(!Array.isArray(targetPoint)||targetPoint.length!==2||!targetPoint.every((x:unknown)=>typeof x==='number'&&Number.isFinite(x)&&x>=0&&x<=1)))throw bad('球员坐标须归一化到 0–1');
    if(captureFps!==undefined&&(!Number.isFinite(captureFps)||captureFps<15||captureFps>480))throw bad('采集帧率须为 15–480 fps');
    const video=await store.get<VideoRecord>('videos',videoId);if(!video)throw bad('视频不存在',404);
    const id=createHash('sha256').update(JSON.stringify({videoId,swingType,targetPoint,captureFps,pipelineVersion:PIPELINE_VERSION})).digest('hex').slice(0,32);
    const r=await locked(`analysis:${id}`,async()=>{const old=await store.get<RecordData>('analyses',id);if(old)return old;const analysis:Analysis={id,videoId,status:'queued',schemaVersion:SCHEMA_VERSION,pipelineVersion:PIPELINE_VERSION,userInput:{swingType,...(targetPoint?{targetPoint}:{})},swings:[],metrics:[],diagnoses:[],stageTimingsMs:{}};const value:RecordData={id,status:'queued',createdAt:new Date().toISOString(),fileName:video.fileName,analysis,attempts:0,captureFps};await store.put('analyses',id,value);return value;});res.status(202).json({id:r.id,status:r.status});void drain();
  }));
  app.get('/api/analyses',wrap(async(_req,res)=>res.json((await store.list<RecordData>('analyses')).sort((a,b)=>b.createdAt.localeCompare(a.createdAt)).map(({id,status,createdAt,fileName,error})=>({id,status,createdAt,fileName,error})))));
  app.get('/api/analyses/:id',wrap(async(req,res)=>{const r=await getRecord(String(req.params.id));res.json({video:r.video,analysis:r.analysis,error:r.error,errorDetail:r.errorDetail,errorCode:r.errorCode});}));
  app.post('/api/analyses/:id/retry',wrap(async(req,res)=>{const id=String(req.params.id);const r=await locked(`analysis:${id}`,async()=>{const r=await getRecord(id);if(r.status!=='failed')throw bad('只有失败的任务可以重试',409);r.status='queued';r.analysis.status='queued';r.error=undefined;r.errorDetail=undefined;r.errorCode=undefined;await store.put('analyses',id,r);return r;});res.status(202).json({id:r.id,status:r.status});void drain();}));
  app.post('/api/analyses/:id/annotations',wrap(async(req,res)=>{const id=String(req.params.id);const {frame}=req.body??{};
    if(id==='demo'){throw bad('演示结果仅供预览，请上传视频后保存标注',409);}
    const result=await locked(`analysis:${id}`,async()=>{
      const r=await getRecord(id);
      if(r.status!=='succeeded'||!r.video||!r.analysis.skeleton)throw bad('任务完成后才能标注',409);
      if(!Number.isSafeInteger(frame)||frame<0||frame>=r.analysis.skeleton.frameCount)throw bad('击球帧超出范围');
      const {timestamps,keypoints}=await skeletonData(r.analysis);
      const existing=r.analysis.swings.find(s=>s.impact.frame===frame);
      if(existing){if(!existing.impact.evidence.includes('manual'))existing.impact.evidence.push('manual');}
      else r.analysis.swings.push({id:`manual-${frame}`,type:r.analysis.userInput.swingType??'unknown',typeConfidence:1,handedness:r.analysis.swings[0]?.handedness??'unknown',impact:{frame,tSec:timestamps[frame],confidence:1,evidence:['manual']},phases:{impactFrame:frame}});
      r.analysis.swings.sort((a,b)=>a.impact.frame-b.impact.frame);
      r.analysis.metrics=calculateMetrics(r.video,r.analysis,timestamps,keypoints);
      await store.put('analyses',id,r);
      // Keep an independent ground-truth sidecar so evaluator labels do not depend on predictions.
      const annotationFile=path.join(store.root,'results',id,'annotations.json');
      let labels:{impactFrames:number[]};try{labels=JSON.parse(await readFile(annotationFile,'utf8'));}catch(e){if((e as NodeJS.ErrnoException).code!=='ENOENT')throw e;labels={impactFrames:[]};}
      if(!labels.impactFrames.includes(frame))labels.impactFrames.push(frame);
      labels.impactFrames.sort((a,b)=>a-b);
      await writeFile(`${annotationFile}.tmp`,JSON.stringify({analysisId:id,videoId:r.video.id,...labels,targetPoint:r.analysis.userInput.targetPoint,updatedAt:new Date().toISOString()},null,2));
      await rename(`${annotationFile}.tmp`,annotationFile);
      return {video:r.video,analysis:r.analysis};
    });res.json(result);
  }));
  app.use('/data/videos',express.static(path.join(store.root,'videos'),{dotfiles:'deny',fallthrough:false}));
  app.use('/data/results',express.static(path.join(store.root,'results'),{dotfiles:'deny',fallthrough:false}));
  app.use('/api',(_req,res)=>res.status(404).json({error:'API 不存在'}));
  const web=path.join(ROOT,'apps/web/dist');try{await stat(web);app.use(express.static(web));app.get('/{*path}',(_req,res)=>res.sendFile(path.join(web,'index.html')));}catch{}
  app.use((err:Error & {status?:number;type?:string},_req:Request,res:Response,_next:NextFunction)=>{res.status(err.status??500).json({error:err.type==='entity.too.large'?'上传分片过大':err.message});});
  return {app,store,close:()=>{closed=true;clearInterval(timer);child?.kill('SIGTERM');}};
}
