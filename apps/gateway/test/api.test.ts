import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm,readFile,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {createApp} from '../src/app.js';
import type {ReplayAdvisorOptions} from '../src/replay-profile.js';

async function setup(runWorker=false, replayAdvisor?:ReplayAdvisorOptions){const dataDir=await mkdtemp(path.join(tmpdir(),'badminton-api-'));const service=await createApp({dataDir,runWorker,replayAdvisor,workerScript:path.resolve('apps/gateway/test/worker-fixture.py')});const server=service.app.listen(0,'127.0.0.1');await new Promise<void>(resolve=>server.once('listening',resolve));const address=server.address() as {port:number};const base=`http://127.0.0.1:${address.port}`;return {...service,dataDir,base,cleanup:async()=>{service.close();await new Promise<void>((resolve,reject)=>server.close(e=>e?reject(e):resolve()));await rm(dataDir,{recursive:true,force:true});}};}
async function json(base:string,url:string,body?:unknown){return fetch(base+url,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});}

test('resumable chunks survive service restart, completion and analysis creation are idempotent',async()=>{
 const s=await setup();try{
  let response=await json(s.base,'/api/uploads',{fileName:'训练.mp4',size:6,mimeType:'video/mp4',fingerprint:'abc'});assert.equal(response.status,200);const upload=await response.json();
  response=await json(s.base,`/api/uploads/${upload.id}/complete`,{});assert.equal(response.status,409);
  response=await fetch(`${s.base}/api/uploads/${upload.id}/chunks/0`,{method:'PUT',headers:{'Content-Type':'application/octet-stream'},body:Buffer.from('short')});assert.equal(response.status,400);
  response=await fetch(`${s.base}/api/uploads/${upload.id}/chunks/0`,{method:'PUT',headers:{'Content-Type':'application/octet-stream'},body:Buffer.from('video!')});assert.equal(response.status,200);
  const second=await createApp({dataDir:s.dataDir,runWorker:false});const saved=await second.store.get<{uploadedChunks:number[]}>('uploads',upload.id);assert.deepEqual(saved?.uploadedChunks,[0]);second.close();
  const video=await (await json(s.base,`/api/uploads/${upload.id}/complete`,{})).json();const again=await (await json(s.base,`/api/uploads/${upload.id}/complete`,{})).json();assert.equal(video.videoId,again.videoId);
  const first=await (await json(s.base,'/api/analyses',{videoId:video.videoId,swingType:'smash',targetPoint:[.4,.6]})).json();const repeat=await (await json(s.base,'/api/analyses',{videoId:video.videoId,swingType:'smash',targetPoint:[.4,.6]})).json();assert.equal(first.id,repeat.id);assert.equal(first.status,'queued');
  const record=await (await json(s.base,`/api/analyses/${first.id}`)).json();assert.deepEqual(record.analysis.userInput.targetPoint,[.4,.6]);assert.deepEqual(record.analysis.metrics,[]);
 }finally{await s.cleanup();}
});

test('rejects invalid inputs and missing resources with JSON errors',async()=>{
 const s=await setup();try{
  assert.equal((await json(s.base,'/api/uploads',{fileName:'file.exe',size:100,fingerprint:'x'})).status,400);
  assert.equal((await json(s.base,'/api/uploads',{fileName:'file.mp4',size:501*1024*1024,fingerprint:'x'})).status,400);
  assert.equal((await json(s.base,'/api/analyses',{videoId:'nope',swingType:'smash',targetPoint:[2,0]})).status,400);
  assert.equal((await json(s.base,'/api/analyses',{videoId:'nope',swingType:'clear'})).status,404);
  assert.equal((await json(s.base,'/api/analyses/missing')).status,404);
  assert.equal((await json(s.base,'/api/no-such-api')).status,404);
  const health=await (await json(s.base,'/api/health')).json();assert.equal(health.schemaVersion,'0.2.0');
 }finally{await s.cleanup();}
});

test('demo is explicit and binary skeleton is accessible',async()=>{
 const s=await setup();try{
  const demo=await (await json(s.base,'/api/demo')).json();assert.equal(demo.demo,true);assert.equal(demo.analysis.schemaVersion,'0.1.0');assert.equal(demo.video.storageKey,'');
  const binary=await fetch(s.base+demo.analysis.skeleton.keypointsKey);assert.equal(binary.status,200);assert.equal((await binary.arrayBuffer()).byteLength,demo.analysis.skeleton.frameCount*65*3*4);
  const annotate=await json(s.base,'/api/analyses/demo/annotations',{frame:20});assert.equal(annotate.status,409);
 }finally{await s.cleanup();}
});

async function enqueue(base:string,mode:string){
 const u=await (await json(base,'/api/uploads',{fileName:`${mode}.mp4`,size:mode.length,fingerprint:mode})).json();
 assert.equal((await fetch(`${base}/api/uploads/${u.id}/chunks/0`,{method:'PUT',headers:{'Content-Type':'application/octet-stream'},body:mode})).status,200);
 const {videoId}=await (await json(base,`/api/uploads/${u.id}/complete`,{})).json();
 return (await (await json(base,'/api/analyses',{videoId,swingType:'smash'})).json()).id as string;
}
async function terminal(base:string,id:string){for(let i=0;i<100;i++){const result=await (await json(base,`/api/analyses/${id}`)).json();if(['succeeded','failed','rejected_quality'].includes(result.analysis.status))return result;await new Promise(resolve=>setTimeout(resolve,50));}throw new Error('Task did not finish');}

test('worker boundary computes real TS metrics and persists repeated manual labels once',async()=>{
 const s=await setup(true);try{
  const id=await enqueue(s.base,'ok'),result=await terminal(s.base,id);
  assert.equal(result.analysis.status,'succeeded');assert.equal(result.analysis.metrics.length,15);
  const frame=result.analysis.swings[0].impact.frame;
  for(let i=0;i<2;i++)assert.equal((await json(s.base,`/api/analyses/${id}/annotations`,{frame})).status,200);
  const annotated=await (await json(s.base,`/api/analyses/${id}`)).json();assert.equal(annotated.analysis.swings.length,3);
  assert.equal(annotated.analysis.swings[0].impact.evidence.filter((e:string)=>e==='manual').length,1);
  const labels=JSON.parse(await readFile(path.join(s.dataDir,'results',id,'annotations.json'),'utf8'));assert.deepEqual(labels.impactFrames,[frame]);
 }finally{await s.cleanup();}
});

test('crashed or timed-out worker cannot report success or prevent the next job',async()=>{
 const old=process.env.WORKER_TIMEOUT_MS;process.env.WORKER_TIMEOUT_MS='1000';
 const s=await setup(true);try{
  const crashed=await enqueue(s.base,'crash');assert.equal((await terminal(s.base,crashed)).analysis.status,'failed');
  const timeout=await enqueue(s.base,'timeout'),ok=await enqueue(s.base,'ok');
  const failed=await terminal(s.base,timeout);assert.equal(failed.analysis.status,'failed');assert.match(failed.error,/超时/);
  assert.equal((await terminal(s.base,ok)).analysis.status,'succeeded');
 }finally{if(old===undefined)delete process.env.WORKER_TIMEOUT_MS;else process.env.WORKER_TIMEOUT_MS=old;await s.cleanup();}
});

test('dependency failures keep diagnostics and a successful retry clears them',async()=>{
 const s=await setup(true);try{
  const id=await enqueue(s.base,'dependencies'),failed=await terminal(s.base,id);
  assert.equal(failed.analysis.status,'failed');assert.equal(failed.errorCode,'worker_dependencies_missing');
  assert.equal(failed.errorDetail,"No module named 'numpy'");assert.match(failed.error,/缺少 Python 依赖/);
  assert.equal(failed.analysis.quality.issues[0].code,'worker_dependencies_missing');
  // The fixture chooses its outcome from the input; simulate a repaired worker.
  const video=await s.store.get<{inputPath:string}>('videos',failed.analysis.videoId);
  await writeFile(video!.inputPath,'ok');
  assert.equal((await json(s.base,`/api/analyses/${id}/retry`,{})).status,202);
  const repaired=await terminal(s.base,id);assert.equal(repaired.analysis.status,'succeeded');
  assert.equal(repaired.error,undefined);assert.equal(repaired.errorCode,undefined);assert.equal(repaired.errorDetail,undefined);
 }finally{await s.cleanup();}
});

test('replay advisor uses stored or demo data, validates completions and coalesces requests', async () => {
  let calls = 0;
  const s = await setup(true, {
    apiKey: 'mock-only', baseUrl: 'https://replay.example/v1', model: 'test-model',
    fetch: async (url, options) => {
      calls++;
      assert.equal(String(url), 'https://replay.example/v1/chat/completions');
      assert.equal(new Headers(options?.headers).get('Authorization'), 'Bearer mock-only');
      const body = JSON.parse(String(options?.body));
      assert.equal(body.model, 'test-model');
      const summary = JSON.parse(body.messages[1].content);
      assert.deepEqual(Object.keys(summary).sort(), [
        'captureFps', 'missingJointFraction', 'sharpReversalFraction', 'timestampGapFraction',
        'usableTripletFraction', 'wristSpeedP95TorsoLengthsPerSec',
      ]);
      assert.ok(Object.values(summary).every(v => v === null || typeof v === 'number' && Number.isFinite(v)));
      const content = calls === 1 ? '{"profile":"invalid"}' : '{"profile":"steady"}';
      return Response.json({choices: [{finish_reason:'stop', message:{content}}]});
    },
  });
  try {
    const path = '/api/analyses/demo/replay-profile';
    assert.deepEqual(await (await json(s.base, '/api/replay-profile/status')).json(), {configured:true});
    assert.equal((await fetch(s.base + path, {method:'POST',body:'{}'})).status, 415);
    assert.equal((await json(s.base,path,{model:'ignored'})).status, 400);
    assert.equal((await json(s.base,'/api/analyses/unknown/replay-profile',{})).status, 404);
    const invalid = await json(s.base,path,{});
    assert.equal(invalid.status, 502);
    assert.equal((await invalid.json()).errorCode, 'invalid_replay_profile');
    const responses = await Promise.all([json(s.base,path,{}),json(s.base,path,{})]);
    for (const result of responses) assert.deepEqual(await result.json(),{profile:'steady'});
    assert.equal(calls,2);
    const id = await enqueue(s.base,'ok');
    assert.equal((await terminal(s.base,id)).analysis.status,'succeeded');
    assert.deepEqual(await (await json(s.base,'/api/analyses/'+id+'/replay-profile',{})).json(),{profile:'steady'});
    assert.equal(calls,3);
  } finally { await s.cleanup(); }
});

test('missing AI credentials preserve local analysis and expose configuration status', async () => {
  const s = await setup(false,{apiKey:''});
  try {
    assert.deepEqual(await (await json(s.base, '/api/replay-profile/status')).json(),{configured:false});
    assert.equal((await json(s.base,'/api/analyses/demo/replay-profile',{})).status,503);
    assert.equal((await json(s.base,'/api/demo')).status,200);
  } finally { await s.cleanup(); }
});

test('advisor timeouts and authentication errors leave the same clip available for retry', async () => {
  let attempt = 0;
  const s = await setup(false, {
    apiKey:'mock-only', baseUrl:'https://replay.example', timeoutMs:20,
    fetch: async (_url, options) => {
      attempt++;
      if (attempt === 1) {
        await new Promise((_,reject) => options!.signal!.addEventListener('abort',reject,{once:true}));
      }
      if (attempt === 2) return new Response('private upstream diagnostic',{status:401});
      return Response.json({choices:[{finish_reason:'stop',message:{content:'{"profile":"responsive"}'}}]});
    },
  });
  try {
    const route = '/api/analyses/demo/replay-profile';
    const timeout = await json(s.base,route,{});
    assert.equal(timeout.status,504);
    assert.equal((await timeout.json()).errorCode,'replay_ai_timeout');
    const auth = await json(s.base,route,{});
    assert.equal(auth.status,502);
    assert.doesNotMatch(await auth.text(),/private upstream diagnostic|mock-only/);
    assert.deepEqual(await (await json(s.base,route,{})).json(),{profile:'responsive'});
  } finally { await s.cleanup(); }
});
