import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,chmod,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {resolveWorkerPython,workerFailureMessage} from '../src/worker-runtime.js';

test('worker discovers a newly installed project environment and honors explicit overrides',async()=>{
  const root=await mkdtemp(path.join(tmpdir(),'badminton python '));
  try{
    assert.equal(resolveWorkerPython(root,''),'python3');
    const python=process.platform==='win32'?path.join(root,'.venv','Scripts','python.exe'):path.join(root,'.venv','bin','python');
    await mkdir(path.dirname(python),{recursive:true});await writeFile(python,'#!/bin/sh\n');await chmod(python,0o755);
    assert.equal(resolveWorkerPython(root,''),python);
    assert.equal(resolveWorkerPython(root,'  '),python);
    assert.equal(resolveWorkerPython(root,'python3'),'python3');
    const custom=path.join(root,'another environment','python');
    assert.equal(resolveWorkerPython(root,custom),custom,'an explicit unavailable interpreter must not silently select another environment');
    if(process.platform!=='win32'){await chmod(python,0o644);assert.equal(resolveWorkerPython(root,''),'python3');}
  }finally{await rm(root,{recursive:true,force:true});}
});

test('missing dependency guidance preserves the exact import failure and leaves unrelated errors alone',()=>{
  const detail="No module named 'numpy'";
  const message=workerFailureMessage(detail,'worker_dependencies_missing')!;
  assert.ok(message.includes(detail));assert.match(message,/缺少 Python 依赖/);assert.match(message,/worker\/requirements.txt/);assert.match(message,/PYTHON/);
  assert.equal(workerFailureMessage('视频无法解码','video_decode_failed'),'视频无法解码');
  assert.equal(workerFailureMessage(undefined,undefined),undefined);
});
