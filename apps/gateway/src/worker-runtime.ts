import {accessSync,constants,statSync} from 'node:fs';
import path from 'node:path';

/** Respect an explicit override; otherwise use the environment installed here. */
export function resolveWorkerPython(root:string,configured=process.env.PYTHON):string {
  if(configured?.trim())return configured.trim();
  const local=process.platform==='win32'?path.join(root,'.venv','Scripts','python.exe'):path.join(root,'.venv','bin','python');
  try{accessSync(local,constants.X_OK);if(statSync(local).isFile())return local;}catch{}
  return 'python3';
}

/** Keep the worker's raw error/code separately; this text is for the result UI. */
export function workerFailureMessage(error:string|undefined,errorCode:string|undefined):string|undefined {
  if(errorCode!=='worker_dependencies_missing')return error;
  return `分析运行环境缺少 Python 依赖${error?`（${error}）`:''}。请在项目目录创建 .venv，并运行 .venv/bin/python -m pip install -r worker/requirements.txt；如已安装，请检查 PYTHON 是否指向该环境，再重试分析。`;
}
