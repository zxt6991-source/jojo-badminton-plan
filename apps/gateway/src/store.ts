import {mkdir, readFile, writeFile, rename, readdir} from 'node:fs/promises';
import path from 'node:path';
import {randomUUID} from 'node:crypto';

export class Store {
  constructor(readonly root: string) {}
  async init() { for (const name of ['uploads', 'videos', 'analyses', 'results']) await mkdir(path.join(this.root,name),{recursive:true}); }
  file(collection:string,id:string) { if(!/^[a-zA-Z0-9_-]+$/.test(id)) throw Object.assign(new Error('无效的资源 ID'),{status:400}); return path.join(this.root,collection,`${id}.json`); }
  async get<T>(collection:string,id:string):Promise<T|undefined> { try { return JSON.parse(await readFile(this.file(collection,id),'utf8')); } catch(e) { if((e as NodeJS.ErrnoException).code==='ENOENT') return undefined; throw e; } }
  async put(collection:string,id:string,value:unknown) { const file=this.file(collection,id); const temp=`${file}.${randomUUID()}.tmp`; await writeFile(temp, JSON.stringify(value,null,2)); await rename(temp,file); }
  async list<T>(collection:string):Promise<T[]> { const files=await readdir(path.join(this.root,collection)); return Promise.all(files.filter(x=>x.endsWith('.json')).map(async x=>JSON.parse(await readFile(path.join(this.root,collection,x),'utf8')))); }
}
