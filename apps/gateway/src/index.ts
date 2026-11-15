import {createApp} from './app.js';
import {existsSync} from 'node:fs';
import {loadEnvFile} from 'node:process';
if(existsSync('.env'))loadEnvFile('.env');
const {app,close}=await createApp();
const port=Number(process.env.PORT??8787),host=process.env.HOST??'127.0.0.1';
const server=app.listen(port,host,()=>console.log(`Badminton API: http://${host}:${port}`));
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>{close();server.close(()=>process.exit(0));});
