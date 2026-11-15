import {spawn} from 'node:child_process';
import {existsSync} from 'node:fs';
import {loadEnvFile} from 'node:process';
if(existsSync('.env'))loadEnvFile('.env');
const children = [spawn('npm', ['run', 'dev:api'], {stdio:'inherit'}), spawn('npm', ['run', 'dev:web'], {stdio:'inherit'})];
let stopping = false;
function stop(code=0) { if(stopping) return; stopping=true; for(const child of children) child.kill('SIGTERM'); setTimeout(()=>process.exit(code),250); }
for(const child of children) child.on('exit', code=>stop(code ?? 0));
process.on('SIGINT',()=>stop()); process.on('SIGTERM',()=>stop());
