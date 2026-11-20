import {accessSync, constants} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const target = process.argv[2];
if (!['tests/worker', 'tests/pose3d'].includes(target)) throw new Error('Unknown Python test suite');
const local = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
let python = process.env.PYTHON?.trim();
if (!python) {
  try {accessSync(local, constants.X_OK); python = local;} catch {python = 'python3';}
}
const result = spawnSync(python, ['-m', 'unittest', 'discover', '-s', target, '-v'], {cwd: root, stdio: 'inherit'});
if (result.error) console.error(`Python tests could not start: ${result.error.message}`);
process.exit(result.status ?? 1);
