import type { Analysis, VideoAsset } from '@badminton/schema';

export interface Result { video: VideoAsset; analysis: Analysis; error?: string; }
export interface HistoryItem { id: string; status: Analysis['status']; createdAt?: string; fileName?: string; }

export async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.error === 'string' ? data.error : data.message || `请求失败（${response.status}）`);
  return data as T;
}

export const json = (value: unknown): RequestInit => ({
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value),
});

export async function uploadVideo(file: File, onProgress: (percent: number) => void, signal: AbortSignal) {
  // Hash both ends to distinguish files even when names and sizes match.
  const head = new Uint8Array(await file.slice(0, 65536).arrayBuffer());
  const tail = new Uint8Array(await file.slice(Math.max(0, file.size - 65536)).arrayBuffer());
  const bytes = new Uint8Array(head.length + tail.length);
  bytes.set(head); bytes.set(tail, head.length);
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))).map(n => n.toString(16).padStart(2, '0')).join('');
  const fingerprint = `${file.size}:${file.lastModified}:${hash}`;
  const session = await api<{ id: string; chunkSize: number; uploadedChunks: number[] }>('/api/uploads', {
    ...json({ fileName: file.name, size: file.size, mimeType: file.type, fingerprint }), signal,
  });
  const total = Math.ceil(file.size / session.chunkSize);
  const complete = new Set(session.uploadedChunks);
  onProgress(Math.round(complete.size / total * 100));
  for (let index = 0; index < total; index++) {
    if (complete.has(index)) continue;
    const response = await fetch(`/api/uploads/${session.id}/chunks/${index}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/octet-stream' },
      body: file.slice(index * session.chunkSize, Math.min(file.size, (index + 1) * session.chunkSize)), signal,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || `第 ${index + 1} 个分片上传失败，请重试续传。`);
    }
    complete.add(index); onProgress(Math.round(complete.size / total * 100));
  }
  return api<{ videoId: string }>(`/api/uploads/${session.id}/complete`, { ...json({}), signal });
}
