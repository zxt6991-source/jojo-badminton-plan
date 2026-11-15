import type { CSSProperties } from 'react';

const paths: Record<string, React.ReactNode> = {
  activity: <><path d="M3 12h4l3-8 4 16 3-8h4" /></>,
  grid: <><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></>,
  folder: <path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z" />,
  upload: <><path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5" /></>,
  arrow: <><path d="M5 12h14m-6-6 6 6-6 6" /></>,
  chevron: <path d="m9 5 7 7-7 7" />,
  chevronDown: <path d="m6 9 6 6 6-6" />,
  book: <><path d="M12 5v16M3 3h5a4 4 0 0 1 4 2 4 4 0 0 1 4-2h5v17h-5a4 4 0 0 0-4 1 4 4 0 0 0-4-1H3Z" /></>,
  play: <path d="m8 5 11 7-11 7V5Z" />,
  pause: <><path d="M8 5v14m8-14v14" /></>,
  prev: <><path d="M5 5v14m13-14-10 7 10 7V5Z" /></>,
  next: <><path d="M19 5v14M6 5l10 7-10 7V5Z" /></>,
  target: <><circle cx="12" cy="12" r="7" /><circle cx="12" cy="12" r="2" /><path d="M12 1v4m0 14v4M1 12h4m14 0h4" /></>,
  eye: <><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z" /><circle cx="12" cy="12" r="3" /></>,
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6m0-10v.1" /></>,
  check: <path d="m5 12 4 4L19 6" />,
  close: <path d="m6 6 12 12M6 18 18 6" />,
  film: <><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M7 3v18m10-18v18M3 8h4m-4 8h4M17 8h4m-4 8h4" /></>,
  clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
  bolt: <path d="m13 2-9 12h7l-1 8 10-13h-8l1-7Z" />,
  bone: <><circle cx="12" cy="4" r="2" /><path d="M12 6v8m-5-5 5 2 5-2m-5 5-5 7m5-7 5 7" /></>,
  menu: <path d="M4 6h16M4 12h16M4 18h16" />,
  refresh: <><path d="M20 8a8 8 0 1 0 1 8M20 3v5h-5" /></>,
  shield: <><path d="M12 2 3 6v6c0 5 9 10 9 10s9-5 9-10V6Z" /><path d="m8 12 3 3 5-6" /></>,
  camera: <><rect x="2" y="6" width="20" height="15" rx="3" /><path d="m7 6 2-3h6l2 3" /><circle cx="12" cy="13" r="4" /></>,
};
export default function Icon({ name, size = 20, className, style }: { name: string; size?: number; className?: string; style?: CSSProperties }) {
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" className={className} style={style}>{paths[name] || paths.info}</svg>;
}
