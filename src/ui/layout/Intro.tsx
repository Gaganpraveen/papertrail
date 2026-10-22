/**
 * Opening intro: the logo's pages slide into place, the name sharpens, then
 * the curtain lifts. Shown once per browser session, skipped for reduced
 * motion and automated browsers. It never blocks clicks.
 */
import { useEffect, useState } from "react";

const SEEN_KEY = "papertrail:intro-seen";

function shouldPlay(): boolean {
  try {
    if (navigator.webdriver) return false;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return false;
    if (sessionStorage.getItem(SEEN_KEY)) return false;
    sessionStorage.setItem(SEEN_KEY, "1");
    return true;
  } catch {
    return false;
  }
}

export function Intro() {
  const [show, setShow] = useState(shouldPlay);

  useEffect(() => {
    if (!show) return;
    const timer = window.setTimeout(() => setShow(false), 1700);
    return () => window.clearTimeout(timer);
  }, [show]);

  if (!show) return null;
  return (
    <div aria-hidden className="pt-intro pointer-events-none fixed inset-0 z-[100] flex items-center justify-center bg-bg">
      <div className="flex items-center gap-3">
        <svg viewBox="0 0 32 32" fill="none" className="size-12 text-ink">
          <rect className="pt-page" style={{ "--o": 0.22, animationDelay: "0ms" } as React.CSSProperties} x="13" y="3" width="15" height="19" rx="3" fill="currentColor" />
          <rect className="pt-page" style={{ "--o": 0.5, animationDelay: "90ms" } as React.CSSProperties} x="8.5" y="6.5" width="15" height="19" rx="3" fill="currentColor" />
          <g className="pt-page" style={{ animationDelay: "180ms" }}>
            <path d="M4 13a3 3 0 0 1 3-3h7.6L19 14.4V26a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V13Z" fill="var(--accent)" />
            <path d="M14.6 10v2.4a2 2 0 0 0 2 2H19" stroke="var(--bg)" strokeWidth="1.4" strokeLinejoin="round" />
          </g>
        </svg>
        <span className="pt-word text-3xl font-semibold text-ink">PaperTrail</span>
      </div>
    </div>
  );
}
