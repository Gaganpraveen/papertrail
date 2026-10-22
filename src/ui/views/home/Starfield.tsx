/**
 * Landing backdrop: a slow, layered star field. Stars sit at three depths, so
 * scrolling and pointer movement shift near stars more than far ones, and a
 * few faint links join close neighbours like a citation graph. It draws one
 * still frame for reduced motion and pauses while off-screen or hidden.
 */
import { useEffect, useRef } from "react";

interface Star {
  x: number;
  y: number;
  z: number; // depth 0.25 (far) … 1 (near)
  r: number;
  phase: number;
  speed: number;
}

const DENSITY = 1 / 5200; // stars per CSS pixel²
const LINK_DISTANCE = 86;

function makeStars(width: number, height: number): Star[] {
  const count = Math.min(420, Math.round(width * height * DENSITY));
  return Array.from({ length: count }, () => {
    const z = [0.25, 0.5, 1][Math.floor(Math.random() * 3)];
    return {
      x: Math.random() * width,
      y: Math.random() * height,
      z,
      r: (0.35 + Math.random() * 0.75) * (0.6 + z * 0.7),
      phase: Math.random() * Math.PI * 2,
      speed: 0.4 + Math.random() * 1.2,
    };
  });
}

function scrollParent(node: HTMLElement): HTMLElement | null {
  for (let el = node.parentElement; el; el = el.parentElement) {
    const { overflowY } = getComputedStyle(el);
    if (overflowY === "auto" || overflowY === "scroll") return el;
  }
  return null;
}

export function Starfield({ className }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let width = 0;
    let height = 0;
    let stars: Star[] = [];
    let ink = "255,255,255";
    let pointerX = 0;
    let pointerY = 0;
    let easedX = 0;
    let easedY = 0;
    let frame = 0;
    let visible = true;
    const scroller = scrollParent(canvas);

    const readInk = () => {
      const value = getComputedStyle(document.documentElement).getPropertyValue("--ink").trim();
      const hex = value.startsWith("#") && value.length === 7 ? value.slice(1) : "f7f8f8";
      ink = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)).join(",");
    };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      stars = makeStars(width, height);
    };

    const draw = (time: number) => {
      const t = time / 1000;
      easedX += (pointerX - easedX) * 0.04;
      easedY += (pointerY - easedY) * 0.04;
      const scroll = scroller ? scroller.scrollTop : window.scrollY;
      ctx.clearRect(0, 0, width, height);

      const placed = stars.map((s) => {
        const drift = reduced ? 0 : t * 4 * s.z;
        const x = (((s.x + drift + easedX * 18 * s.z) % width) + width) % width;
        const y = (((s.y - scroll * 0.25 * s.z + easedY * 12 * s.z) % height) + height) % height;
        const twinkle = reduced ? 0.8 : 0.55 + 0.45 * Math.sin(t * s.speed + s.phase);
        return { x, y, s, alpha: (0.25 + 0.6 * s.z) * twinkle };
      });

      // Faint links between near neighbours in the front layer.
      ctx.lineWidth = 0.6;
      const near = placed.filter((p) => p.s.z === 1);
      for (let i = 0; i < near.length; i++) {
        for (let j = i + 1; j < near.length; j++) {
          const dx = near[i].x - near[j].x;
          const dy = near[i].y - near[j].y;
          const d = Math.hypot(dx, dy);
          if (d > LINK_DISTANCE) continue;
          ctx.strokeStyle = `rgba(${ink},${0.09 * (1 - d / LINK_DISTANCE)})`;
          ctx.beginPath();
          ctx.moveTo(near[i].x, near[i].y);
          ctx.lineTo(near[j].x, near[j].y);
          ctx.stroke();
        }
      }

      for (const p of placed) {
        ctx.fillStyle = `rgba(${ink},${p.alpha})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.s.r, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    const loop = (time: number) => {
      draw(time);
      if (visible && !document.hidden) frame = requestAnimationFrame(loop);
    };
    const start = () => {
      cancelAnimationFrame(frame);
      if (reduced) draw(0);
      else frame = requestAnimationFrame(loop);
    };

    const onPointer = (e: PointerEvent) => {
      pointerX = e.clientX / window.innerWidth - 0.5;
      pointerY = e.clientY / window.innerHeight - 0.5;
    };
    const onVisibility = () => {
      if (!document.hidden && visible) start();
    };

    readInk();
    resize();
    start();

    const observer = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) start();
    });
    observer.observe(canvas);
    const resizeObserver = new ResizeObserver(() => {
      resize();
      if (reduced) draw(0);
    });
    resizeObserver.observe(canvas);
    const themeObserver = new MutationObserver(() => {
      readInk();
      if (reduced) draw(0);
    });
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    window.addEventListener("pointermove", onPointer, { passive: true });
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      resizeObserver.disconnect();
      themeObserver.disconnect();
      window.removeEventListener("pointermove", onPointer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return <canvas ref={canvasRef} aria-hidden className={className} />;
}
