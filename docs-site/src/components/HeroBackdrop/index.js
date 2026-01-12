import React, {useEffect, useRef} from 'react';

import styles from './styles.module.css';

function prefersReducedMotion() {
  if (typeof window === 'undefined') return true;
  return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

export default function HeroBackdrop() {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const stateRef = useRef({
    width: 0,
    height: 0,
    dpr: 1,
    t: 0,
    points: [],
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    const reduced = prefersReducedMotion();
    const ctx = canvas.getContext('2d');
    if (!ctx) return undefined;

    const state = stateRef.current;

    const resize = () => {
      const parent = canvas.parentElement;
      const rect = (parent || canvas).getBoundingClientRect();
      const dpr = clamp(window.devicePixelRatio || 1, 1, 2);
      state.dpr = dpr;
      state.width = Math.max(1, Math.floor(rect.width));
      state.height = Math.max(1, Math.floor(rect.height));

      canvas.width = Math.floor(state.width * dpr);
      canvas.height = Math.floor(state.height * dpr);
      canvas.style.width = `${state.width}px`;
      canvas.style.height = `${state.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const targetCount = clamp(Math.floor((state.width * state.height) / 25000), 18, 70);
      if (!state.points || state.points.length !== targetCount) {
        state.points = new Array(targetCount).fill(0).map(() => {
          const x = Math.random() * state.width;
          const y = Math.random() * state.height;
          const r = 0.8 + Math.random() * 1.8;
          const vx = (Math.random() - 0.5) * 0.18;
          const vy = (Math.random() - 0.5) * 0.18;
          return {x, y, r, vx, vy};
        });
      }
    };

    const draw = () => {
      const {width, height, points} = state;
      state.t += 1;

      ctx.clearRect(0, 0, width, height);

      // soft vignette
      const g = ctx.createRadialGradient(width * 0.55, height * 0.25, 0, width * 0.55, height * 0.25, Math.max(width, height));
      g.addColorStop(0, 'rgba(99,102,241,0.12)');
      g.addColorStop(0.55, 'rgba(34,211,238,0.08)');
      g.addColorStop(1, 'rgba(2,6,23,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, width, height);

      // points
      for (const p of points) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < -20) p.x = width + 20;
        if (p.x > width + 20) p.x = -20;
        if (p.y < -20) p.y = height + 20;
        if (p.y > height + 20) p.y = -20;

        ctx.beginPath();
        ctx.fillStyle = 'rgba(148,163,184,0.30)';
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }

      // links
      ctx.lineWidth = 1;
      for (let i = 0; i < points.length; i++) {
        const a = points[i];
        for (let j = i + 1; j < points.length; j++) {
          const b = points[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const d2 = dx * dx + dy * dy;
          const maxD = 170;
          if (d2 > maxD * maxD) continue;
          const alpha = 1 - Math.sqrt(d2) / maxD;
          ctx.strokeStyle = `rgba(99,102,241,${0.16 * alpha})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }

      rafRef.current = window.requestAnimationFrame(draw);
    };

    const start = () => {
      resize();
      if (!reduced) {
        rafRef.current = window.requestAnimationFrame(draw);
      } else {
        // Reduced motion: paint once.
        draw();
        if (rafRef.current) {
          window.cancelAnimationFrame(rafRef.current);
          rafRef.current = null;
        }
      }
    };

    const ro = new ResizeObserver(() => resize());
    if (canvas.parentElement) ro.observe(canvas.parentElement);

    start();

    return () => {
      ro.disconnect();
      if (rafRef.current) window.cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, []);

  return (
    <div className={styles.wrap} aria-hidden="true">
      <canvas ref={canvasRef} className={styles.canvas} />
      <div className={styles.noise} />
    </div>
  );
}
