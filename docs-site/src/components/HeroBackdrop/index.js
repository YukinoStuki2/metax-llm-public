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

      // soft vignette with multiple layers
      const t = state.t;
      const cx = width * (0.55 + 0.06 * Math.sin(t / 480));
      const cy = height * (0.25 + 0.04 * Math.cos(t / 580));
      
      // Primary purple glow
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(width, height) * 0.85);
      g.addColorStop(0, 'rgba(99,102,241,0.18)');
      g.addColorStop(0.4, 'rgba(139,92,246,0.12)');
      g.addColorStop(0.7, 'rgba(34,211,238,0.08)');
      g.addColorStop(1, 'rgba(2,6,23,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, width, height);
      
      // Secondary cyan accent
      const cx2 = width * (0.75 + 0.05 * Math.cos(t / 420));
      const cy2 = height * (0.65 + 0.04 * Math.sin(t / 520));
      const g2 = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, Math.max(width, height) * 0.5);
      g2.addColorStop(0, 'rgba(34,211,238,0.10)');
      g2.addColorStop(0.5, 'rgba(99,102,241,0.05)');
      g2.addColorStop(1, 'rgba(2,6,23,0)');
      ctx.fillStyle = g2;
      ctx.fillRect(0, 0, width, height);
      
      // Tertiary pink accent
      const cx3 = width * (0.20 + 0.04 * Math.sin(t / 550));
      const cy3 = height * (0.80 + 0.03 * Math.cos(t / 480));
      const g3 = ctx.createRadialGradient(cx3, cy3, 0, cx3, cy3, Math.max(width, height) * 0.4);
      g3.addColorStop(0, 'rgba(236,72,153,0.08)');
      g3.addColorStop(1, 'rgba(2,6,23,0)');
      ctx.fillStyle = g3;
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
        // Gradient color based on position
        const hue = 240 + (p.x / width) * 30;
        const sat = 70 + (p.y / height) * 20;
        ctx.fillStyle = `hsla(${hue}, ${sat}%, 70%, 0.45)`;
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
        
        // Add subtle glow to larger points
        if (p.r > 1.5) {
          ctx.beginPath();
          ctx.fillStyle = `hsla(${hue}, ${sat}%, 70%, 0.15)`;
          ctx.arc(p.x, p.y, p.r * 2.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // links with gradient effect
      ctx.lineWidth = 1.2;
      for (let i = 0; i < points.length; i++) {
        const a = points[i];
        for (let j = i + 1; j < points.length; j++) {
          const b = points[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const d2 = dx * dx + dy * dy;
          const maxD = 200;
          if (d2 > maxD * maxD) continue;
          const alpha = 1 - Math.sqrt(d2) / maxD;
          
          // Create gradient for lines
          const gradient = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
          gradient.addColorStop(0, `rgba(99,102,241,${0.22 * alpha})`);
          gradient.addColorStop(0.5, `rgba(34,211,238,${0.18 * alpha})`);
          gradient.addColorStop(1, `rgba(139,92,246,${0.22 * alpha})`);
          
          ctx.strokeStyle = gradient;
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
