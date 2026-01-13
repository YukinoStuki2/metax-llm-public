import React, {useEffect, useRef} from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import HeroBackdrop from '@site/src/components/HeroBackdrop';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  const heroRef = useRef(null);

  useEffect(() => {
    const el = heroRef.current;
    if (!el) return undefined;

    const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) return undefined;

    const onMove = (e) => {
      const r = el.getBoundingClientRect();
      const x = (e.clientX - r.left) / Math.max(1, r.width);
      const y = (e.clientY - r.top) / Math.max(1, r.height);
      el.style.setProperty('--mx', String(x));
      el.style.setProperty('--my', String(y));
    };

    const onLeave = () => {
      el.style.setProperty('--mx', '0.5');
      el.style.setProperty('--my', '0.35');
    };

    // init
    onLeave();

    el.addEventListener('pointermove', onMove, {passive: true});
    el.addEventListener('pointerleave', onLeave, {passive: true});
    return () => {
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerleave', onLeave);
    };
  }, []);

  return (
    <header ref={heroRef} className={clsx(styles.hero)}>
      <HeroBackdrop />
      <div className={styles.heroInner}>
        <div className={styles.heroContent}>
          <div className={styles.eyebrow}>metax-llm-public · 推理服务与评测工程</div>
          <h1 className={styles.title}>{siteConfig.title}</h1>
          <p className={styles.subtitle}>{siteConfig.tagline}</p>

          <div className={styles.ctaRow}>
            <Link className={clsx('button button--lg', styles.ctaPrimary)} to="/docs/guide">
              开始
            </Link>
            <Link className={clsx('button button--lg', styles.ctaSecondary)} to="/docs/quickstart">
              快速启动
            </Link>
            <Link
              className={clsx('button button--lg', styles.ctaGhost)}
              href="https://github.com/YukinoStuki2/metax-llm-public"
            >
              GitHub
            </Link>
          </div>

          <div className={styles.note}>
            <span className={styles.noteText}>
              2025年秋季中国科学院大学《GPU架构与编程》摩尔线程一等奖
            </span>
          </div>
        </div>

        <div className={styles.heroPanel} aria-hidden="true">
          <div className={styles.panelGlow} />
          <div className={styles.panelGrid} />
          <div className={styles.panelChip}>
            <div className={styles.panelChipTitle}>Judge Contract</div>
            <div className={styles.panelChipBody}>
              <div className={styles.kvRow}>
                <span className={styles.kvKey}>GET /</span>
                <span className={styles.kvVal}>health</span>
              </div>
              <div className={styles.kvRow}>
                <span className={styles.kvKey}>POST /predict</span>
                <span className={styles.kvVal}>{'{"prompt":"..."}'}</span>
              </div>
              <div className={styles.kvRow}>
                <span className={styles.kvKey}>port</span>
                <span className={styles.kvVal}>8000</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  return (
    <Layout title="主页" description="LLM 评测推理服务与工程说明">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
