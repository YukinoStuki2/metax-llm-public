import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx(styles.hero)}>
      <div className={styles.heroInner}>
        <div className={styles.heroContent}>
          <div className={styles.eyebrow}>metax-llm-public · 推理服务与评测工程</div>
          <h1 className={styles.title}>{siteConfig.title}</h1>
          <p className={styles.subtitle}>{siteConfig.tagline}</p>

          <div className={styles.ctaRow}>
            <Link className={clsx('button button--lg', styles.ctaPrimary)} to="/docs/quickstart">
              快速开始
            </Link>
            <Link className={clsx('button button--lg', styles.ctaSecondary)} to="/docs/service/serve">
              查看 API 契约
            </Link>
            <Link
              className={clsx('button button--lg', styles.ctaGhost)}
              href="https://github.com/YukinoStuki2/metax-llm-public"
            >
              GitHub
            </Link>
          </div>

          <div className={styles.note}>
            评测 Run 阶段断网：服务端请求路径中不要引入任何外部网络调用。
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
