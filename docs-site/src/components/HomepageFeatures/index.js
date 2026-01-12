import React, {useEffect, useRef} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: '推理服务（Judge 契约）',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        核心接口固定：<b>GET /</b> 健康检查、<b>POST /predict</b> 预测。务必保持快速返回、
        并在响应里剔除 <code>&lt;think&gt;...&lt;/think&gt;</code>。
        <div className={styles.links}>
          <Link to="/docs/service/serve">接口与服务说明</Link>
          <span className={styles.dot}>·</span>
          <Link to="/docs/scripts/run-model">启动脚本与参数对齐</Link>
        </div>
      </>
    ),
  },
  {
    title: '本地评测（RougeL）',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        用本地脚本模拟评测机调用方式，快速回归效果与吞吐。
        评测输出为空会直接记 0 分。
        <div className={styles.links}>
          <Link to="/docs/eval/eval_local">eval_local 使用说明</Link>
          <span className={styles.dot}>·</span>
          <Link to="/docs/quickstart">一键跑通流程</Link>
        </div>
      </>
    ),
  },
  {
    title: 'WebUI 调试与 RAG',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        提供参数透传、System Prompt 管理、Batch 测试入口，并可按需启用 RAG。
        评测环境 Run 阶段断网，请保持默认关闭或确保不触网。
        <div className={styles.links}>
          <Link to="/docs/webui/overview">WebUI 说明</Link>
        </div>
      </>
    ),
  },
  {
    title: '量化 / 上传 / 自动调参',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        支持 AWQ 量化、上传到 ModelScope、以及自动调参脚本（断点续跑）。
        <div className={styles.links}>
          <Link to="/docs/quant/awq">AWQ 量化</Link>
          <span className={styles.dot}>·</span>
          <Link to="/docs/model/upload_model">上传模型</Link>
          <span className={styles.dot}>·</span>
          <Link to="/docs/tuning/auto_tune">自动调参</Link>
        </div>
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) {
      el.dataset.visible = 'true';
      return undefined;
    }

    const io = new IntersectionObserver(
      (entries) => {
        for (const ent of entries) {
          if (ent.isIntersecting) {
            ent.target.dataset.visible = 'true';
            io.unobserve(ent.target);
          }
        }
      },
      {root: null, rootMargin: '0px 0px -10% 0px', threshold: 0.15}
    );

    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div ref={ref} className={clsx('col col--6', styles.col)} data-visible="false">
      <div className={styles.featureCard}>
        <div className={styles.iconWrap}>
          <Svg className={styles.featureSvg} role="img" />
        </div>
        <div className={styles.featureBody}>
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
