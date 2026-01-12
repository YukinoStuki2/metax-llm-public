import React, {useEffect, useRef} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: '推理服务',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        vllm、Batch处理、<b>Token Routing</b> 、数据集预热、文本截断、
        <b>Qwen2.5-0.5B</b>全参微调、cuda与沐曦支持。
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
        用本地脚本模拟评测机调用方式，输出准确率与Token/s。
        可选基础题和加分题、是否Batch、单题准确率与Token。
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
        联网查询、本地知识库、固定URL知识寻找、推理测信息显示。
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
        支持 AWQ 量化、上传到 ModelScope、以及断点续跑自动调参脚本。
        自动调参，邮件和飞书机器人通知，最好参数保存与对比。
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
