import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description={siteConfig.tagline}>
      <main style={{padding: '3rem 0'}}>
        <div className="container">
          <h1>{siteConfig.title}</h1>
          <p>{siteConfig.tagline}</p>
          <div style={{display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginTop: '1rem'}}>
            <Link className="button button--primary" to="/docs/intro">
              开始阅读
            </Link>
            <Link className="button button--secondary" to="/docs/quickstart">
              快速启动
            </Link>
            <Link className="button button--secondary" href="https://github.com/YukinoStuki2/metax-demo-mirror">
              GitHub
            </Link>
          </div>
        </div>
      </main>
    </Layout>
  );
}
