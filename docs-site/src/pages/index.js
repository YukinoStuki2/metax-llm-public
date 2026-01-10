import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {Redirect} from '@docusaurus/router';

export default function Home() {
  // 根域名直接进入文档首页
  const {siteConfig} = useDocusaurusContext();
  void siteConfig;
  return <Redirect to="/docs/intro" />;
}
