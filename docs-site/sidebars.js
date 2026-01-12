// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  tutorialSidebar: [
    'guide',
    'intro',
    'quickstart',
    {
      type: 'category',
      label: '交付与同步',
      link: {type: 'generated-index', description: '面向评测交付与 GitHub→Gitee 同步说明。'},
      items: ['gitee-sync', 'files'],
    },
    {
      type: 'category',
      label: '推理服务',
      link: {type: 'generated-index', description: '后端服务（serve.py）与 Judge API 契约。'},
      items: ['service/serve', 'service/serve_env', 'service/serve_code', 'service/serve_faq'],
    },
    {
      type: 'category',
      label: '启动与脚本',
      link: {type: 'generated-index', description: '本地/云主机启动、评测模拟与常用脚本。'},
      items: ['scripts/run-model', 'scripts/judge'],
    },
    {
      type: 'category',
      label: 'WebUI',
      link: {type: 'generated-index', description: 'Gradio WebUI：调试、参数透传、可选 RAG、batch 测试入口。'},
      items: ['webui/overview', 'webui/start-webui'],
    },
    {
      type: 'category',
      label: '评测',
      link: {type: 'generated-index', description: '本地评测与结果解释（RougeL-F1 / tokens/s）。'},
      items: ['eval/eval_local'],
    },
    {
      type: 'category',
      label: '自动调参',
      link: {type: 'generated-index', description: 'auto_tune：自动搜索推理参数组合，平衡准确率与吞吐。'},
      items: ['tuning/auto_tune'],
    },
    {
      type: 'category',
      label: '模型工程',
      link: {type: 'generated-index', description: '下载/融合/上传模型的工程脚本与规范。'},
      items: ['model/download_model', 'model/merge_adapter', 'model/upload_model'],
    },
    {
      type: 'category',
      label: '量化',
      link: {type: 'generated-index', description: 'AWQ 量化与校准集生成。'},
      items: ['quant/awq', 'quant/sample_calib'],
    },
  ],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
   */
};

export default sidebars;
