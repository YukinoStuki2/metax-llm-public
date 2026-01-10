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
    'intro',
    'quickstart',
    {
      type: 'category',
      label: '交付与同步',
      items: ['gitee-sync', 'files'],
    },
    {
      type: 'category',
      label: '推理服务',
      items: ['service/serve'],
    },
    {
      type: 'category',
      label: '脚本与工具',
      items: ['scripts/run-model', 'scripts/judge'],
    },
    {
      type: 'category',
      label: 'WebUI',
      items: ['webui/overview'],
    },
    {
      type: 'category',
      label: '评测',
      items: ['eval/eval_local'],
    },
    {
      type: 'category',
      label: '自动调参',
      items: ['tuning/auto_tune'],
    },
    {
      type: 'category',
      label: '模型工程',
      items: ['model/download_model', 'model/merge_adapter', 'model/upload_model'],
    },
    {
      type: 'category',
      label: '量化',
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
