// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'metax-llm-public 文档',
  tagline: 'LLM 评测推理服务与工程说明',
  favicon: 'img/YukinoStuki-icon.png',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // 站点线上地址（Cloudflare Pages + 自定义域名时请改成你的域名，例如 https://docs.example.com）
  url: 'https://docs-gpu.yukino.uk',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  onBrokenLinks: 'throw',

  // 本站默认中文
  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // 将文档挂在 /docs，保留站点根路径作为主页
          routeBasePath: 'docs',
          editUrl: 'https://github.com/YukinoStuki2/metax-llm/tree/master/docs-site/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        // 默认进入暗色模式（首次访问）。用户仍可通过切换按钮覆盖（localStorage 会记住）。
        defaultMode: 'dark',
        respectPrefersColorScheme: false,
      },
      navbar: {
        title: 'metax-llm-public',
        logo: {
          alt: 'metax-llm-public',
          src: 'img/YukinoStuki-icon.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: '文档',
          },
            {
              to: '/docs/guide',
              label: '导航',
              position: 'left',
            },
          {
            to: '/docs/quickstart',
            label: '快速启动',
            position: 'left',
          },
          {
            to: '/docs/service/serve',
            label: '服务',
            position: 'left',
          },
          {
            to: '/docs/eval/eval_local',
            label: '评测',
            position: 'left',
          },
          {
            to: '/docs/tuning/auto_tune',
            label: '调参',
            position: 'left',
          },
          {
            to: '/docs/webui/overview',
            label: 'WebUI',
            position: 'left',
          },
          {
            href: 'https://github.com/YukinoStuki2/metax-llm-public',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Links',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/YukinoStuki2/metax-llm-public',
              },
              {
                label: 'yukinostuki@qq.com',
                href: 'mailto:yukinostuki@qq.com',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} YukinoStuki2. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
