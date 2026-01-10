# docs-site（Docusaurus 文档站）

本目录是本仓库的文档站点源代码，使用 Docusaurus 构建静态站点，推荐部署到 Cloudflare Pages 并绑定自定义域名。

## 本地开发（WSL / Linux）

如果你的 WSL 环境没有 Linux 版 Node.js，但系统里能看到 Windows 的 npm/npx（通常在 `/mnt/c` 或 `/mnt/d`），请不要使用它们（会触发 CMD.EXE 并尝试写入 Windows 路径）。

本仓库提供“仅在仓库内生效”的 Linux Node 工具链示例：

```bash
cd /home/yukinostuki/metax-llm-public
export PATH="$PWD/.tools/node/bin:$PATH"

cd docs-site
npm start
```

开发服务器默认端口 3000：打开 http://127.0.0.1:3000/

## 构建

```bash
cd docs-site
npm run build
```

产物输出到 `docs-site/build/`。

## Cloudflare Pages 部署（根域名）

在 Cloudflare Pages 新建项目，连接此 GitHub 仓库。

推荐配置：

- Root directory：`docs-site`
- Build command：`npm ci && npm run build`
- Build output directory：`build`

Node 版本建议：20。可在 Cloudflare Pages 环境变量里设置：

- `NODE_VERSION=20`

重要：根域名部署时，`docusaurus.config.js` 里的 `baseUrl` 应保持为 `/`。
同时请把 `url` 改为你的真实域名（例如 `https://docs.example.com`）。
