---
sidebar_position: 3
---

# Gitee 同步（评测用精简仓库）

本项目以 GitHub 仓库为“源仓库”。为了满足评测平台要求，会把少量必要文件同步到 Gitee（并裁剪掉其余文件）。

同步机制（概念说明）：

- GitHub Actions 在每次 push 到 `master` 时触发
- runner 上执行“白名单裁剪”：只保留评测必须的顶层文件
- 将裁剪结果提交为一次临时提交，并强制推送到 Gitee 的 `master`

文档同步策略：

- GitHub 主仓库可保留完整开发文档
- 同步到 Gitee 时只同步精简版 `README_GITEE.md`，并在同步过程中重命名为 `README.md`

注意事项：

- Run 阶段断网：服务请求路径内不要做任何联网探测
- 评测 API 契约不可破坏：`GET /` 与 `POST /predict`
