---
sidebar_position: 2
title: 文档导航（阅读路径）
---

本仓库的文档更像“工具箱手册”。为了避免页面散落、找不到入口，建议按下面的阅读路径使用。

<div class="neoGrid">
  <a class="neoCard" href="/docs/quickstart">
    <div class="neoCardTitle">快速开始</div>
    <div class="neoCardDesc">最快跑通：启动后端、WebUI、自测 API、理解 batch。</div>
    <div class="neoCardMeta">推荐新手从这里开始</div>
  </a>

  <a class="neoCard" href="/docs/service/serve">
    <div class="neoCardTitle">推理服务（Judge 契约）</div>
    <div class="neoCardDesc">必须不破坏的接口：GET / 与 POST /predict，以及 batch 行为。</div>
    <div class="neoCardMeta">评测链路核心</div>
  </a>

  <a class="neoCard" href="/docs/scripts/run-model">
    <div class="neoCardTitle">启动与参数对齐</div>
    <div class="neoCardDesc">run_model.sh / env_force.sh：本地复现评测环境，管理环境变量与默认值。</div>
    <div class="neoCardMeta">复现与调试必备</div>
  </a>

  <a class="neoCard" href="/docs/eval/eval_local">
    <div class="neoCardTitle">本地评测（RougeL）</div>
    <div class="neoCardDesc">eval_local.py：模拟评测机口径，快速回归准确率与吞吐。</div>
    <div class="neoCardMeta">优化前先会测</div>
  </a>

  <a class="neoCard" href="/docs/webui/overview">
    <div class="neoCardTitle">WebUI 调试</div>
    <div class="neoCardDesc">生成参数透传、SYSTEM_PROMPT、Batch 测试入口与可选 RAG。</div>
    <div class="neoCardMeta">交互式排障与演示</div>
  </a>

  <a class="neoCard" href="/docs/tuning/auto_tune">
    <div class="neoCardTitle">自动调参</div>
    <div class="neoCardDesc">auto_tune：循环试参、跑评测、选最优；支持断点续跑与通知。</div>
    <div class="neoCardMeta">省时省力找最优</div>
  </a>

  <a class="neoCard" href="/docs/model/download_model">
    <div class="neoCardTitle">模型工程</div>
    <div class="neoCardDesc">下载 / 融合 LoRA / 上传：围绕 ModelScope 的工程脚本与规范。</div>
    <div class="neoCardMeta">产物管理与发布</div>
  </a>

  <a class="neoCard" href="/docs/quant/awq">
    <div class="neoCardTitle">量化（AWQ）</div>
    <div class="neoCardDesc">AutoAWQ 4bit + 校准集抽样：在约束下探索更高吞吐。</div>
    <div class="neoCardMeta">性能优化路线之一</div>
  </a>
</div>

---

## 推荐阅读路径

- 第一次来：快速开始 → 推理服务（契约） → 启动与参数对齐 → 本地评测
- 想提升成绩：本地评测 → 自动调参（或手动调参） → 必要时尝试量化
- 想做交互演示：启动后端 → 启动 WebUI → 用 Batch 测试快速检查

## 交付与同步（评测提交）

- 如果你要向评测平台提交：先看 Gitee 同步与文件白名单裁剪逻辑。
- 文件总览：见“文件与目录说明（全仓库）”。
