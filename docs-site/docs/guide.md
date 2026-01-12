---
sidebar_position: 2
title: 文档导航（阅读路径）
---

本仓库的文档更像“工具箱手册”。写了很多脚本的详细情况，内容较多可以按照本页寻找重点。

---

## 阅读须知

- 每一份文档都有对应Github仓库内的一份或多份脚本
- 其余文件均为辅助性脚本，不影响评测最终得分，但是对于调试过程有很大的帮助

- 大部分脚本都设置了不少的环境变量和参数，本文档的作用在于查询参数作用和具体代码细节

## 下一步

- 如果想直接启动，请看“快速开始”。

- 如果想查询具体参数，可以在以下卡片内寻找需要查询的文档进入

- 文件总览：见“文件与目录说明（全仓库）”。

---

## 入口（卡片导航）

<div className="neoGrid">
  <a className="neoCard" href="/docs/quickstart">
    <div className="neoCardTitle">快速开始</div>
    <div className="neoCardDesc">最快跑通：启动后端、探活、跑一次本地评测。</div>
    <div className="neoCardMeta">第一次来先看这里</div>
  </a>

  <a className="neoCard" href="/docs/files">
    <div className="neoCardTitle">文件与目录说明</div>
    <div className="neoCardDesc">全仓库文件/脚本用途索引：从入口脚本到调参/量化工具。</div>
    <div className="neoCardMeta">找文件先看这里</div>
  </a>

  <a className="neoCard" href="/docs/service/serve">
    <div className="neoCardTitle">推理服务（Judge 契约）</div>
    <div className="neoCardDesc">GET / 与 POST /predict；评测 run 阶段断网等约束。</div>
    <div className="neoCardMeta">评测链路核心</div>
  </a>

  <a className="neoCard" href="/docs/category/启动与脚本">
    <div className="neoCardTitle">启动与脚本</div>
    <div className="neoCardDesc">run_model.sh / env_force.sh / judge.sh：本地复现评测流程与常用工具。</div>
    <div className="neoCardMeta">复现与调试</div>
  </a>

  <a className="neoCard" href="/docs/category/评测">
    <div className="neoCardTitle">评测</div>
    <div className="neoCardDesc">eval_local.py：按评测口径计算 RougeL-F1，并辅助观察吞吐。</div>
    <div className="neoCardMeta">优化前先会测</div>
  </a>

  <a className="neoCard" href="/docs/category/webui">
    <div className="neoCardTitle">WebUI</div>
    <div className="neoCardDesc">Gradio 调试入口：参数透传、batch 测试、（可选）RAG。</div>
    <div className="neoCardMeta">交互式排障</div>
  </a>

  <a className="neoCard" href="/docs/category/自动调参">
    <div className="neoCardTitle">自动调参</div>
    <div className="neoCardDesc">auto_tune：自动搜索推理参数组合，记录结果并选最优。</div>
    <div className="neoCardMeta">省时省力</div>
  </a>

  <a className="neoCard" href="/docs/category/模型工程">
    <div className="neoCardTitle">模型工程</div>
    <div className="neoCardDesc">下载 / 融合 LoRA / 上传：围绕 ModelScope 的工程脚本与规范。</div>
    <div className="neoCardMeta">产物管理</div>
  </a>

  <a className="neoCard" href="/docs/category/量化">
    <div className="neoCardTitle">量化</div>
    <div className="neoCardDesc">AWQ 量化 + 校准集生成：探索更高吞吐的路线之一。</div>
    <div className="neoCardMeta">性能优化</div>
  </a>
</div>

---
