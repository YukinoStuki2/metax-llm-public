---
sidebar_position: 1
---

# 项目介绍

本仓库用于“微调开源大模型进行问答评测”，并提供WebUI进行调试和使用。

本项目来自 **2025 年秋季中国科学院大学《GPU架构与编程》课程项目二**，并在「沐曦赛道」获得 **摩尔线程一等奖**。

相关仓库：

- Gitee（用于实际提交/评测）：[https://gitee.com/yukinostuki/metax-demo](https://gitee.com/yukinostuki/metax-demo)

- GitHub（用于开发）：[https://github.com/YukinoStuki2/metax-llm-public](https://github.com/YukinoStuki2/metax-llm-public)

> 两个仓库仅文件数量不同（Gitee 为评测白名单裁剪版），代码与版本保持同步。

## 模型仓库

[YukinoStuki/Qwen2.5-0.5B-Plus-LLM （快速-适用于评测高吞吐）](https://modelscope.cn/models/YukinoStuki/Qwen2.5-0.5B-Plus-LLM)

[YukinoStuki/Qwen3-4B-Plus-LLM](https://modelscope.cn/models/YukinoStuki/Qwen3-4B-Plus-LLM)[（智能-不过拟合-可以回答书本外的问题）](https://modelscope.cn/models/YukinoStuki/Qwen2.5-0.5B-Plus-LLM)

还有其他微调的大模型如**英语版**、1.7B、不同数据集等，详情可以看仓库**README**有介绍

均使用**LlamaFactory**进行微调，数据集使用**Easy-dataset**进行制作，制作时使用了ChatGPT-5.2Pro的API。

仓库里有一份用于评测的数据集data.jsonl，此外还有英语数据集和其他几套数据集没有传入仓库内。

## 目标与评分

- **准确率**：RougeL-F1（jieba 分词），常见参考阈值 ≥ 0.35

- **吞吐**：尽量提高 tokens/s（在限定时间内完成更多题目）

微调数据集来源于《Programming Massively Parallel Processors.2017》

## 评测接口

- `GET /`：健康检查（必须快速返回）

- `POST /predict`：请求 `{"prompt":"..."}`，响应 `{"response":"..."}`

- 端口：固定 `8000`

本项目也支持 batch：当 `BATCH_MODE=1` 时，`GET /` 返回 `{"status":"batch"}`，评测系统可能会以数组形式一次性推送所有问题到 `POST /predict`。

## 关键约束

- Build 阶段允许联网（下载依赖/权重）

- Run 阶段断网

- 评测机配置：

  - OS：Ubuntu 24.04

  - CPU：24 核

  - 内存：200GB

  - 磁盘：1TB

  - GPU：MXC500（64GB 显存）

  - 网络：100Mbps

- 时间限制

  * docker build：900s

  * health（`GET /`）：180s

  * predict（推理阶段总计）：360s

## 下一步

- 从「快速启动」开始：Docker 启动 / WebUI 启动 / API 自测

- 需要本地复现或调参：看「启动脚本（run_model.sh / env_force.sh）」与「推理服务（serve.py）」

文档站：[https://docs-gpu.yukino.uk](https://docs-gpu.yukino.uk)