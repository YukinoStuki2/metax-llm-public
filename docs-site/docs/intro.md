---
sidebar_position: 1
---

# 项目介绍

本仓库用于“微调开源大模型进行问答评测”，并提供符合评测系统契约的推理服务（HTTP）。

本项目来自 **2025 年秋季中国科学院大学《GPU架构与编程》课程项目二**，并在「沐曦赛道」获得 **摩尔线程一等奖**。

相关仓库：

- Gitee（用于实际提交/评测）：[https://gitee.com/yukinostuki/metax-demo](https://gitee.com/yukinostuki/metax-demo)
- GitHub（用于开发）：[https://github.com/YukinoStuki2/metax-llm-public](https://github.com/YukinoStuki2/metax-llm-public)

> 两个仓库仅文件数量不同（Gitee 为评测白名单裁剪版），代码与版本保持同步。

## 目标与评分

- **准确率**：RougeL-F1（jieba 分词），常见参考阈值 ≥ 0.35
- **吞吐**：尽量提高 tokens/s（在限定时间内完成更多题目）

微调数据集来源于《Programming Massively Parallel Processors.2017》。

## 评测契约（不要破坏）

- `GET /`：健康检查（必须快速返回）
- `POST /predict`：请求 `{"prompt":"..."}`，响应 `{"response":"..."}`
- 端口：固定 `8000`

本项目也支持 batch：当 `BATCH_MODE=1` 时，`GET /` 返回 `{"status":"batch"}`，评测系统可能会以数组形式一次性推送所有问题到 `POST /predict`。

## 关键约束

- Build 阶段允许联网（下载依赖/权重）
- Run 阶段断网（请求路径内不要做任何联网操作）

## 下一步

- 从「快速开始」开始：Docker 启动 / WebUI 启动 / API 自测
- 需要本地复现或调参：看「启动脚本（run_model.sh / env_force.sh）」与「推理服务（serve.py）」

文档站：[https://docs-gpu.yukino.uk](https://docs-gpu.yukino.uk)
