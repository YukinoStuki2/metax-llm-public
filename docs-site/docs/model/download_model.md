---
title: 模型下载（download_model.py）
sidebar_position: 60
---

# 模型下载脚本详解

## 1) 脚本功能概述

### 1.1 作用与定位

`download_model.py` 用于在 **Docker build 阶段**（或任何“有网”的环境）从 ModelScope 拉取模型权重到本地目录。

本项目的评测/运行阶段通常是**断网**的：如果不在 build 阶段完成下载，`serve.py` 在 run 阶段将无法再在线拉取模型，从而导致启动失败。

### 1.2 能完成什么

- 下载一个“主模型”（必选）到本地 `cache_dir`。
- （可选）再下载一个 “draft 模型”（用于 speculative decoding 之类的加速方案），并可配置为“失败不阻塞构建”。
- 输出下载后的真实落盘目录（`Resolved model_dir: ...`），便于在 Dockerfile/启动脚本中引用。

### 1.3 输入与输出

**输入：**

- 模型标识：`--model_name`（如 `YukinoStuki/Qwen3-4B-Plus-Merged`）
- 版本：`--revision`（如 `master`）
- 下载目录：`--cache_dir`（如 `./model` 或 `/app/model`）
- （可选）ModelScope Token：`--token` 或环境变量 `MODELSCOPE_API_TOKEN`
- （可选）draft 模型标识与版本：`--draft_model_name` / `--draft_revision`

**输出：**

- 标准输出打印：
  - 是否使用 token 登录
  - 主模型下载成功/失败
  - 主模型与 draft 模型（若下载）的 `Resolved ..._dir`
- 进程退出码：
  - `0`：主模型下载成功，且（若启用）draft 模型下载成功或被允许失败
  - `1`：主模型下载失败
  - `2`：draft 模型下载失败且未启用“可选失败”策略

**会改变什么：**

- 在 `--cache_dir` 指定目录下写入/更新模型缓存文件（权重、配置、tokenizer 等）。
- 不修改项目源码、不修改运行时配置；仅产生（或更新）本地模型文件。

**不改变：**

- 不修改项目源码与运行时推理参数。

---

## 2) 参数与环境变量详解

本脚本所有可调项都来自命令行参数；其中一部分参数支持“默认从环境变量读取”。

### 2.1 主模型参数（必选项的默认值）

#### `--model_name`

- **作用**：要下载的主模型在 ModelScope 的标识。
- **默认值**：优先取环境变量 `MODEL_ID`；若未设置，则为 `YukinoStuki/Qwen3-4B-Plus-Merged`。
- **格式**：通常为 `组织/模型名`。
- **影响**：决定下载哪个模型。

#### `--revision`

- **作用**：主模型的版本（分支/Tag/Commit）。
- **默认值**：`master`
- **影响**：同一模型不同 revision 可能是不同权重；这会影响精度与体积。

#### `--cache_dir`

- **作用**：下载缓存目录（模型会被下载到此目录下，由 ModelScope SDK 决定具体层级）。
- **默认值**：`./`（当前目录）
- **建议**：在 Docker build 时，建议明确传入一个稳定目录（例如 `./model` 或 `/app/model`），以便后续脚本/服务稳定引用。

### 2.2 认证参数（可选）

#### `--token`

- **作用**：ModelScope API Token，用于下载私有模型或提高下载稳定性。
- **默认值**：环境变量 `MODELSCOPE_API_TOKEN`，若仍为空则匿名下载。
- **行为细节**：
  - 脚本会对 token 做 `strip()`，非空则调用 `HubApi().login(token)`。
  - token 不会被脚本打印到日志里（仅打印“Using ModelScope token ...”）。

### 2.3 Draft 模型参数（可选）

> Draft 模型在本项目中通常用于 speculative decoding 的实验/加速方案；评测并不强制要求。

#### `--draft_model_name`

- **作用**：额外下载的 draft 模型标识。
- **默认值**：环境变量 `SPEC_DRAFT_MODEL_ID`；为空则**跳过**下载 draft。
- **影响**：不影响主模型下载流程；只有该值非空时才会尝试下载。

#### `--draft_revision`

- **作用**：draft 模型版本。
- **默认值**：环境变量 `SPEC_DRAFT_MODEL_REVISION`；若未设置则为 `master`。

#### `--draft_optional`

- **作用**：控制 draft 下载失败时是否“允许继续”。
- **默认值**：
  - 这是一个 `store_true` 参数，但脚本额外设置了默认值：当环境变量 `SPEC_DRAFT_OPTIONAL` 为 `1` 时默认开启；否则默认关闭。
  - 换句话说：即使未显式传入 `--draft_optional`，它也可能默认为开启。
- **行为**：
  - 开启：draft 下载失败打印 warning 并继续，最终退出码仍可为 `0`
  - 关闭：draft 下载失败会 `sys.exit(2)`，使构建/流水线失败

### 2.4 环境变量

- `MODEL_ID`
  - 对应参数：`--model_name`
  - 作用：主模型 ID；未设置时使用脚本内置默认 `YukinoStuki/Qwen3-4B-Plus-Merged`。
- `MODELSCOPE_API_TOKEN`
  - 对应参数：`--token`
  - 作用：登录下载（私有/稳定）；为空则匿名。
- `SPEC_DRAFT_MODEL_ID`
  - 对应参数：`--draft_model_name`
  - 作用：draft 模型 ID；为空则跳过。
- `SPEC_DRAFT_MODEL_REVISION`
  - 对应参数：`--draft_revision`
  - 作用：draft 版本；未设置默认 `master`。
- `SPEC_DRAFT_OPTIONAL`
  - 对应参数：影响 `--draft_optional` 的默认值
  - 作用：draft 失败是否允许继续；`1` 表示默认开启。

---

## 3) 代码实现详解

本脚本的实现非常直接：解析参数 → （可选）登录 → 下载主模型 → （可选）下载 draft → 根据结果退出。

### 3.1 依赖库与它们的职责

- `argparse`：命令行参数解析。
- `os`：读取环境变量，为参数提供默认值。
- `sys`：在错误场景下设置退出码。
- `modelscope.snapshot_download`：核心下载函数，负责拉取权重并落盘。
- `modelscope.hub.api.HubApi`：用于 token 登录。

### 3.2 `parse_args()` 做了什么

`parse_args()` 定义了全部 CLI 参数，并实现“环境变量默认值注入”：

- `--model_name` 默认读取 `MODEL_ID`。
- `--token` 默认读取 `MODELSCOPE_API_TOKEN`。
- `--draft_model_name` 默认读取 `SPEC_DRAFT_MODEL_ID`。
- `--draft_revision` 默认读取 `SPEC_DRAFT_MODEL_REVISION`。
- `--draft_optional` 的默认值来源于 `SPEC_DRAFT_OPTIONAL == '1'`。

这一点很关键：它让 Dockerfile 或外部启动脚本可以只通过 `ENV ...` 就改变下载行为，而不必改命令行。

### 3.3 主流程（`__main__`）逐步解释

1. 解析参数，得到 `sh_args`。
2. 若 `token` 非空：
   - 调用 `HubApi().login(token)` 进行登录。
   - 打印提示：`Using ModelScope token for authenticated download.`
3. 下载主模型：
   - 调用 `snapshot_download(sh_args.model_name, cache_dir=..., revision=...)`。
   - 成功后打印：
     - `Model download successful!`
     - `Resolved model_dir: ...`
4. 若 `draft_model_name` 非空：尝试下载 draft：
   - 先打印 `[spec] Downloading draft model: ...`
   - 调用 `snapshot_download(draft_name, cache_dir=..., revision=...)`
   - 成功打印 draft 的 resolved 目录
   - 失败：
     - 若 `draft_optional=True`：打印“optional, continue”，不终止
     - 否则：打印失败并 `sys.exit(2)`
5. 任意主模型下载异常：打印失败并 `sys.exit(1)`。

### 3.4 与本项目离线运行约束的关系

- 评测/运行阶段断网时，**唯一可靠的方式**是把模型文件在 build 阶段就下载到镜像内。
- 这个脚本通常被 Dockerfile 调用，并配合 `--cache_dir` 把模型放到 `/app/model/...` 之类的固定路径。

---

## 4) 常见问题

### Q1：`--cache_dir` 应该设到哪里？

如果在 Dockerfile 里下载，建议把 `--cache_dir` 设到镜像内固定位置（例如 `./model` 或 `/app/model`），并确保 `serve.py`/启动脚本用相同路径加载。

### Q2：为什么脚本显示下载成功，但找不到模型目录？

`snapshot_download()` 的落盘结构由 ModelScope SDK 决定，最终真实路径会打印在 `Resolved model_dir: ...`。请以该输出为准，并把它作为后续加载的路径来源。

### Q3：已设置 `--token`，为什么还是下载失败？

常见原因：

- Token 无效/过期；重新生成后再试。
- 模型名或 revision 不存在。
- 网络/代理问题导致 SDK 无法访问。

### Q4：未显式传入 `--draft_optional`，为什么 draft 失败没有让构建失败？

因为 `--draft_optional` 的默认值受环境变量 `SPEC_DRAFT_OPTIONAL` 控制：当它为 `1` 时默认开启。若希望 draft 失败直接失败构建，请显式设置 `SPEC_DRAFT_OPTIONAL=0`，或确保环境变量不是 `1`。

### Q5：主模型下载失败时脚本的退出码是什么？

主模型下载失败会退出码 `1`。draft 下载失败在 `draft_optional=False` 时会退出码 `2`。

### Q6：能否只下载主模型，不下载 draft？

可以。保持 `--draft_model_name` 为空即可（默认就为空，或显式 `--draft_model_name ""` / 不设置 `SPEC_DRAFT_MODEL_ID`）。

### Q7：如何验证脚本已经使用 token 登录？

当 token 非空时脚本会打印：`Using ModelScope token for authenticated download.`。如果没看到这行，说明 `--token`/`MODELSCOPE_API_TOKEN` 仍为空或仅包含空白字符。
