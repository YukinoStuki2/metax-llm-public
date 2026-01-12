---
title: 上传模型到 ModelScope（upload_model.py）
sidebar_position: 80
---

# 上传模型脚本详解

## 1) 脚本功能概述

### 1.1 作用与定位

`upload_model.py` 用于把一个本地模型目录上传到 ModelScope 的指定仓库（repo）。它主要用于：

- 将量化后的模型、融合后的模型、或导出的 HF 目录结构上传到 ModelScope，便于后续在 Docker build 阶段通过 `snapshot_download` 拉取。

- 在模型迭代/调参过程中，把“可复现的模型目录”集中管理。

它不会参与线上评测的推理链路；属于构建与发布环节的工具脚本。

### 1.2 输入与输出

**输入：**

- `--repo-id`：目标 ModelScope 仓库 ID（例如 `User/RepoName`）

- `--model-dir`：本地要上传的目录路径

- （可选）`--token` 或环境变量 token：用于登录鉴权

- `--commit-message`：上传时的提交信息

**输出：**

- 控制台打印上传路径、目标 repo、成功或失败原因。

- 进程退出码：

  - 成功：`0`

  - 失败：`1`（包括参数缺失、目录不存在、上传异常）

**不改变：**

- 不修改本地目录内容（只读遍历上传）。

**会改变什么：**

- 会向 ModelScope 远端仓库上传文件（相当于一次提交）。

---

## 2) 参数与环境变量详解

### 2.1 命令行参数

#### `--repo-id`

- **作用**：目标 ModelScope 仓库 ID。

- **默认值**：优先读取环境变量 `REPO_ID`；若未设置，则默认 `YukinoStuki/Qwen3-4B-Plus-LLM-AWQ`。

- **要求**：不能为空；为空会直接报错并退出。

#### `--model-dir`

- **作用**：要上传的本地模型目录。

- **默认值**：优先读取环境变量 `MODEL_DIR`；若未设置，则默认 `model/YukinoStuki/Qwen3-4B-Plus-LLM-AWQ`。

- **要求**：路径必须存在；不存在会直接报错并退出。

- **路径处理**：支持 `~`（通过 `Path(...).expanduser()` 展开）。

#### `--token`

- **作用**：ModelScope API Token。

- **默认值**：空字符串（脚本不会内置任何 token）。

- **读取优先级**：

  1. 若命令行传了 `--token` 且非空：使用该 token

  2. 否则从环境变量读取：`MODELSCOPE_API_TOKEN` 或 `MODELSCOPE_TOKEN`

  3. 仍为空：不登录，尝试使用本机已有的 modelscope 登录缓存（如果存在）

- **影响**：

  - token 缺失时脚本仍会尝试上传，但更可能因权限不足失败。

  - 上传失败且未提供 token 时，脚本会打印明确提示。

#### `--commit-message`

- **作用**：上传时的 commit message。

- **默认值**：`upload model folder`

### 2.2 环境变量

- `REPO_ID`
   - 对应参数：覆盖 `--repo-id` 默认值
   - 作用：决定上传到哪个 repo。
- `MODEL_DIR`
   - 对应参数：覆盖 `--model-dir` 默认值
   - 作用：决定上传哪个本地目录。
- `MODELSCOPE_API_TOKEN`
   - 对应参数：用于 token 读取（优先）
   - 作用：登录与权限；与 `--token` 同时存在时以参数为准。
- `MODELSCOPE_TOKEN`
   - 对应参数：用于 token 读取（备选）
   - 作用：当 `MODELSCOPE_API_TOKEN` 未设置时使用。

---

## 3) 代码实现详解

### 3.1 依赖库与职责

- `argparse`：解析命令行参数。

- `os`：读取环境变量。

- `sys`：控制退出码与错误打印。

- `pathlib.Path`：路径处理与 `~` 展开。

- `modelscope.hub.api.HubApi`：登录与上传实现（`login` / `upload_folder`）。

### 3.2 关键函数与流程

#### `_env_token()`

- 从环境变量读取 token：优先 `MODELSCOPE_API_TOKEN`，其次 `MODELSCOPE_TOKEN`。

- 返回前会 `strip()`，避免只有空白字符的情况。

#### `_parse_args()`

- 定义 4 个参数：`--repo-id`、`--model-dir`、`--token`、`--commit-message`。

- `--repo-id` 与 `--model-dir` 支持用环境变量覆盖默认值：`REPO_ID`、`MODEL_DIR`。

- `--token` 默认空，强调“脚本不写死 token”。

#### `main()` 主流程

1. 解析参数并做规范化：

   - `repo_id = strip()`

   - `folder = Path(model_dir).expanduser()`

   - `token = args.token 或 _env_token()`

2. 参数校验：

   - `repo_id` 为空：直接退出码 `1`

   - `folder` 不存在：直接退出码 `1`

3. 轻量目录校验（warning 级别）：

   - 检查目录下是否至少包含 `config.json` 与 `tokenizer_config.json`

   - 缺失时仅打印 warning：

   - 上传的是“完整模型目录”时，这通常意味着导出目录不对

   - 上传的是 LoRA adapter 时，可忽略该 warning

4. 登录策略：

   - token 非空：`api.login(token)`

   - token 为空：打印 warning，仍尝试上传（依赖本机已有登录缓存或匿名权限）

5. 执行上传：

   - `api.upload_folder(repo_id=..., folder_path=..., commit_message=...)`

   - 失败会捕获异常并打印 `ERROR: 上传失败`，然后以退出码 `1` 结束

   - 若失败且未提供 token，会额外提示设置 `MODELSCOPE_API_TOKEN` 或使用 `--token`

6. 成功打印 `Upload finished.`。

---

## 4) 常见问题

### Q1：脚本提示“未提供 token”，是否可以忽略？

可以忽略，但只有在以下情况才可能成功：

- 运行环境已经通过 modelscope 的本地配置缓存登录过；或

- 目标 repo 允许匿名/无需鉴权的上传（通常不成立）。

更稳妥的方式是设置 `MODELSCOPE_API_TOKEN`（或命令行传 `--token`）。

### Q2：为什么提示目录缺少 `config.json` / `tokenizer_config.json`？

脚本做的是“轻量防呆”：完整 HF 模型目录通常至少包含这两个文件。

- 上传的是完整模型时：请检查 `--model-dir` 是否指向了正确的导出目录（例如融合/量化后的输出目录）。

- 上传的是 LoRA adapter 时：这两个文件本来就可能不存在，可以忽略 warning。

### Q3：上传失败后如何快速定位原因？

脚本会打印 `repr(e)`。

常见原因：

- token 权限不足或 repo_id 不存在

- 网络问题导致上传中断

- 本地目录包含不可读文件或路径异常

如果失败时未提供 token，优先按提示补充 `MODELSCOPE_API_TOKEN` 再重试。

### Q4：`--repo-id` 和 `REPO_ID` 谁的优先级更高？

命令行参数优先级更高；只有当未传 `--repo-id` 时才会读取 `REPO_ID`。

### Q5：上传会覆盖远端已有文件吗？

`upload_folder` 的具体行为由 ModelScope SDK 与仓库状态决定，通常表现为一次提交更新：同名文件会更新为本地版本。上传前建议确认目标 repo_id 与目录内容，避免误传。