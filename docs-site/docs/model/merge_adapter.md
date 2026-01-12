---
title: 融合 LoRA Adapter（merge_adapter.py）
sidebar_position: 70
---

# 融合 LoRA Adapter 脚本详解

## 1) 脚本功能概述

### 1.1 作用与定位

`merge_adapter.py` 用于把一个 LoRA/PEFT adapter 仓库合并进 base model，导出一个可直接推理的“完整模型目录”。主要面向以下场景：

- 存在 base model（在 ModelScope 上），也存在 adapter（在 Git 仓库中）。
- 需要得到一个无需依赖 adapter 的“已融合权重”，便于部署、量化或离线运行。

### 1.2 它能完成什么

按流程它会完成：

1. 从 ModelScope 下载 base model（`snapshot_download`）。
2. `git clone` adapter 仓库到工作目录；可选 checkout 一个 ref（branch/tag/commit）。
3. 在 adapter 仓库内寻找常见的 PEFT 权重文件，并处理 Git LFS pointer（如需）。
4. 确保 `adapter_config.json` 存在（必要时由环境变量提供）。
5. 使用 `peft` 把 adapter merge 到 base model，导出到 `output_dir`。

### 1.3 输入与输出

**输入：**

- base model：`--base_model` / `--base_revision`
- adapter 仓库：`--adapter_repo_url` / `--adapter_ref`
- 路径：`--cache_dir`（模型下载缓存）、`--work_dir`（克隆工作区）、`--output_dir`（导出目录）
- （必要时）adapter_config：`ADAPTER_CONFIG_JSON` 或 `ADAPTER_CONFIG_PATH`

**输出：**

- 一个“完整模型目录”（`--output_dir`），包含：
  - `tokenizer` 文件（`tokenizer.save_pretrained`）
  - 融合后的模型权重（`merged_model.save_pretrained(..., safe_serialization=True)`）
- 控制台日志（每个阶段都有 `[merge_adapter] ...` 前缀）
- 退出码：成功为 `0`；任何未捕获异常会打印 `FAILED` 并以 `1` 退出。

**会改变什么：**

- 若 `work_dir` 或 `output_dir` 已存在，脚本会**删除并重建**（`ensure_clean_dir` 使用 `shutil.rmtree`）。
- 会调用外部命令 `git`（以及可能的 `git lfs`），因此需要相应工具可用。

**不改变：**

- 不修改远端 base model 或 adapter 仓库本身；所有变更均发生在本地工作目录与导出目录。

:::caution
`--output_dir` 目录会被清空重建；不要指向包含重要文件的路径。
:::

---

## 2) 参数与环境变量详解

### 2.1 命令行参数

#### `--base_model`

- **作用**：ModelScope base model id。
- **默认值**：环境变量 `BASE_MODEL`，若未设置则 `Qwen/Qwen3-4B`。

#### `--base_revision`

- **作用**：base model revision（分支/tag/commit）。
- **默认值**：环境变量 `BASE_REVISION`，否则 `master`。

#### `--cache_dir`

- **作用**：ModelScope 下载缓存目录。
- **默认值**：环境变量 `MODEL_CACHE_DIR`，否则为当前工作目录下的 `./model`。

#### `--adapter_repo_url`

- **作用**：adapter 仓库的 git 地址。
- **默认值**：环境变量 `ADAPTER_REPO_URL`，否则 `git@gitee.com:yukinostuki/qwen3-4b-plus.git`。
- **注意**：默认是 SSH URL；在 Docker build 环境中若没有配置 SSH key，会克隆失败。可改用 HTTPS URL 或在构建环境注入 SSH 凭据。

#### `--adapter_ref`

- **作用**：可选的 git ref（branch/tag/commit）。
- **默认值**：环境变量 `ADAPTER_REPO_REF`，否则空字符串（不 checkout）。
- **行为**：脚本会在浅克隆后执行：
  - `git fetch --depth 1 origin <ref>`
  - `git checkout FETCH_HEAD`

#### `--work_dir`

- **作用**：工作目录，用于克隆 adapter。
- **默认值**：环境变量 `MERGE_WORK_DIR`，否则为当前工作目录下的 `./merge_work`。
- **内部结构**：adapter 会被克隆到 `<work_dir>/adapter_repo`。

#### `--output_dir`

- **作用**：导出融合后完整模型的目录。
- **默认值**：环境变量 `MERGED_MODEL_DIR`，否则为当前工作目录下的 `./merged`。
- **行为**：若存在会被删除并重建。

### 2.2 环境变量

- `BASE_MODEL`
  - 对应参数：`--base_model`
  - 作用：base model id。
- `BASE_REVISION`
  - 对应参数：`--base_revision`
  - 作用：base revision。
- `MODEL_CACHE_DIR`
  - 对应参数：`--cache_dir`
  - 作用：ModelScope 下载缓存目录。
- `ADAPTER_REPO_URL`
  - 对应参数：`--adapter_repo_url`
  - 作用：adapter 仓库地址。
- `ADAPTER_REPO_REF`
  - 对应参数：`--adapter_ref`
  - 作用：adapter 仓库 ref。
- `MERGE_WORK_DIR`
  - 对应参数：`--work_dir`
  - 作用：克隆工作目录。
- `MERGED_MODEL_DIR`
  - 对应参数：`--output_dir`
  - 作用：融合导出目录。
- `ADAPTER_CONFIG_JSON`
  - 作用：adapter_config.json 的原始 JSON 字符串；仅在仓库缺失 config 时使用。
- `ADAPTER_CONFIG_PATH`
  - 作用：adapter_config.json 的本地文件路径；仅在仓库缺失 config 时使用。

---

## 3) 代码实现详解

### 3.1 外部依赖与运行环境

脚本会用到：

- ModelScope：`modelscope.snapshot_download`
- Git：`git clone / fetch / checkout`
- Git LFS（可选）：当权重文件是 LFS pointer 时会执行 `git lfs install --local` 与 `git lfs pull`
- Python 依赖：`torch`、`transformers`、`peft`

:::note
融合过程通常比较吃内存与时间，建议在单独的机器/构建阶段完成，不建议放到评测机 run 阶段临时执行。
:::

### 3.2 关键函数与职责

#### `run(cmd, cwd=None)`

- 包装 `subprocess.run(check=True)`。
- 任何命令失败会直接抛异常，最终被最外层捕获并以退出码 `1` 结束。

#### `ensure_clean_dir(path)`

- 若目录已存在：`shutil.rmtree(path)` 清空。
- 然后 `os.makedirs(path, exist_ok=True)` 重新创建。

#### `clone_adapter_repo(repo_url, dest_dir, ref)`

- 先清空并创建目标目录。
- `git clone --depth 1 <repo_url> <dest_dir>` 进行浅克隆。
- 若 `ref` 非空：
  - `git fetch --depth 1 origin <ref>`
  - `git checkout FETCH_HEAD`

#### `find_adapter_weights(adapter_dir)`

在 adapter 目录下按顺序寻找以下文件之一：

1. `adapter_model.safetensors`
2. `adapter_model.bin`
3. `pytorch_model.bin`

若都不存在则报错。

#### `is_git_lfs_pointer(file_path)` 与 `materialize_lfs_file(file_path, repo_dir)`

- `is_git_lfs_pointer` 通过读取文件头判断是否为 LFS pointer。
- 若是 pointer：
  - 尝试 `git lfs install --local` 与 `git lfs pull` 拉取真实大文件
  - 拉取后仍是 pointer 会报错，提示检查 LFS 可用性

#### `resolve_adapter_config(adapter_dir)`

确保 `adapter_config.json` 存在：

- 若仓库自带 `adapter_config.json`：直接使用。
- 否则按优先级尝试：
  1. `ADAPTER_CONFIG_JSON`：解析为 JSON 后写入 `adapter_config.json`
  2. `ADAPTER_CONFIG_PATH`：拷贝该文件到 `adapter_config.json`
- 两者都未提供则直接报错。

### 3.3 `main()` 主流程说明

主流程按顺序执行：

1. 解析参数与默认值（默认路径基于当前工作目录，避免依赖固定的 `/app`）。
2. 下载 base model：
   - `snapshot_download(args.base_model, cache_dir=args.cache_dir, revision=args.base_revision)`
3. 克隆 adapter repo 到 `<work_dir>/adapter_repo`。
4. 找到 adapter 权重文件，并在必要时拉取 LFS 大文件。
5. 解决 `adapter_config.json` 的存在性。
6. 加载并融合：
   - `AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True, use_fast=False)`
   - `AutoModelForCausalLM.from_pretrained(..., torch_dtype=float16)` 优先 fp16；失败时回退 fp32
   - `PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)`
   - `merge_and_unload()` 得到融合后的模型
7. 清空并写入 `output_dir`：保存 tokenizer 与融合后的 safetensors 权重。

---

## 4) 常见问题

### Q1：运行时报 git clone 失败（权限/认证错误）怎么办？

`--adapter_repo_url` 默认是 SSH 地址（如 `git@gitee.com:...`），在没有 SSH key 的环境（尤其是 Docker build）通常会失败。

解决方式：

- 改用 HTTPS 仓库地址（如果仓库允许）。
- 或在构建环境中配置 SSH key 与 known_hosts。

### Q2：提示找不到 adapter 权重文件怎么办？

脚本只会在仓库根目录寻找三种常见文件名：

- `adapter_model.safetensors`
- `adapter_model.bin`
- `pytorch_model.bin`

请确认 adapter 仓库里至少包含其中之一，且位于仓库根目录（或按需调整仓库结构）。

### Q3：提示缺少 adapter_config.json 怎么办？

这是 PEFT 融合所必需的文件。可二选一提供：

- `ADAPTER_CONFIG_JSON`：把 JSON 原文放到环境变量
- `ADAPTER_CONFIG_PATH`：指向一个本地 `adapter_config.json`

脚本会把它写入/拷贝到 adapter 仓库目录下再继续融合。

### Q4：Git LFS pointer 是什么？为什么会报错？

有些仓库把大权重文件交给 Git LFS 管理，git clone 得到的只是一个很小的“指针文件”。脚本会尝试自动 `git lfs pull`；如果环境缺少 git-lfs 或无法下载 LFS 对象，就会失败并提示安装/配置 git-lfs。

### Q5：为什么会先尝试 fp16 再回退 fp32？

脚本优先 fp16 是为了降低内存占用；但在某些 CPU 环境或特定权重下 fp16 加载可能失败，因此会打印 `fp16 load failed, fallback to fp32` 并使用 fp32 继续。

### Q6：融合输出目录里会有哪些文件？

`output_dir` 会包含 tokenizer 与融合后的模型权重（以 safetensors 形式保存），并可被 `transformers` 直接 `from_pretrained(output_dir)` 加载。
