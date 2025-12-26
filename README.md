## 融合 Adapter（LoRA/PEFT）到基座模型

如果你的微调仓库里只有 `adapter_model.safetensors`（没有完整的 merged 权重），可以在构建阶段下载基座模型并进行融合。

本项目默认在 Docker 构建阶段执行 [merge_adapter.py](merge_adapter.py)：

- 从 ModelScope 下载基座模型（默认 `Qwen/Qwen3-4B`）
- 从 Gitee clone 你的 adapter 仓库（默认 `https://gitee.com/yukinostuki/qwen3-4b-plus.git`）
- 使用 PEFT 将 adapter 融合到基座模型并导出到 `/app/model/merged`
- 运行时 `MODEL_DIR=/app/model/merged`

### 环境变量

- `ADAPTER_REPO_URL`：adapter 仓库地址（可用 https 或 ssh 地址）
- `ADAPTER_REPO_REF`：可选，指定分支/Tag/Commit
- `BASE_MODEL`：ModelScope 基座模型 ID，默认 `Qwen/Qwen3-4B`
- `BASE_REVISION`：基座模型 revision，默认 `master`
- `MERGED_MODEL_DIR`：融合输出目录，默认 `/app/model/merged`

### adapter_config.json 的要求

PEFT 融合需要 `adapter_config.json`。

如果你的 adapter 仓库里没有该文件（只提供了 `adapter_model.safetensors`），请额外提供其配置：

- `ADAPTER_CONFIG_JSON`：直接传 JSON 字符串
- 或 `ADAPTER_CONFIG_PATH`：指向一个 json 文件路径（构建阶段可用 COPY 注入）

否则融合脚本会报错并提示你补齐配置。

### WSL/本地运行（创建 Python 虚拟环境）

在 WSL（Ubuntu 24.04 等）里，如果你发现 `python3 -m venv` 报 `ensurepip is not available`，说明系统没装 venv/pip 组件。

1) 安装系统依赖（需要输入 sudo 密码）：

```bash
sudo apt update
sudo apt install -y python3.12-venv python3-pip
```

2) 创建并激活虚拟环境：

```bash
cd /path/to/metax-demo-mirror
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

3) 安装 merge 所需 Python 依赖（推荐用最小集合）：

```bash
pip install -r requirements-merge.txt
```

4) 安装 PyTorch

- CPU-only（WSL/无 GPU 最稳）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

5) 运行融合：

```bash
python merge_adapter.py \
  --base_model Qwen/Qwen3-4B \
  --adapter_repo_url https://gitee.com/yukinostuki/qwen3-4b-plus.git \
  --output_dir ./merged
```

如果你的 adapter 仓库里缺 `adapter_config.json`，按上文用 `ADAPTER_CONFIG_JSON` 或 `ADAPTER_CONFIG_PATH` 提供即可。

# 大模型推理服务模板(MetaX 沐曦)

本项目是一个极简的大模型推理服务模板，旨在帮助您快速构建一个可以通过API调用的推理服务器。

## 默认模型（ModelScope）

当前仓库默认直接从 ModelScope 下载已融合的模型权重：`YukinoStuki/Qwen3-4B-Plus-Merged`。

- 构建阶段由 `download_model.py` 下载到 `./model/$MODEL_ID`
- 运行阶段默认从 `MODEL_DIR=./model/$MODEL_ID` 加载（见 `Dockerfile` / `serve.py`）

## 项目结构

- `Dockerfile`: 用于构建容器镜像的配置文件，MetaX提供docker构建流程，该文件中
    - `FROM`是通过拉取服务器中已存在的docker，因此不要进行改动
    - `EXPOSE`的端口是作为评测的主要接口，因此不要随意变动
- `serve.py`: 推理服务的核心代码，您需要在此文件中修改和优化您的模型加载与推理逻辑
    - `model_local_dict`: 是将模型映射到本地模型的dict
    - Notes: 这个程序不能访问Internet。
- `requirements.txt`: Python依赖列表。您可以添加需要的库。
- `.gitignore`: Git版本控制忽略的文件列表。
- `download_model.py`: 下载权重的脚本，不建议进行修改，模型参数、下载位置和版本都可以在`Dockerfile`中进行调整
    - `--model_name`: 模型名称，该名称是在modelscope中可以被检索的，例如：需要下载`DeepSeek-V3.2`,在modelscope中可知，`model_name`为`deepseek-ai/DeepSeek-V3.2`, 那么配置为`deepseek-ai/DeepSeek-V3.2`即可从modelscope中下载模型，如果您需要下载自己的微调模型，可以在modelscope中上传自己的模型，并调整该参数即可使用；
    - `--cache_dir`: 模型缓存地址，该地址是model存储的位置，例如指定下载`DeepSeek-V3.2`，在`/app`路径中，那么模型存放的位置在`/app/deepseek-ai/DeepSeek-V3.2`中；
    - `--revision`: 模型参数的Git版本，该版本对应modelscope仓库中的版本，您可根据自己微调数个版本，上传到同一仓库中，拉取时采用不同版本的revision即可；
    - Notes: 如果您的模型为非公开，请打开`download_model.py`进行相应的配置，本模板已将该部分注释([代码](download_model.py#L16-L17))，对注释内容取消注释并注入相应的内容即可配置非公开模型。在使用非公开模型时，建议在非judge环境中进行download_model的环境验证，以免浪费judge次数。
- `README.md`: 本说明文档

## 如何修改

您需要关注的核心文件是 `serve.py`.

您可以完全替换`serve.py`的内容，只要保证容器运行后，能提供模板中的'/predict'和'/'等端点即可。

## 评测系统的规则

评测系统会向 /predict 端点发送 POST 请求，其JSON body格式为: 
```json
{
  "prompt": "Your question here"
}
```
您的服务必须能够正确处理此请求，并返回一个JSON格式的响应，格式为: 
```json
{
  "response": "Your model's answer here"
}
```
请务必保持此API契约不变！

## 环境说明

### 软件包版本

主要软件包(vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64)版本如下：

|软件|版本|
|:--:|:--:|
|python|3.10|
|ubuntu|22.04|
|pytorch|2.6|
|vLLM|0.10.0|

`软件使用的Note`:
- 如果您需要其他的镜像，请您先查询[沐曦开发者社区](https://developer.metax-tech.com/softnova/docker)，查找您需要的docker镜像，后联系龚昊助教添加相应的镜像。
- 建议您先在`OpenHydra`中使用添加的软件，避免软件兼容性带来的问题（非GPU相关的软件都可以兼容，GPU相关软件或依赖GPU相关软件的软件建议验证后使用）。
- `OpenHydra`的访问地址请查询`沐曦GPU实验平台操作手册`，欢迎您的使用。

### judge平台的配置说明

judge机器的配置如下：

``` text
os: ubuntu24.04
cpu: 24核
内存: 200GB
磁盘: 1T
GPU: MXC500(显存：64GB)
网络带宽：100Mbps
```

judge系统的配置如下：

``` text
docker build stage: 900s
docker run - health check stage: 180s
docker run - predict stage: 360s
```
