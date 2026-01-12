---
title: 启动脚本（start_webui.sh）
sidebar_position: 31
---

本页面对应脚本 `start_webui.sh`：一个面向本地/云主机的 WebUI 启动器，用来快速拉起 `webui.py`（Gradio），并在需要时自动创建虚拟环境与安装依赖。

文档结构：

1. 用法 / 作用 / 输入输出 / 副作用
2. 参数与环境变量详解
3. 代码详解
4. 常见问题（FAQ）

---

## 1) 用法 / 作用 / 输入输出 / 副作用

### 作用与定位

- 目标：让你在一台机器上更省事地启动 WebUI。
- 它不启动后端：后端仍然需要你自己用 `./run_model.sh`（或其他方式）启动。
- 它不参与评测：评测只关心 `serve.py` 的 `/` 与 `/predict`，WebUI 相关脚本不在评测链路中。

### 用法

最常用（默认创建/使用 `./.venv`）：

```bash
./start_webui.sh
```

指定后端地址：

```bash
API_BASE_URL=http://127.0.0.1:8000 ./start_webui.sh
```

指定虚拟环境路径与 Python 解释器：

```bash
VENV_PATH=./.venv-webui PYTHON_BIN=python3.10 ./start_webui.sh
```

### 输入与输出

- 输入：主要通过环境变量（见第 2 节）。
- 输出：
  - 标准输出打印当前配置、依赖安装进度、后端探活结果。
  - 成功启动后，会运行 `python webui.py` 并进入阻塞态直到你停止 WebUI。

### 副作用

- 若 `VENV_PATH` 不存在，会创建一个虚拟环境目录。
- 会在虚拟环境内执行 `pip install`（默认每次都 `--upgrade`）。
- 会用 `curl` 对 `${API_BASE_URL}/` 做一次快速探活（仅提示，不会阻止继续启动）。

---

## 2) 参数与环境变量详解

脚本支持的环境变量（均为“有就用，没有才用默认”，即 `VAR=${VAR:-default}`）：

- `VENV_PATH`
  - 默认：`./.venv`
  - 作用：虚拟环境目录。
- `PYTHON_BIN`
  - 默认：`python3`
  - 作用：仅用于“创建虚拟环境”步骤：`${PYTHON_BIN} -m venv ${VENV_PATH}`。
- `API_BASE_URL`
  - 默认：`http://127.0.0.1:8000`
  - 作用：传给 `webui.py` 的后端地址；同时也用于 `curl ${API_BASE_URL}/` 探活。
- `WEBUI_HOST`
  - 默认：`0.0.0.0`
  - 作用：传给 `webui.py`，WebUI 监听地址。
- `WEBUI_PORT`
  - 默认：`7860`
  - 作用：传给 `webui.py`，WebUI 监听端口。
- `WEBUI_SHARE`
  - 默认：`0`
  - 作用：传给 `webui.py`，控制 Gradio 的 `share` 开关（是否创建分享链接）。

依赖安装相关（写死在脚本里，不通过变量控制）：

- 会升级：`pip setuptools wheel`
- 会安装/升级：`requirements-webui.txt` 与 `requirements-eval.txt`

---

## 3) 代码详解

按 `start_webui.sh` 的真实执行顺序说明：

1. 严格模式
   - `set -euo pipefail`
   - 含义：任意命令失败会退出；未定义变量视为错误；管道中任意一段失败都会退出。

2. 读取配置（环境变量默认值）
   - `VENV_PATH/API_BASE_URL/WEBUI_PORT/WEBUI_HOST/WEBUI_SHARE/PYTHON_BIN` 都是“未设置则使用默认”。

3. 创建虚拟环境（若不存在）
   - 若 `${VENV_PATH}` 目录不存在：
     - 先检查 `${PYTHON_BIN}` 是否可执行（`command -v`）。
     - 然后执行：`${PYTHON_BIN} -m venv ${VENV_PATH}`。

4. 激活虚拟环境
   - `source "${VENV_PATH}/bin/activate"`
   - 后续的 `python`/`pip` 将指向该 venv。

5. 安装/更新依赖
   - 先升级基础构建工具：`python -m pip install --upgrade pip setuptools wheel`
   - 再安装两份 requirements：
     - `python -m pip install -r requirements-webui.txt -r requirements-eval.txt --upgrade`
   - 备注：该步骤会访问网络下载依赖，适合本地/云主机；评测机 run 阶段通常断网，不应依赖此脚本。

6. 探活后端（仅提示）
   - `curl -s --max-time 5 "${API_BASE_URL}/"`：
     - 成功：提示“后端已就绪”。
     - 失败：打印提示并建议另开终端运行 `./run_model.sh`，但仍继续启动 WebUI。

7. 启动 WebUI
   - `export API_BASE_URL WEBUI_PORT WEBUI_HOST WEBUI_SHARE`
   - 执行：`python webui.py`

---

## 4) 常见问题（FAQ）

### Q1：我只启动了 start_webui.sh，为什么无法对话？

因为它不会启动后端 `serve.py`。你需要另开一个终端启动后端，例如：

```bash
./run_model.sh
```

然后确保 `API_BASE_URL` 指向可访问的后端地址。

### Q2：脚本每次都在装依赖，太慢怎么办？

这是脚本的当前行为：每次都会 `pip install ... --upgrade`。

可选做法：

- 手动维护虚拟环境依赖：你可以先执行一次 `./start_webui.sh` 完成安装，之后直接 `source ./.venv/bin/activate && python webui.py`。
- 或改造脚本：加一个 `SKIP_WEBUI_PIP_INSTALL=1` 之类的开关（如你需要我也可以直接帮你改脚本并同步文档）。

### Q3：报错 “找不到 Python：xxx” 怎么办？

脚本只在创建 venv 时使用 `PYTHON_BIN`。请确认机器上该解释器存在：

- `PYTHON_BIN=python3.10 ./start_webui.sh`

或安装对应版本的 Python。

### Q4：端口冲突怎么办？

修改 WebUI 端口即可：

```bash
WEBUI_PORT=7861 ./start_webui.sh
```

后端端口（默认 8000）由 `serve.py` 决定，和 WebUI 端口独立。

### Q5：断网环境能用这个脚本吗？

如果虚拟环境里已经装好了依赖，且你关闭了 WebUI 的联网 RAG 功能，那么启动 WebUI 本身可以不依赖网络。

但 `start_webui.sh` 默认会执行 `pip install --upgrade`，在断网环境通常会失败；这种情况下建议直接激活已有 venv 并运行 `python webui.py`。