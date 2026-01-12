# Copilot Instructions

- Project purpose: fine-tune an open-source LLM for QA evaluation; build a Q/A dataset (from a book), serve inference via HTTP, and optimize accuracy + throughput for the judge.

## Source of truth / sync

- Repository role: GitHub is the source of truth; pushes sync to Gitee repo `yukinostuki/metax-demo`.
- Automation: [workflows/sync_to_gitee.yml](./workflows/sync_to_gitee.yml) runs on every push to `master`.
- Sync method: sets up SSH using `GITEE_SSH_PRIVATE_KEY`, then force-pushes `master:master` to Gitee.
- Workflow debugging: check GitHub Actions logs for `Sync to Gitee`; most common failures are missing/invalid `GITEE_SSH_PRIVATE_KEY` or SSH known_hosts issues.

## Judge API contract (MUST NOT BREAK)

- HTTP endpoints:
	- `GET /` health check. Must return quickly; judge calls this before `/predict`.
	- `POST /predict` with JSON `{"prompt":"..."}`; return JSON `{"response":"..."}`.
- Port: keep Dockerfile `EXPOSE 8000` and service on port 8000.
- Output discipline: avoid very long answers; do not output chain-of-thought; strip `<think>...</think>` if present.

## Environment constraints

- Build stage: Internet allowed (download deps/weights).
- Run stage: NO Internet. Do not add network calls in request path.
- Time limits (template defaults): build 900s, health 180s, predict 360s.

## Scoring + targets

- Accuracy: RougeL-F1 on jieba tokenization; empty prediction or missing responses score 0; fewer predictions are padded with empty strings.
- Throughput on the platform may be estimated using `sum(len(str(response))) / sum(predict_request_time)` (can look inflated); still optimize real latency.
- Current goals: accuracy ≥ 0.35 and throughput preferably > 300 tokens/s.
- Submissions: limited attempts (20 total). Avoid submitting again before the previous run completes.

## Project-specific facts (assume unless user overrides)

- Base image is fixed (do not change `FROM` in Dockerfile).
- Model source: ModelScope `YukinoStuki/Qwen3-4B-Plus-Merged` (merged weights). Downloaded during build via `download_model.py` into `/app/model/...`.
- Serving entrypoint: `serve.py` (FastAPI+uvicorn). Prefer vLLM when available; transformers is a fallback.
- Local eval helper: `eval_local.py` calls `/predict` and computes RougeL-F1 in the same style as the judge.

## Work style

- Default language: respond in Chinese (zh-cn) unless the user explicitly asks for English.
- Keep changes minimal and measurable; prioritize speed optimizations that do not degrade accuracy.
- Do not add heavy CI; this repo’s primary automation is GitHub→Gitee sync.
- `requirements.txt` 只包含 serve.py 与 download_model.py 所需依赖，非必要不要改动或添加评测无关的库。

## 运行参数对齐（重要，后续维护规则）

- 参数来源分工：
	- `Dockerfile`：评测机使用的默认环境变量（必须与调优后的参数一致）。
	- `run_model.sh`：本地/云主机复现评测流程的启动脚本；**不应强制覆盖外部传入的环境变量**，而是“有就用，没有才用默认”。
	- `env_force.sh`：**强制导入**一整套默认参数（用于清理上一次测试遗留变量）；该脚本必须用 `source ./env_force.sh` 执行。

- 修改约束：
	- 每次改环境变量/推理参数（例如 `MODEL_ID`、`MAX_NEW_TOKENS`、`MAX_MODEL_LEN`、vLLM 相关开关等），或修改 `run_model.sh` / `Dockerfile` / `env_force.sh` 任一文件时，必须同步检查并对齐另外两份文件的对应参数，避免“本地好/线上坏”或“脚本不一致”。

- 推荐用法：
	- 想确保本次运行是“干净参数”：`source ./env_force.sh && ./run_model.sh`
	- 想临时覆盖某个参数：先 `source ./env_force.sh`，再在运行前 `export MODEL_ID=...`（或直接 `MODEL_ID=... ./run_model.sh`）。

## Command preflight (MUST DO BEFORE RUNNING COMMANDS)

- `sudo` commands:
	- Always prefer non-interactive checks first (e.g. `sudo -n true`) to detect whether a password prompt will appear.
	- If a password prompt is required, stop and ask the user to input it (do not keep retrying blindly).
	- Avoid running long install steps until sudo readiness is confirmed.

- Python commands:
	- Always run Python via the repo venv interpreter when available: `./.venv/bin/python` (or absolute path).
	- If you plan to use `python`/`python3` directly, first confirm you are in the venv (e.g. `echo $VIRTUAL_ENV`) and that it points to this repo’s `.venv`.
	- Likewise, use `./.venv/bin/pip` for installs to avoid contaminating system Python.

- Node / Docusaurus commands (IMPORTANT for WSL + bundled toolchain):
	- This repo bundles Node.js under `.tools/node/`. When running docs-site commands, ALWAYS use the bundled toolchain instead of any system/Windows Node.
	- Before any `npm`/`npx`/`node` command, export PATH to prefer the bundled binaries:
		- `cd /home/yukinostuki/metax-demo-mirror && export PATH="$PWD/.tools/node/bin:$PATH"`
	- Then run commands from `docs-site/` (example build command):
		- `cd docs-site && npm run build --silent`
	- Do NOT run `npm` without the PATH override; new agent sessions may otherwise resolve to Windows/system Node and fail (e.g. `/usr/bin/env: 'node': No such file or directory`).

- Other commands (general rule):
	- Before executing, verify the required environment exists (e.g. tool installed, file/dir present, correct working directory).
	- Examples: check `.venv/` exists before Python work; check `requirements*.txt` exists before installs; check Docker availability before docker commands; check model directories exist before starting the server.
