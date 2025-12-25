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
- Model source: ModelScope `yukinostuki/qwen3-4b-ft-v1` (merged weights). Downloaded during build via `download_model.py` into `/app/model/...`.
- Serving entrypoint: `serve.py` (FastAPI+uvicorn). Prefer vLLM when available; transformers is a fallback.
- Local eval helper: `eval_local.py` calls `/predict` and computes RougeL-F1 in the same style as the judge.

## Work style

- Keep changes minimal and measurable; prioritize speed optimizations that do not degrade accuracy.
- Do not add heavy CI; this repo’s primary automation is GitHub→Gitee sync.
