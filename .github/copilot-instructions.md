# Copilot Instructions

- Repository role: GitHub is the source of truth; changes push automatically to upstream Gitee repo `yukinostuki/metax-demo`.
- Primary automation lives in [workflows/sync_to_gitee.yml](./workflows/sync_to_gitee.yml); triggers on every push to `master` branch.
- Sync process: checks out code with full history, sets up SSH with `GITEE_SSH_PRIVATE_KEY`, force-pushes to Gitee `master:master`. Assumes Gitee accepts force pushes.
- Make edits directly in GitHub; the workflow propagates changes to Gitee automatically. Avoid manual pushes to Gitee to prevent conflicts.
- Secrets required: `GITEE_SSH_PRIVATE_KEY` (ED25519 key with write access to Gitee repo).
- CI expectations: No build/test workflows; only sync-on-push. Avoid adding heavy CI unless needed for validation before sync.
- Default branch is `master`; all development happens here. Push to `master` triggers immediate sync to Gitee.
- When debugging sync issues, inspect Actions logs for the `Sync to Gitee` workflow and validate SSH known_hosts entry for `gitee.com`.
- Keep commits atomic and meaningful; every push to `master` will sync to Gitee.
- Any additional project documentation lives in this repo. README and project files in [root](../).
- File/dir map: root contains project code; [workflows](./workflows) contains GitHub→Gitee automation; no source code stored in `.github`.
- Branch policy: development on `master`; feature branches optional but must merge to `master` for sync. Gitee will mirror `master` only.
- Releases/tags: create in GitHub; manually push tags to Gitee or extend workflow to include `git push --tags gitee`.
- Common failure: missing or invalid `GITEE_SSH_PRIVATE_KEY` yields auth errors during push. Rotate the key in repo secrets and re-run workflow.
- Local testing of the sync job: replicate steps in [workflows/sync_to_gitee.yml](./workflows/sync_to_gitee.yml) with your own tokens; ensure SSH auth and `git push` succeed.
- Ownership: GitHub is primary; Gitee is downstream mirror receiving force-pushes from GitHub Actions.
