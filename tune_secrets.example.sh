#!/usr/bin/env bash
# 仅作示例：复制为 tune_secrets.sh（已被 .gitignore 忽略）后再填写。
# 用法：
#   cp tune_secrets.example.sh tune_secrets.sh
#   vim tune_secrets.sh
#   chmod 600 tune_secrets.sh
#
# auto_tune.sh 会在启动时自动 source ./tune_secrets.sh（若存在）。

# ===== 飞书 Webhook（推荐） =====
# 你的机器人 Webhook URL
export TUNE_WEBHOOK_KIND="feishu"
export TUNE_WEBHOOK_URL=""

# 可选：飞书签名密钥（若开启签名校验才需要）
# 若不确定，建议留空，避免 19021（签名校验失败）。
export TUNE_FEISHU_SECRET=""

# ===== 邮件通知（兜底） =====
# 163 邮箱示例（SMTPS 465）。如不用邮件，可全部留空。
export TUNE_SMTP_HOST="smtp.163.com"
export TUNE_SMTP_PORT="465"
export TUNE_SMTP_SSL="1"
# STARTTLS：587 常用；与 SSL 二选一
export TUNE_SMTP_NO_STARTTLS="1"

export TUNE_SMTP_USER=""
export TUNE_SMTP_PASS=""  # 163 通常需要授权码，不是登录密码

# 收件人（逗号分隔也支持）
export TUNE_SMTP_TO=""
# 发件人显示名/邮箱
export TUNE_SMTP_FROM=""

# 触发哪些邮件（可选）：best,crashed,done,abnormal
export TUNE_EMAIL_KINDS="best,crashed,done"

# ===== 可选：加速重复运行（按需打开） =====
# export SKIP_PIP_INSTALL=1
# export SKIP_MODEL_DOWNLOAD=1
