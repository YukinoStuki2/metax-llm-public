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
export FEISHU_WEBHOOK_URL=""

# 可选：飞书签名密钥（若开启签名校验才需要）
# 若不确定，建议留空，避免 19021（签名校验失败）。
export FEISHU_SIGNING_SECRET=""

# ===== 邮件通知（兜底） =====
# 163 邮箱示例（SMTPS 465）。如不用邮件，可全部留空。
export SMTP_HOST="smtp.163.com"
export SMTP_PORT="465"
export SMTP_USE_SSL="1"
export SMTP_USE_STARTTLS="0"

export SMTP_USERNAME=""
export SMTP_PASSWORD=""  # 163 通常需要授权码，不是登录密码

# 收件人（逗号分隔也支持）
export EMAIL_TO=""
# 发件人显示名/邮箱
export EMAIL_FROM=""

# 触发哪些邮件（可选）：error,finish,progress
export EMAIL_KINDS="error,finish"
