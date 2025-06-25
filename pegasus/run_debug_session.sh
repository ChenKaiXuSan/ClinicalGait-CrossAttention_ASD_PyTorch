#!/bin/bash

# === 1. 参数设定 ===
PROJECT_NAME="SKIING"                 # 你的项目名（必须改）
ENV_PATH="/home/SKIING/chenkaixu/.venv/med_atn"  # venv 虚拟环境路径（或用 conda 环境名）
TIME="01:00:00"                        # 申请时长（最大 01:00:00）

# === 2. 启动 debug 节点会话 ===
echo "🟡 请求 debug 节点会话: ${TIME}"
qlogin -A "$PROJECT_NAME" -q debug -l elapstim_req=$TIME

