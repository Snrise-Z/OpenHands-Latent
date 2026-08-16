#!/usr/bin/env bash

source ~/.bashrc
SWEUTIL_DIR=/swe_util

# FIXME: Cannot read SWE_INSTANCE_ID from the environment variable
# SWE_INSTANCE_ID=django__django-11099
if [ -z "$SWE_INSTANCE_ID" ]; then
    echo "Error: SWE_INSTANCE_ID is not set." >&2
    exit 1
fi

# Read the swe-bench-test-lite.json file and extract the required item based on instance_id
item=$(jq --arg INSTANCE_ID "$SWE_INSTANCE_ID" '.[] | select(.instance_id == $INSTANCE_ID)' $SWEUTIL_DIR/eval_data/instances/swe-bench-instance.json)

if [[ -z "$item" ]]; then
  echo "No item found for the provided instance ID."
  exit 1
fi


WORKSPACE_NAME=$(echo "$item" | jq -r '(.repo | tostring) + "__" + (.version | tostring) | gsub("/"; "__")')

echo "WORKSPACE_NAME: $WORKSPACE_NAME"

# Clear the workspace
if [ -d /workspace ]; then
    rm -rf /workspace/*
else
    mkdir /workspace
fi
# Copy repo to workspace
if [ -d /workspace/$WORKSPACE_NAME ]; then
    rm -rf /workspace/$WORKSPACE_NAME
fi
mkdir -p /workspace
# 用 tar 管道而不是 cp -r:udocker 的 F3(fakechroot)模式下 cp 会建出空目录后
# stat 失败,导致仓库被复制成空壳、随后 git 命令报 "not a git repository"。
mkdir -p /workspace/$WORKSPACE_NAME
(cd /testbed && tar cf - .) | (cd /workspace/$WORKSPACE_NAME && tar xf -)

# Activate instance-specific environment
if [ -d /opt/miniconda3 ]; then
    . /opt/miniconda3/etc/profile.d/conda.sh
    conda activate testbed
fi
