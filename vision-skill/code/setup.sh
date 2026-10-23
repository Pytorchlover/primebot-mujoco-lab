#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv_bin="$(command -v uv || true)"
if [ -z "$uv_bin" ] && [ -x "$HOME/.local/bin/uv" ]; then uv_bin="$HOME/.local/bin/uv"; fi
if [ ! -d .venv ]; then
  if [ -n "$uv_bin" ]; then "$uv_bin" venv --python 3.11 .venv
  else python3 -m venv .venv; fi
fi
if [ -n "$uv_bin" ]; then
  "$uv_bin" pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  "$uv_bin" pip install --python .venv/bin/python -r code/requirements.txt
else
  .venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/python -m pip install -r code/requirements.txt
fi
if [ ! -d upstream/.git ]; then git clone https://atomgit.com/primebot/carry-clothes.git upstream; fi
git -C upstream checkout --detach 301b02ab86efa62cba1d4fdf70268715af9625d9
mkdir -p model results
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('google/owlvit-base-patch32', revision='cbc355fb364588351c5d51c7f74465e8e7ec6f72', local_dir='model', allow_patterns=['*.json','*.txt','*.safetensors'])
PY
.venv/bin/python code/prepare_assets.py
.venv/bin/python code/patch_upstream.py
