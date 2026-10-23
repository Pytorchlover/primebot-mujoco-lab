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
  "$uv_bin" pip install --python .venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  "$uv_bin" pip install --python .venv/bin/python -r code/requirements.txt
else
  .venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/python -m pip install -r code/requirements.txt
fi
if [ ! -d upstream/.git ]; then git clone https://github.com/huang-kai/primebuddy.git upstream; fi
git -C upstream checkout --detach 8731c2fcbe4aed7183882cab14f403df96bab62e
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B',revision='1cfa9a7208912126459214e8b04321603b3df60c',local_dir='model',allow_patterns=['*.json','*.txt','*.safetensors'])
PY
