#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv_bin="$(command -v uv || true)"
if [ -z "$uv_bin" ] && [ -x "$HOME/.local/bin/uv" ]; then uv_bin="$HOME/.local/bin/uv"; fi
if [ -n "$uv_bin" ]; then
 "$uv_bin" pip install --python .venv/bin/python -r code/requirements-sim.txt
else
 .venv/bin/python -m pip install -r code/requirements-sim.txt
fi
mkdir -p sources
if [ ! -d sources/q1-urdf/.git ]; then git clone https://atomgit.com/primebot/q1-urdf.git sources/q1-urdf; fi
git -C sources/q1-urdf checkout --detach 562b764a18613d459140ed9b255fa30434e346ef
