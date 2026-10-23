#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p sources
if [ ! -d sources/q1-urdf/.git ]; then git clone https://atomgit.com/primebot/q1-urdf.git sources/q1-urdf; fi
if [ ! -d sources/t1-urdf/.git ]; then git clone https://atomgit.com/primebot/t1-urdf.git sources/t1-urdf; fi
git -C sources/q1-urdf checkout 562b764a18613d459140ed9b255fa30434e346ef
git -C sources/t1-urdf checkout e6482e6b0883e93e378b8b6fb40fca6704050a6f
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
MUJOCO_GL=egl .venv/bin/python code/probe.py
MUJOCO_GL=egl .venv/bin/python code/experiment.py
