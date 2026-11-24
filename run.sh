#!/usr/bin/env bash
set -euo pipefail

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python rk_gnn.py
