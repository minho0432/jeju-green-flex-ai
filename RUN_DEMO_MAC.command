#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "먼저 터미널에서 python3 -m venv .venv 를 실행하세요."
  read -r
  exit 1
fi

".venv/bin/python" -m pip install -r requirements.txt
".venv/bin/python" scripts/validate_data.py
".venv/bin/python" scripts/train_models.py
".venv/bin/python" -m streamlit run app.py
