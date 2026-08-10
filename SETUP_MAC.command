#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "[1/4] 프로젝트 전용 파이썬을 준비합니다."
python3 -m venv .venv

echo "[2/4] 필요한 프로그램을 설치합니다."
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements.txt

echo "[3/4] pandas를 확인합니다."
".venv/bin/python" -c "import sys, pandas; print('Python:', sys.executable); print('pandas:', pandas.__version__)"

echo "[4/4] 데이터를 검사합니다."
".venv/bin/python" scripts/validate_data.py

echo "설치와 검사가 모두 성공했습니다."
read -r -p "Enter를 누르면 창을 닫습니다."
