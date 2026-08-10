#!/usr/bin/env bash
set -e
python3 -m pip install -r requirements.txt
python3 scripts/download_smp.py
python3 scripts/prepare_data.py
python3 scripts/validate_data.py
echo "완료: data/processed/train.csv를 AI 담당자에게 전달하세요."
