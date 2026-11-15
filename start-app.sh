#!/usr/bin/env bash
#
# start-app.sh — 데이터 분석 Lab (Streamlit) 실행 스크립트
#   사용법:  ./start-app.sh            # 기본 포트(8501)로 실행
#            ./start-app.sh 8600       # 포트 지정
#
set -euo pipefail

# 스크립트가 있는 디렉토리로 이동 (어디서 실행해도 동작)
cd "$(dirname "$0")"

APP="eda_lab_app.py"
PORT="${1:-8501}"

# 1) pipenv 설치 확인
if ! command -v pipenv >/dev/null 2>&1; then
  echo "[오류] pipenv 가 설치되어 있지 않습니다."
  echo "      설치:  pip install pipenv   또는   brew install pipenv"
  exit 1
fi

# 2) 가상환경 / 의존성 확인 (없으면 설치)
if ! pipenv --venv >/dev/null 2>&1; then
  echo "[설정] 가상환경이 없어 의존성을 설치합니다 (pipenv install)..."
  pipenv install
fi

# 3) Streamlit 설치 확인 (누락 시 안내)
if ! pipenv run python -c "import streamlit" >/dev/null 2>&1; then
  echo "[설정] streamlit 미설치 — pipenv install 로 의존성을 동기화합니다..."
  pipenv install
fi

# 4) 앱 파일 존재 확인
if [ ! -f "$APP" ]; then
  echo "[오류] $APP 파일을 찾을 수 없습니다. (현재 위치: $(pwd))"
  exit 1
fi

echo "▶ 데이터 분석 Lab 실행 — http://localhost:${PORT}"
echo "  (종료: Ctrl+C)"
exec pipenv run streamlit run "$APP" --server.port "$PORT"
