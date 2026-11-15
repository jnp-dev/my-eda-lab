@echo off
chcp 65001 >nul
REM ============================================================
REM  start-app.bat — 데이터 분석 Lab (Streamlit) 실행 스크립트
REM    사용법:  start-app.bat            # 기본 포트(8501)로 실행
REM             start-app.bat 8600       # 포트 지정
REM ============================================================
setlocal

REM 스크립트가 있는 디렉토리로 이동 (어디서 실행해도 동작)
cd /d "%~dp0"

set "APP=eda_lab_app.py"
if "%~1"=="" (set "PORT=8501") else (set "PORT=%~1")

REM 1) python 설치 확인
where python >nul 2>&1
if errorlevel 1 (
    echo [오류] python 이 설치되어 있지 않습니다.
    echo        설치:  https://www.python.org/downloads/
    exit /b 1
)

REM 2) Streamlit 설치 확인 (누락 시 pip 로 설치)
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [설정] streamlit 미설치 — pip 로 의존성을 설치합니다...
    if exist "requirements.txt" (
        python -m pip install -r requirements.txt
    ) else (
        python -m pip install streamlit
    )
    if errorlevel 1 exit /b 1
)

REM 3) 앱 파일 존재 확인
if not exist "%APP%" (
    echo [오류] %APP% 파일을 찾을 수 없습니다. ^(현재 위치: %CD%^)
    exit /b 1
)

echo ▶ 데이터 분석 Lab 실행 — http://localhost:%PORT%
echo   ^(종료: Ctrl+C^)
python -m streamlit run "%APP%" --server.port "%PORT%"

endlocal
