@echo off
chcp 65001 > nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto no_venv

echo [1/4] 데이터를 검사합니다.
".venv\Scripts\python.exe" scripts\validate_data.py
if errorlevel 1 goto error

echo [2/4] AI 모델을 학습하고 5회 검증합니다.
".venv\Scripts\python.exe" scripts\train_models.py
if errorlevel 1 goto error

echo [3/4] AI 결과와 충전계획을 검사합니다.
".venv\Scripts\python.exe" scripts\validate_ai.py
if errorlevel 1 goto error

echo [4/4] 데모 화면을 실행합니다.
".venv\Scripts\python.exe" -m streamlit run app.py
exit /b 0

:no_venv
echo 먼저 SETUP_WINDOWS.bat을 더블클릭해 주세요.
pause
exit /b 1

:error
echo.
echo 실행 중 문제가 발생했습니다. 위 오류가 모두 보이게 캡처해 주세요.
pause
exit /b 1
