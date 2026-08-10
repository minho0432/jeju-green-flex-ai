@echo off
chcp 65001 > nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto no_venv

echo [1/3] 제주 시간별 SMP를 가져옵니다.
".venv\Scripts\python.exe" scripts\download_smp.py
if errorlevel 1 goto error

echo [2/3] 발전량, SMP, 날씨를 하나로 합칩니다.
".venv\Scripts\python.exe" scripts\prepare_data.py
if errorlevel 1 goto error

echo [3/3] 데이터 오류를 검사합니다.
".venv\Scripts\python.exe" scripts\validate_data.py
if errorlevel 1 goto error

echo.
echo 완료되었습니다.
echo AI 담당자에게 data\processed\train.csv 파일을 전달하세요.
pause
exit /b 0

:no_venv
echo 먼저 SETUP_WINDOWS.bat을 더블클릭해 주세요.
pause
exit /b 1

:error
echo.
echo 실행 중 문제가 발생했습니다. 이 화면을 캡처해 전달해 주세요.
pause
exit /b 1
