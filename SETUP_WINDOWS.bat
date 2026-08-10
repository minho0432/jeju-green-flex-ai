@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ==================================================
echo Jeju Green Flex AI 전용 파이썬 환경을 준비합니다.
echo 이 창을 닫지 말고 완료 문구를 확인하세요.
echo ==================================================
echo.

where py > nul 2>&1
if errorlevel 1 goto use_python
set "PY_CMD=py"
goto python_found

:use_python
where python > nul 2>&1
if errorlevel 1 goto no_python
set "PY_CMD=python"

:python_found
echo [1/5] 사용 중인 파이썬을 확인합니다.
%PY_CMD% --version
if errorlevel 1 goto error

echo.
echo [2/5] 이 프로젝트만 사용하는 .venv 폴더를 만듭니다.
%PY_CMD% -m venv .venv
if errorlevel 1 goto error

echo.
echo [3/5] pip를 준비합니다.
".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error

echo.
echo [4/5] 프로젝트에 필요한 도구를 설치합니다.
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo [5/5] pandas 설치와 데이터를 검사합니다.
".venv\Scripts\python.exe" -c "import sys, pandas; print('Python 위치:', sys.executable); print('pandas 버전:', pandas.__version__)"
if errorlevel 1 goto error
".venv\Scripts\python.exe" scripts\validate_data.py
if errorlevel 1 goto error

echo.
echo ==================================================
echo 설치와 검사가 모두 성공했습니다.
echo VS Code에서 Ctrl+Shift+P를 누른 뒤
echo Python: Select Interpreter를 선택하고
echo .venv\Scripts\python.exe를 선택하세요.
echo ==================================================
pause
exit /b 0

:no_python
echo.
echo Python을 찾지 못했습니다.
echo Python 설치 시 Add Python to PATH를 체크해야 합니다.
pause
exit /b 1

:error
echo.
echo ==================================================
echo 위쪽에 표시된 빨간색 오류 내용을 캡처해서 보내 주세요.
echo 마지막 한 줄만 자르지 말고 전체가 보이게 찍어 주세요.
echo ==================================================
pause
exit /b 1
