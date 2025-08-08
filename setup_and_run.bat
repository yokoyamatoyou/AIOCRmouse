@echo off
setlocal

REM Force UTF-8 code page to avoid mojibake
chcp 65001 >nul

REM --- Settings ---
set "VENV_DIR=.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "REQUIREMENTS_FILE=requirements.txt"

echo [AIOCR] Setup and launch starting...

REM Create venv if missing
if not exist "%VENV_DIR%\Scripts\activate" (
    echo Creating virtual environment: %VENV_DIR%
    py -3 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment. Ensure Python is installed and in PATH.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate"

echo Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

echo Installing dependencies from %REQUIREMENTS_FILE% ...
"%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo pip install failed.
    pause
    exit /b 1
)

REM Add src to PYTHONPATH
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

echo Launching Streamlit app...
"%PYTHON_EXE%" -m streamlit run src/app/main.py

echo.
echo If the app did not open automatically, open: http://localhost:8501

endlocal
