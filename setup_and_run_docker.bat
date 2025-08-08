@echo off
setlocal
chcp 65001 >nul

echo [AIOCR] Building Docker image...
docker --version >nul 2>&1
if errorlevel 1 (
  echo Docker が見つかりません。Docker Desktop をインストールして起動してください。
  pause
  exit /b 1
)

docker build -t aiocr-app .
if errorlevel 1 (
  echo 画像のビルドに失敗しました。
  pause
  exit /b 1
)

echo [AIOCR] Starting container on http://localhost:8501 ...
docker run --rm -p 8501:8501 -v "%cd%\workspace":/app/workspace -v "%cd%\templates":/app/templates aiocr-app

endlocal


