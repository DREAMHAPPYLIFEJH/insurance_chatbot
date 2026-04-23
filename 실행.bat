@echo off
title 동양생명 가입설계 챗봇
echo.
echo ============================================
echo   동양생명 가입설계 챗봇 시작 중...
echo ============================================
echo.

:: Docker Desktop 실행 확인
echo [1/4] Docker 확인 중...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker Desktop을 시작합니다. 잠시 기다려주세요...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo Docker가 완전히 켜질 때까지 대기 중...
    :wait_docker
    timeout /t 5 /nobreak >nul
    docker info >nul 2>&1
    if %errorlevel% neq 0 goto wait_docker
    echo Docker 준비 완료!
)

echo.
echo [2/4] 기존 컨테이너 정리 중...
docker-compose down >nul 2>&1

echo.
echo [3/4] 챗봇 서버 시작 중... (AI 모델 로딩으로 1~2분 소요)
docker-compose up -d --build

echo.
echo [4/4] 서버 준비 대기 중...
:wait_server
timeout /t 5 /nobreak >nul
curl -s http://localhost:8000/docs >nul 2>&1
if %errorlevel% neq 0 (
    echo    아직 로딩 중...
    goto wait_server
)

echo.
echo ============================================
echo   챗봇이 준비되었습니다!
echo ============================================
echo.
start "" http://localhost:8501

echo 브라우저가 자동으로 열립니다.
echo 종료하려면 이 창을 닫지 말고 아무 키나 누르세요.
echo (창을 닫으면 서버도 종료됩니다)
pause >nul

echo.
echo 서버를 종료합니다...
docker-compose down
echo 종료 완료.
