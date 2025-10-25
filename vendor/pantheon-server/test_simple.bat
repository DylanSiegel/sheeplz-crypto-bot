@echo off
echo 🧪 Simple Redis Test
echo ===================

REM Check if environment variable is set
if "%PANTHEON_REDIS_PASSWORD%"=="" (
    echo ERROR: PANTHEON_REDIS_PASSWORD environment variable is not set
    echo Please set it with: $env:PANTHEON_REDIS_PASSWORD="your_password"
    echo Exiting...
    exit /b 1
)

python test_simple.py
