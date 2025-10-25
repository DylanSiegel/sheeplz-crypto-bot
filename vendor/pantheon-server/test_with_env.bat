@echo off
echo 🧪 Testing Secure Redis Cache with Environment Variables
echo ==================================================

REM Check if environment variable is set
if "%PANTHEON_REDIS_PASSWORD%"=="" (
    echo ERROR: PANTHEON_REDIS_PASSWORD environment variable is not set
    echo Please set it with: $env:PANTHEON_REDIS_PASSWORD="your_password"
    echo Exiting...
    exit /b 1
)

echo Environment variable configured externally

echo.
echo Running test script...
python test_final_secure.py

echo.
echo Test complete!
pause
