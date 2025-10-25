@echo off
echo Running Complete Redis Cache Test Suite
echo ==========================================

REM Check if environment variable is set (batch file may not see PowerShell session vars)
echo Checking environment variable...
echo PANTHEON_REDIS_PASSWORD: %PANTHEON_REDIS_PASSWORD%

echo Environment configured for testing
echo.

rem Run the comprehensive test suite
python test_redis_complete.py

echo.
echo Test suite completed!
pause
