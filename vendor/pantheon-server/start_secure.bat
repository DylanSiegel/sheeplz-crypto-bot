@echo off
echo 🔒 Starting Pantheon Server with Secure Configuration
echo ==================================================

REM Check if environment variable is set
if "%PANTHEON_REDIS_PASSWORD%"=="" (
    echo ERROR: PANTHEON_REDIS_PASSWORD environment variable is not set
    echo Please set it with: $env:PANTHEON_REDIS_PASSWORD="your_password"
    echo Exiting...
    pause
    exit /b 1
)

echo 🔒 Using environment variable for Redis authentication
echo 🚀 Starting Pantheon Server...off
echo 🔒 Starting Pantheon Server with Secure Configuration
echo ==================================================

echo 🔒 Using environment variable for Redis authentication
echo � Starting Pantheon Server...

echo 🚀 Starting Pantheon Server...
"C:\Dev\repo\Pantheon\pantheon-server\.venv\Scripts\python.exe" run.py dev
