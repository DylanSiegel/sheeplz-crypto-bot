#!/usr/bin/env powershell
# Secure startup script for Pantheon Server
# Sets environment variable and starts the server

Write-Host "🔒 Starting Pantheon Server with Secure Configuration" -ForegroundColor Green
Write-Host "=" * 50

# Set Redis password environment variable for this session
$env:PANTHEON_REDIS_PASSWORD = "pantheon_server**!"

# Verify environment variable is set
if ($env:PANTHEON_REDIS_PASSWORD) {
    Write-Host "✅ Environment variable set: PANTHEON_REDIS_PASSWORD" -ForegroundColor Green
    Write-Host "🔒 Password length: $($env:PANTHEON_REDIS_PASSWORD.Length) characters" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to set environment variable" -ForegroundColor Red
    exit 1
}

# Start the server
Write-Host "🚀 Starting Pantheon Server..." -ForegroundColor Yellow
$pythonPath = "C:\Dev\repo\Pantheon\pantheon-server\.venv\Scripts\python.exe"
& $pythonPath run.py dev
