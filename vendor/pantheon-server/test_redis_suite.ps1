# PowerShell Test Suite for Redis Cache
Write-Host "Running Complete Redis Cache Test Suite" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Check if environment variable is set
if ([string]::IsNullOrEmpty($env:PANTHEON_REDIS_PASSWORD)) {
    Write-Host "ERROR: PANTHEON_REDIS_PASSWORD environment variable is not set" -ForegroundColor Red
    Write-Host "Please set it with: `$env:PANTHEON_REDIS_PASSWORD=`"your_password`"" -ForegroundColor Yellow
    Write-Host "Exiting..." -ForegroundColor Red
    exit 1
}

Write-Host "Environment configured for testing" -ForegroundColor Green
Write-Host ""

# Run the comprehensive test suite
python test_redis_complete.py
