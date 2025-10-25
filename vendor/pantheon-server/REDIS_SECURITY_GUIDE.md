# 🔒 Redis Security Configuration Guide

## Overview

The Pantheon Server supports multiple methods for securely configuring the Redis password, with automatic fallback between environment sources.

## Security Priority Order

The Redis service will check for passwords in this order:
1. **System Environment Variable**: `PANTHEON_REDIS_PASSWORD` (Most Secure)
2. **Local Environment File**: `REDIS_PASSWORD` in `.env` (Development)

## Development Setup (Current)

For local development, the password should be set via environment variable:

```env
# Password should be set via PANTHEON_REDIS_PASSWORD environment variable
# Do not store passwords in .env file for security
```

**⚠️ Security Note**: Never store passwords in version-controlled files.

## Production Setup (Recommended)

### Option 1: System Environment Variable

Set the Redis password as a system environment variable:

#### Windows
```powershell
# Persistent user environment variable
[System.Environment]::SetEnvironmentVariable("PANTHEON_REDIS_PASSWORD", "your_secure_password", "User")

# System-wide (requires admin)
[System.Environment]::SetEnvironmentVariable("PANTHEON_REDIS_PASSWORD", "your_secure_password", "Machine")
```

#### Linux/macOS
```bash
# Add to ~/.bashrc or ~/.profile
export PANTHEON_REDIS_PASSWORD="your_secure_password"

# System-wide (requires sudo)
echo 'PANTHEON_REDIS_PASSWORD="your_secure_password"' | sudo tee -a /etc/environment
```

### Option 2: Docker Environment

For containerized deployments:

```yaml
# docker-compose.yml
services:
  pantheon-server:
    environment:
      - PANTHEON_REDIS_PASSWORD=your_secure_password
```

### Option 3: Cloud Secret Management

#### Azure Key Vault
```python
from azure.keyvault.secrets import SecretClient
# Retrieve password from Azure Key Vault
```

#### AWS Secrets Manager
```python
import boto3
# Retrieve password from AWS Secrets Manager
```

#### Kubernetes Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: redis-credentials
data:
  password: <base64-encoded-password>
```

## Migration Guide

### Step 1: Set System Environment Variable

Choose your platform and set the environment variable using the commands above.

### Step 2: Update .env File

Remove the password from `.env` file:

```env
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
# REDIS_PASSWORD is now handled via PANTHEON_REDIS_PASSWORD environment variable
REDIS_DB=0
```

### Step 3: Restart Application

The Redis service will automatically detect and use the system environment variable.

### Step 4: Verify Configuration

Check that the password is loaded from the environment variable:

```bash
# The application will log which password source is being used
# Look for debug logs: "PANTHEON_REDIS_PASSWORD: ***" vs "REDIS_PASSWORD: ***"
```

## Security Best Practices

### ✅ Do
- Use system environment variables for production
- Rotate passwords regularly
- Use strong, unique passwords
- Limit access to environment variables
- Use secret management services in cloud environments
- Keep `.env` in `.gitignore`

### ❌ Don't
- Commit passwords to version control
- Use weak or default passwords
- Share environment configuration files
- Use the same password across environments
- Store passwords in plain text files

## Current Implementation

The Redis service automatically handles both methods:

```python
# Priority: System env var > .env file
self.password = os.getenv("PANTHEON_REDIS_PASSWORD") or os.getenv("REDIS_PASSWORD")
```

This provides:
- **Development flexibility**: Works with .env files
- **Production security**: Prioritizes system environment variables
- **Graceful fallback**: Ensures the application works in both scenarios
- **Zero configuration**: Automatic detection and usage

## Verification

To verify your security setup:

1. **Check environment variables**:
   ```bash
   echo $PANTHEON_REDIS_PASSWORD  # Linux/macOS
   echo %PANTHEON_REDIS_PASSWORD% # Windows CMD
   echo $env:PANTHEON_REDIS_PASSWORD # Windows PowerShell
   ```

2. **Check application logs**: Look for Redis connection success messages

3. **Test Redis connectivity**: The `/cache/health` endpoint shows Redis status

## Status

- ✅ **Current**: Development-ready with .env file support
- ✅ **Production-ready**: Environment variable fallback implemented
- ✅ **Secure**: No hardcoded passwords in code
- ✅ **Flexible**: Multiple configuration methods supported
