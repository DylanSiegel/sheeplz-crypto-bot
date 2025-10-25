# 🔒 Security Configuration Guide

## Redis Password Security Setup

For security reasons, the Redis password should be stored as a system environment variable rather than in configuration files.

### Windows Setup

#### Option 1: PowerShell (Persistent - Recommended)
```powershell
# Set user environment variable (persists across sessions)
[System.Environment]::SetEnvironmentVariable("PANTHEON_REDIS_PASSWORD", "your_redis_password_here", "User")

# Verify it's set
$env:PANTHEON_REDIS_PASSWORD
```

#### Option 2: Command Prompt (Persistent)
```cmd
# Set user environment variable
setx PANTHEON_REDIS_PASSWORD "your_redis_password_here"

# Verify it's set (after reopening terminal)
echo %PANTHEON_REDIS_PASSWORD%
```

#### Option 3: Windows GUI
1. Open "Environment Variables" from Start menu
2. Click "Environment Variables..." button
3. Under "User variables", click "New..."
4. Variable name: `PANTHEON_REDIS_PASSWORD`
5. Variable value: `your_redis_password_here`
6. Click OK

### Linux/macOS Setup

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export PANTHEON_REDIS_PASSWORD="your_redis_password_here"' >> ~/.bashrc

# Reload shell configuration
source ~/.bashrc

# Verify it's set
echo $PANTHEON_REDIS_PASSWORD
```

### Docker Environment Setup

```bash
# For Docker containers
docker run -e PANTHEON_REDIS_PASSWORD="your_redis_password_here" ...

# Or in docker-compose.yml
environment:
  - PANTHEON_REDIS_PASSWORD=your_redis_password_here
```

### Production Environment Setup

For production deployments, consider using:

- **Azure Key Vault** (Azure)
- **AWS Secrets Manager** (AWS)
- **Google Secret Manager** (GCP)
- **HashiCorp Vault** (On-premise)
- **Kubernetes Secrets** (K8s)

### Verification

After setting the environment variable, restart your terminal and verify:

```bash
# Check if the application can access the password
python -c "import os; print('Password set:', bool(os.getenv('PANTHEON_REDIS_PASSWORD')))"
```

### Security Best Practices

1. ✅ **Never commit passwords to version control**
2. ✅ **Use environment variables for sensitive data**
3. ✅ **Rotate passwords regularly**
4. ✅ **Use strong, unique passwords**
5. ✅ **Limit access to environment variables**
6. ✅ **Use secret management services in production**

### Current Configuration

The application will check for the Redis password in this order:
1. `REDIS_PASSWORD` environment variable (from .env file)
2. `PANTHEON_REDIS_PASSWORD` system environment variable

This provides flexibility while maintaining security.
