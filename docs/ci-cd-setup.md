# CI/CD Setup Guide for SPIDERWEB

This guide walks you through setting up automated CI/CD deployments to Azure using GitHub Actions.

## Overview

The CI/CD pipeline provides:
- **Automated Testing**: Runs on every push and pull request
- **Multi-Environment Deployment**: Dev, Staging, and Production
- **Docker Image Building**: Automated container builds in Azure Container Registry
- **Health Checks**: Automated verification after deployment
- **Secrets Management**: Secure API key handling

## Architecture

```
GitHub Push → GitHub Actions → Build & Test → Deploy to Azure
                                    ↓
                            Azure Container Registry
                                    ↓
                            Azure Container Apps
```

## Prerequisites

1. **Azure Subscription** with permissions for:
   - Container Apps
   - Container Registry
   - Resource Groups

2. **GitHub Repository** with admin access

3. **Azure CLI** installed locally (for setup)

## Setup Instructions

### Step 1: Create Azure Service Principal

Create a service principal for GitHub Actions to authenticate with Azure:

```bash
# Set your subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

# Create service principal
az ad sp create-for-rbac \
  --name "github-actions-spiderweb" \
  --role contributor \
  --scopes /subscriptions/${SUBSCRIPTION_ID}/resourceGroups/spiderweb-ml-rg \
  --sdk-auth
```

Save the entire JSON output - you'll need it in the next step.

### Step 2: Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add the following secrets:

#### Required Secrets

| Secret Name | Description | How to Get |
|-------------|-------------|------------|
| `AZURE_CREDENTIALS` | Azure service principal credentials | Output from Step 1 |
| `OPENAI_API_KEY` | OpenAI API key (dev/staging) | https://platform.openai.com |
| `ANTHROPIC_API_KEY` | Anthropic API key (dev/staging) | https://console.anthropic.com |
| `OPENAI_API_KEY_PROD` | OpenAI API key (production) | Same as above, separate key |
| `ANTHROPIC_API_KEY_PROD` | Anthropic API key (production) | Same as above, separate key |

#### Optional Secrets

| Secret Name | Description |
|-------------|-------------|
| `PERPLEXITY_API_KEY` | Perplexity API key (dev/staging) |
| `PERPLEXITY_API_KEY_PROD` | Perplexity API key (production) |
| `GEMINI_API_KEY` | Google Gemini API key (dev/staging) |
| `GEMINI_API_KEY_PROD` | Google Gemini API key (production) |
| `XAI_API_KEY` | xAI Grok API key (dev/staging) |
| `XAI_API_KEY_PROD` | xAI Grok API key (production) |

### Step 3: Initial Azure Infrastructure Setup

Run the deployment script once to create initial infrastructure:

```bash
# Run from the repository root
./deploy-azure.sh
```

This creates:
- Resource Group
- Container Registry
- Container Apps Environment
- Log Analytics Workspace

### Step 4: Create GitHub Environments

Go to your GitHub repository → Settings → Environments

Create three environments with the following settings:

#### Development Environment
- **Name**: `development`
- **Protection rules**: None (auto-deploy)
- **Deployment branches**: `develop` branch only

#### Staging Environment
- **Name**: `staging`
- **Protection rules**: 
  - Required reviewers: 1
  - Wait timer: 5 minutes
- **Deployment branches**: Manual trigger only

#### Production Environment
- **Name**: `production`
- **Protection rules**:
  - Required reviewers: 2
  - Wait timer: 10 minutes
- **Deployment branches**: `main` branch only

### Step 5: Verify Workflow File

Ensure `.github/workflows/azure-deploy.yml` exists and is configured correctly:

```bash
# Check if workflow file exists
ls -la .github/workflows/azure-deploy.yml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('.github/workflows/azure-deploy.yml'))"
```

## Usage

### Automatic Deployments

#### Deploy to Development
```bash
# Push to develop branch
git checkout develop
git add .
git commit -m "Feature: Add new functionality"
git push origin develop
```

#### Deploy to Production
```bash
# Push to main branch (after PR approval)
git checkout main
git merge develop
git push origin main
```

### Manual Deployments

#### Deploy to Any Environment

1. Go to GitHub repository → Actions
2. Select "Deploy to Azure Container Apps" workflow
3. Click "Run workflow"
4. Select environment (dev/staging/prod)
5. Click "Run workflow"

## Monitoring Deployments

### View Workflow Runs

1. Go to GitHub repository → Actions
2. Click on a workflow run to see details
3. View logs for each job and step

### Check Deployment Status

```bash
# Get Container App status
az containerapp show \
  --name spiderweb-ml-app-prod \
  --resource-group spiderweb-ml-rg \
  --query properties.runningStatus

# View recent revisions
az containerapp revision list \
  --name spiderweb-ml-app-prod \
  --resource-group spiderweb-ml-rg \
  --query "[].{Name:name, Active:properties.active, Created:properties.createdTime}"
```

### View Application Logs

```bash
# Stream logs from production
az containerapp logs show \
  --name spiderweb-ml-app-prod \
  --resource-group spiderweb-ml-rg \
  --follow
```

## Workflow Details

### Build and Test Job

Runs on every push and PR:
1. Checkout code
2. Set up Python 3.11
3. Install dependencies
4. Run tests
5. Build Docker image (no push)

### Deploy Jobs

Three separate deploy jobs for each environment:

#### Development
- **Trigger**: Push to `develop` branch
- **Resources**: Min 0, Max 5 replicas (scale-to-zero)
- **CPU/Memory**: 1.0 CPU, 2Gi memory

#### Staging
- **Trigger**: Manual workflow dispatch
- **Resources**: Min 1, Max 10 replicas
- **CPU/Memory**: 1.0 CPU, 2Gi memory
- **Approval**: 1 reviewer required

#### Production
- **Trigger**: Push to `main` branch
- **Resources**: Min 2, Max 20 replicas
- **CPU/Memory**: 2.0 CPU, 4Gi memory
- **Approval**: 2 reviewers required
- **Health checks**: Automated verification

## Customization

### Modify Deployment Parameters

Edit `.github/workflows/azure-deploy.yml`:

```yaml
# Example: Increase production resources
- name: Deploy to Container Apps
  run: |
    az deployment group create \
      --parameters \
        minReplicas=3 \
        maxReplicas=30 \
        cpuCores=4.0 \
        memorySize=8Gi
```

### Add Additional Tests

Add test steps in the `build-and-test` job:

```yaml
- name: Run integration tests
  run: |
    python -m pytest tests/integration/

- name: Run security scan
  run: |
    pip install bandit
    bandit -r demo_server.py
```

### Add Notifications

Add notification steps:

```yaml
- name: Notify Slack
  if: success()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## Troubleshooting

### Deployment Fails

1. **Check workflow logs** in GitHub Actions
2. **Verify secrets** are configured correctly
3. **Check Azure resources** exist:
   ```bash
   az group show --name spiderweb-ml-rg
   az acr show --name spiderwebmlacr
   ```

### Authentication Errors

Recreate service principal:
```bash
# Delete old service principal
az ad sp delete --id <app-id>

# Create new one
az ad sp create-for-rbac \
  --name "github-actions-spiderweb" \
  --role contributor \
  --scopes /subscriptions/${SUBSCRIPTION_ID}/resourceGroups/spiderweb-ml-rg \
  --sdk-auth
```

Update `AZURE_CREDENTIALS` secret in GitHub.

### Image Build Fails

Check Dockerfile and dependencies:
```bash
# Test build locally
docker build -t spiderweb-ml:test .

# Test run locally
docker run -p 5000:5000 spiderweb-ml:test
```

### Health Check Fails

1. Check application logs
2. Verify environment variables
3. Test health endpoint manually:
   ```bash
   curl https://your-app.azurecontainerapps.io/health
   ```

## Best Practices

### Security
- ✅ Use separate API keys for each environment
- ✅ Store all secrets in GitHub Secrets
- ✅ Enable branch protection rules
- ✅ Require pull request reviews
- ✅ Use minimal permissions for service principal

### Deployment Strategy
- ✅ Always test in dev first
- ✅ Use staging for final validation
- ✅ Deploy to production during low-traffic periods
- ✅ Monitor logs after deployment
- ✅ Have rollback plan ready

### Performance
- ✅ Use Docker build cache
- ✅ Minimize image size
- ✅ Set appropriate resource limits
- ✅ Configure auto-scaling rules
- ✅ Monitor costs regularly

## Rollback Procedure

### Manual Rollback

```bash
# List revisions
az containerapp revision list \
  --name spiderweb-ml-app-prod \
  --resource-group spiderweb-ml-rg

# Activate previous revision
az containerapp revision activate \
  --name spiderweb-ml-app-prod \
  --resource-group spiderweb-ml-rg \
  --revision <previous-revision-name>

# Deactivate current revision
az containerapp revision deactivate \
  --name spiderweb-ml-app-prod \
  --resource-group spiderweb-ml-rg \
  --revision <current-revision-name>
```

### Automated Rollback

Add to workflow:

```yaml
- name: Rollback on failure
  if: failure()
  run: |
    PREVIOUS_REVISION=$(az containerapp revision list \
      --name spiderweb-ml-app-prod \
      --resource-group spiderweb-ml-rg \
      --query "[?properties.active==\`false\`] | [0].name" -o tsv)
    
    az containerapp revision activate \
      --name spiderweb-ml-app-prod \
      --resource-group spiderweb-ml-rg \
      --revision $PREVIOUS_REVISION
```

## Cost Management

Monitor GitHub Actions usage:
- **Free tier**: 2,000 minutes/month for public repos
- **Private repos**: 3,000 minutes/month (Pro)
- **Additional**: $0.008/minute

Optimize workflow:
- Use workflow caching
- Skip unnecessary steps with `if` conditions
- Use self-hosted runners for large deployments

## Next Steps

- Set up [monitoring and alerting](monitoring.md)
- Configure [custom domains](custom-domain.md)
- Review [security best practices](security.md)
- Implement [blue-green deployments](blue-green-deploy.md)

---

**Last Updated**: October 2025
