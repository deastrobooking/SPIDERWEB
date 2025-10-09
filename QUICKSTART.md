# SPIDERWEB - Quick Start Guide

Get SPIDERWEB running locally and deploy to Azure in minutes.

## Local Development (5 minutes)

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install anthropic flask openai requests

# Or use uv (faster)
uv pip install -r pyproject.toml
```

### 2. Configure API Keys (Optional)
```bash
# For full functionality, set at least one API key
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# App works in demo mode without keys
```

### 3. Start the Server
```bash
python3 demo_server.py
```

### 4. Open Dashboard
Visit: http://localhost:5000

## Azure Deployment (15 minutes)

### Quick Deploy
```bash
# 1. Login to Azure
az login

# 2. Run automated deployment
./deploy-azure.sh

# 3. Configure secrets (after deployment)
az containerapp secret set \
  --name spiderweb-ml-app-prod \
  --resource-group spiderweb-ml-rg \
  --secrets openai-api-key="sk-..." anthropic-api-key="sk-ant-..."
```

### Access Your App
The deployment script outputs your app URL:
```
https://spiderweb-ml-app-prod.azurecontainerapps.io
```

## GitHub Actions CI/CD (10 minutes)

### 1. Create Service Principal
```bash
az ad sp create-for-rbac \
  --name "github-actions-spiderweb" \
  --role contributor \
  --scopes /subscriptions/$(az account show --query id -o tsv)/resourceGroups/spiderweb-ml-rg \
  --sdk-auth
```

### 2. Add GitHub Secrets
Go to: Repository → Settings → Secrets → Actions

Add these secrets:
- `AZURE_CREDENTIALS` (output from step 1)
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

### 3. Push to Deploy
```bash
git push origin main  # Deploys to production
git push origin develop  # Deploys to dev
```

## API Examples

### Check Service Status
```bash
curl https://your-app.azurecontainerapps.io/v1/ai/status
```

### Generate Synthetic Data
```bash
curl -X POST https://your-app.azurecontainerapps.io/v1/ai/synthetic-data \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_type": "classification",
    "num_samples": 100,
    "features": ["age", "income"],
    "target": "approved"
  }'
```

### Analyze Model
```bash
curl -X POST https://your-app.azurecontainerapps.io/v1/ai/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "neural_network",
    "architecture": "3-layer MLP",
    "performance_metrics": {"accuracy": 0.85}
  }'
```

## Next Steps

- 📖 Read the [full documentation](docs/README.md)
- 🚀 Review [Azure deployment guide](docs/deployment-azure.md)
- 🔄 Set up [CI/CD pipeline](docs/ci-cd-setup.md)
- ✅ Complete [deployment checklist](docs/deployment-checklist.md)

## Troubleshooting

### Local Server Won't Start
```bash
# Check Python version (needs 3.11+)
python3 --version

# Install missing dependencies
pip install flask openai anthropic
```

### Azure Deployment Fails
```bash
# Check Azure CLI login
az account show

# Verify resource group exists
az group show --name spiderweb-ml-rg

# Check deployment logs
az containerapp logs show --name spiderweb-ml-app-prod --resource-group spiderweb-ml-rg
```

### API Keys Not Working
- Verify keys are valid at provider's website
- Check secret configuration in Azure
- Restart the container app after updating secrets

## Support

For help:
1. Check [documentation](docs/)
2. Review [troubleshooting guide](docs/deployment-azure.md#troubleshooting)
3. Open an issue on GitHub
4. Contact Azure support for Azure-specific issues

---

**Built with ❤️ using Rust and Python**
