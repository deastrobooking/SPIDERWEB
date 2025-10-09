# Azure Deployment Authentication Setup

## Issue: Azure Login Failed in GitHub Actions

You're seeing this error because the GitHub Actions workflow needs Azure credentials to authenticate.

## Solution: Set Up Azure Service Principal

Follow these steps to configure authentication:

### Step 1: Create Azure Service Principal

Run this command in your terminal (you must be logged into Azure CLI):

```bash
# Login to Azure first
az login

# Get your subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "Subscription ID: ${SUBSCRIPTION_ID}"

# Create service principal with contributor role
az ad sp create-for-rbac \
  --name "github-actions-spiderweb-$(date +%s)" \
  --role contributor \
  --scopes /subscriptions/${SUBSCRIPTION_ID} \
  --sdk-auth
```

**IMPORTANT**: Save the entire JSON output! You'll need it in the next step.

The output will look like this:
```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

### Step 2: Add GitHub Secrets

1. Go to your GitHub repository: https://github.com/deastrobooking/SPIDERWEB
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the following secrets:

#### Required Secret for Azure Authentication

| Secret Name | Value |
|-------------|-------|
| `AZURE_CREDENTIALS` | Paste the **entire JSON output** from Step 1 |

#### Required Secrets for API Keys

| Secret Name | Description | Where to Get |
|-------------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API key (dev/staging) | https://platform.openai.com/api-keys |
| `ANTHROPIC_API_KEY` | Anthropic API key (dev/staging) | https://console.anthropic.com/ |
| `OPENAI_API_KEY_PROD` | OpenAI API key (production) | Same as above, use separate key |
| `ANTHROPIC_API_KEY_PROD` | Anthropic API key (production) | Same as above, use separate key |

#### Optional Secrets for Additional AI Services

| Secret Name | Description |
|-------------|-------------|
| `PERPLEXITY_API_KEY` | Perplexity API key (dev/staging) |
| `PERPLEXITY_API_KEY_PROD` | Perplexity API key (production) |
| `GEMINI_API_KEY` | Google Gemini API key (dev/staging) |
| `GEMINI_API_KEY_PROD` | Google Gemini API key (production) |
| `XAI_API_KEY` | xAI Grok API key (dev/staging) |
| `XAI_API_KEY_PROD` | xAI Grok API key (production) |

### Step 3: Create GitHub Environments

1. Go to **Settings** → **Environments**
2. Create three environments:

#### Development
- Click **New environment**
- Name: `development`
- Protection rules: None
- Click **Configure environment**

#### Staging
- Click **New environment**
- Name: `staging`
- Protection rules:
  - ☑ Required reviewers: Add yourself
  - ☑ Wait timer: 5 minutes (optional)
- Click **Configure environment**

#### Production
- Click **New environment**
- Name: `production`
- Protection rules:
  - ☑ Required reviewers: Add yourself (or 2+ team members)
  - ☑ Wait timer: 10 minutes (optional)
  - ☑ Deployment branches: Selected branches → `main`
- Click **Configure environment**

### Step 4: Deploy Infrastructure First (Important!)

Before GitHub Actions can deploy, you need to create the Azure infrastructure:

```bash
# Make sure you're in the project directory
cd /workspaces/SPIDERWEB

# Run the deployment script
./deploy-azure.sh
```

This creates:
- Resource Group: `spiderweb-ml-rg`
- Container Registry: `spiderwebmlacr`
- Container Apps Environment
- Initial Container App

### Step 5: Retry GitHub Actions Deployment

After completing steps 1-4, push your code:

```bash
# Add all changes
git add .

# Commit
git commit -m "Add Azure deployment configuration"

# Push to trigger deployment
git push origin main
```

Or manually trigger the workflow:
1. Go to **Actions** → **Deploy to Azure Container Apps**
2. Click **Run workflow**
3. Select environment: `prod`
4. Click **Run workflow**

## Alternative: Deploy Directly from Terminal

If you prefer to deploy directly from your terminal without GitHub Actions:

```bash
# 1. Make sure Azure CLI is installed and you're logged in
az login

# 2. Run the deployment script
./deploy-azure.sh

# 3. After deployment, configure API keys
RESOURCE_GROUP="spiderweb-ml-rg"
CONTAINER_APP_NAME="spiderweb-ml-app-prod"

az containerapp secret set \
  --name ${CONTAINER_APP_NAME} \
  --resource-group ${RESOURCE_GROUP} \
  --secrets \
    openai-api-key="${OPENAI_API_KEY:-demo-mode}" \
    anthropic-api-key="${ANTHROPIC_API_KEY:-demo-mode}" \
    perplexity-api-key="${PERPLEXITY_API_KEY:-demo-mode}" \
    gemini-api-key="${GEMINI_API_KEY:-demo-mode}" \
    xai-api-key="${XAI_API_KEY:-demo-mode}"

# 4. Get your app URL
APP_URL=$(az containerapp show \
  --name ${CONTAINER_APP_NAME} \
  --resource-group ${RESOURCE_GROUP} \
  --query properties.configuration.ingress.fqdn -o tsv)

echo "🎉 Deployment complete!"
echo "📍 App URL: https://${APP_URL}"
echo "🏥 Health check: https://${APP_URL}/health"
```

## Verification Steps

After setup, verify everything works:

```bash
# 1. Check if secrets are configured in GitHub
# Go to: Settings → Secrets and variables → Actions

# 2. Check Azure resources exist
az group show --name spiderweb-ml-rg
az acr show --name spiderwebmlacr

# 3. Test local deployment script
./deploy-azure.sh

# 4. Check if app is running
APP_URL=$(az containerapp show \
  --name spiderweb-ml-app-prod \
  --resource-group spiderweb-ml-rg \
  --query properties.configuration.ingress.fqdn -o tsv)

curl https://${APP_URL}/health
```

## Troubleshooting

### Issue: "az: command not found"
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### Issue: "Insufficient privileges"
```bash
# Make sure you have Owner or Contributor role on the subscription
az role assignment list --assignee $(az account show --query user.name -o tsv)
```

### Issue: "Resource group already exists"
```bash
# Use the existing resource group or delete it first
az group delete --name spiderweb-ml-rg --yes
./deploy-azure.sh
```

### Issue: GitHub Actions still failing
1. Verify `AZURE_CREDENTIALS` secret is the complete JSON (not just client ID)
2. Check that service principal has contributor access
3. Ensure GitHub environments are created correctly
4. Try running the workflow manually first

## Quick Checklist

- [ ] Azure CLI installed and logged in (`az login`)
- [ ] Service principal created
- [ ] `AZURE_CREDENTIALS` added to GitHub Secrets
- [ ] API key secrets added to GitHub Secrets
- [ ] GitHub environments created (development, staging, production)
- [ ] Initial infrastructure deployed (`./deploy-azure.sh`)
- [ ] App is accessible at the URL
- [ ] GitHub Actions workflow tested

## Need Help?

If you're still having issues:

1. Check the [CI/CD setup guide](docs/ci-cd-setup.md) for detailed instructions
2. Review the [deployment troubleshooting](docs/deployment-azure.md#troubleshooting) section
3. Verify your Azure subscription has sufficient permissions
4. Check GitHub Actions logs for specific error messages

---

**Next Steps**: Once authentication is configured, you can deploy with:
```bash
git push origin main
```

Or use the manual deployment script:
```bash
./deploy-azure.sh
```
