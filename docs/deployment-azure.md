# Azure Deployment Guide for SPIDERWEB

This guide provides comprehensive instructions for deploying the SPIDERWEB ML-as-a-Service Platform to Microsoft Azure.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Options](#deployment-options)
3. [Quick Start Deployment](#quick-start-deployment)
4. [Manual Deployment Steps](#manual-deployment-steps)
5. [Configuration](#configuration)
6. [Monitoring and Scaling](#monitoring-and-scaling)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools

- **Azure CLI** (version 2.50.0+)
  ```bash
  curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
  ```

- **Docker** (for local testing)
  ```bash
  sudo apt-get update
  sudo apt-get install docker.io
  ```

- **Azure Subscription** with permissions to create:
  - Resource Groups
  - Container Registries
  - Container Apps / App Services
  - Log Analytics Workspaces

### API Keys

Obtain API keys from the following providers:

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Perplexity**: https://www.perplexity.ai/settings/api
- **Google Gemini**: https://makersuite.google.com/app/apikey
- **xAI (Grok)**: https://console.x.ai/

> **Note**: The application supports graceful degradation. You can deploy with partial API key configuration.

## Deployment Options

### Option 1: Azure Container Apps (Recommended)
- **Best for**: Production workloads with automatic scaling
- **Cost**: Pay-per-use, scales to zero
- **Features**: Built-in load balancing, automatic HTTPS, managed certificates

### Option 2: Azure App Service
- **Best for**: Traditional web app deployments
- **Cost**: Fixed monthly cost based on tier
- **Features**: Easy deployment, good for stable workloads

### Option 3: Azure Kubernetes Service (AKS)
- **Best for**: Complex microservice architectures
- **Cost**: Higher, requires cluster management
- **Features**: Maximum flexibility and control

## Quick Start Deployment

### Automated Deployment Script

The fastest way to deploy SPIDERWEB to Azure:

```bash
# 1. Clone the repository (if not already)
git clone <repository-url>
cd SPIDERWEB

# 2. Login to Azure
az login

# 3. Run the deployment script
./deploy-azure.sh
```

The script will:
1. Create a resource group
2. Set up Azure Container Registry
3. Build and push the Docker image
4. Deploy infrastructure using Bicep
5. Create a Container App with auto-scaling

### Configure API Keys

After deployment completes, configure your API keys:

```bash
# Set your resource group and app name
RESOURCE_GROUP="spiderweb-ml-rg"
CONTAINER_APP_NAME="spiderweb-ml-app-prod"

# Configure secrets
az containerapp secret set \
  --name ${CONTAINER_APP_NAME} \
  --resource-group ${RESOURCE_GROUP} \
  --secrets \
    openai-api-key="sk-..." \
    anthropic-api-key="sk-ant-..." \
    perplexity-api-key="pplx-..." \
    gemini-api-key="..." \
    xai-api-key="xai-..."

# Restart the app to apply changes
az containerapp revision restart \
  --name ${CONTAINER_APP_NAME} \
  --resource-group ${RESOURCE_GROUP}
```

## Manual Deployment Steps

### Step 1: Create Resource Group

```bash
RESOURCE_GROUP="spiderweb-ml-rg"
LOCATION="eastus"

az group create \
  --name ${RESOURCE_GROUP} \
  --location ${LOCATION}
```

### Step 2: Create Container Registry

```bash
ACR_NAME="spiderwebmlacr"

az acr create \
  --resource-group ${RESOURCE_GROUP} \
  --name ${ACR_NAME} \
  --sku Basic \
  --admin-enabled true
```

### Step 3: Build and Push Docker Image

```bash
# Login to ACR
az acr login --name ${ACR_NAME}

# Build and push
az acr build \
  --registry ${ACR_NAME} \
  --image spiderweb-ml:latest \
  --image spiderweb-ml:v1.0.0 \
  --file Dockerfile \
  .
```

### Step 4: Deploy with Bicep

```bash
# Deploy infrastructure
az deployment group create \
  --resource-group ${RESOURCE_GROUP} \
  --template-file bicep/main.bicep \
  --parameters \
    appName="spiderweb-ml" \
    environment="prod" \
    containerRegistryName=${ACR_NAME}
```

### Step 5: Verify Deployment

```bash
# Get the app URL
APP_URL=$(az containerapp show \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP} \
  --query properties.configuration.ingress.fqdn -o tsv)

# Test the health endpoint
curl https://${APP_URL}/health

# Test the API status
curl https://${APP_URL}/v1/ai/status
```

## Configuration

### Environment Variables

The following environment variables are automatically configured:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Optional* |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | Optional* |
| `PERPLEXITY_API_KEY` | Perplexity API key | Optional* |
| `GEMINI_API_KEY` | Google Gemini API key | Optional* |
| `XAI_API_KEY` | xAI Grok API key | Optional* |
| `FLASK_ENV` | Flask environment (production) | Auto-set |
| `PYTHONUNBUFFERED` | Python output buffering (1) | Auto-set |

\* At least one AI service API key is recommended for full functionality. The app runs in demo mode without any keys.

### Scaling Configuration

#### Container Apps Auto-Scaling

Edit `bicep/main.bicep` to adjust scaling parameters:

```bicep
scale: {
  minReplicas: 1      // Minimum instances (0 for scale-to-zero)
  maxReplicas: 10     // Maximum instances
  rules: [
    {
      name: 'http-scaling'
      http: {
        metadata: {
          concurrentRequests: '50'  // Requests per instance
        }
      }
    }
  ]
}
```

#### Resource Allocation

Adjust CPU and memory in deployment parameters:

```bash
az deployment group create \
  --template-file bicep/main.bicep \
  --parameters \
    cpuCores="2.0" \
    memorySize="4Gi"
```

Available configurations:
- **CPU**: 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0 cores
- **Memory**: 0.5Gi, 1.0Gi, 1.5Gi, 2.0Gi, 2.5Gi, 3.0Gi, 3.5Gi, 4.0Gi

### Custom Domain and HTTPS

#### Add Custom Domain

```bash
# Add custom domain
az containerapp hostname add \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP} \
  --hostname ml.yourdomain.com

# Bind certificate (managed certificate)
az containerapp hostname bind \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP} \
  --hostname ml.yourdomain.com \
  --environment spiderweb-ml-env-prod \
  --validation-method CNAME
```

## Monitoring and Scaling

### View Logs

```bash
# Stream live logs
az containerapp logs show \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP} \
  --follow

# Query specific time range
az containerapp logs show \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP} \
  --tail 100
```

### View Metrics

```bash
# View replica count
az containerapp replica list \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP}

# View revision history
az containerapp revision list \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP}
```

### Application Insights Integration

To enable detailed telemetry:

```bash
# Create Application Insights
az monitor app-insights component create \
  --app spiderweb-ml-insights \
  --location ${LOCATION} \
  --resource-group ${RESOURCE_GROUP}

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app spiderweb-ml-insights \
  --resource-group ${RESOURCE_GROUP} \
  --query instrumentationKey -o tsv)

# Update container app
az containerapp update \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP} \
  --set-env-vars APPINSIGHTS_INSTRUMENTATIONKEY=${INSTRUMENTATION_KEY}
```

## Troubleshooting

### Common Issues

#### 1. Container Fails to Start

```bash
# Check logs
az containerapp logs show \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP} \
  --tail 50

# Check revision status
az containerapp revision show \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP}
```

#### 2. API Keys Not Working

Verify secrets are properly configured:

```bash
# List secrets (values hidden)
az containerapp secret list \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP}

# Update a secret
az containerapp secret set \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP} \
  --secrets openai-api-key="new-key-value"
```

#### 3. High Latency or Timeouts

Increase resources or scaling limits:

```bash
# Scale up manually
az containerapp update \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP} \
  --min-replicas 2 \
  --max-replicas 20 \
  --cpu 2.0 \
  --memory 4.0Gi
```

#### 4. Image Pull Failures

Verify ACR credentials:

```bash
# Get ACR credentials
az acr credential show --name ${ACR_NAME}

# Update container app registry credentials
az containerapp registry set \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP} \
  --server ${ACR_NAME}.azurecr.io \
  --username $(az acr credential show --name ${ACR_NAME} --query username -o tsv) \
  --password $(az acr credential show --name ${ACR_NAME} --query passwords[0].value -o tsv)
```

### Health Check Endpoints

- **Health**: `https://your-app.azurecontainerapps.io/health`
- **AI Status**: `https://your-app.azurecontainerapps.io/v1/ai/status`
- **Dashboard**: `https://your-app.azurecontainerapps.io/`

### Performance Tuning

#### Optimize for Cost

```bash
# Enable scale-to-zero for non-production
az containerapp update \
  --name spiderweb-ml-app-dev \
  --resource-group ${RESOURCE_GROUP} \
  --min-replicas 0
```

#### Optimize for Performance

```bash
# Increase resources for production
az containerapp update \
  --name spiderweb-ml-app-prod \
  --resource-group ${RESOURCE_GROUP} \
  --cpu 2.0 \
  --memory 4.0Gi \
  --min-replicas 3 \
  --max-replicas 30
```

## Cost Estimation

### Azure Container Apps Pricing (East US)

| Component | Cost |
|-----------|------|
| Container Apps (1 vCPU, 2GB) | ~$0.000024/vCPU-second + $0.000003/GB-second |
| Container Registry (Basic) | $5/month |
| Log Analytics | $2.30/GB ingested |

**Estimated Monthly Cost**: $50-200 depending on usage

### Cost Optimization Tips

1. Use scale-to-zero for development environments
2. Enable auto-scaling to match demand
3. Set appropriate min/max replica counts
4. Use consumption-based pricing
5. Monitor and optimize using Azure Cost Management

## Security Best Practices

1. **Never commit API keys to source control**
2. **Use Azure Key Vault for secrets**:
   ```bash
   # Create Key Vault
   az keyvault create \
     --name spiderweb-ml-kv \
     --resource-group ${RESOURCE_GROUP}
   
   # Store secret
   az keyvault secret set \
     --vault-name spiderweb-ml-kv \
     --name openai-api-key \
     --value "sk-..."
   ```
3. **Enable managed identity** for the Container App
4. **Use private endpoints** for ACR and Key Vault
5. **Enable firewall rules** to restrict access
6. **Regular security updates** by rebuilding images

## Next Steps

- Set up [CI/CD with GitHub Actions](ci-cd-setup.md)
- Configure [custom domains and SSL](custom-domain.md)
- Enable [monitoring and alerting](monitoring.md)
- Review [production best practices](production-checklist.md)

## Support

For issues or questions:
- Check the [main documentation](README.md)
- Review [troubleshooting guide](#troubleshooting)
- Open an issue on the repository
- Contact Azure Support for Azure-specific issues

---

**Last Updated**: October 2025
