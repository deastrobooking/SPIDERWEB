#!/bin/bash
# Azure Deployment Script for SPIDERWEB ML-as-a-Service Platform

set -e

# Configuration
RESOURCE_GROUP="spiderweb-ml-rg"
LOCATION="eastus"
APP_NAME="spiderweb-ml"
ENVIRONMENT="prod"
ACR_NAME="${APP_NAME//[-]/_}acr"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}SPIDERWEB Azure Deployment Script${NC}"
echo -e "${GREEN}================================================${NC}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}Error: Azure CLI is not installed${NC}"
    echo "Please install from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if logged in to Azure
echo -e "\n${YELLOW}Checking Azure login status...${NC}"
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}Not logged in. Please login to Azure...${NC}"
    az login
fi

# Display current subscription
SUBSCRIPTION=$(az account show --query name -o tsv)
echo -e "${GREEN}Using subscription: ${SUBSCRIPTION}${NC}"

# Create resource group
echo -e "\n${YELLOW}Creating resource group: ${RESOURCE_GROUP}${NC}"
az group create \
    --name ${RESOURCE_GROUP} \
    --location ${LOCATION} \
    --tags environment=${ENVIRONMENT} project=spiderweb

# Create Container Registry
echo -e "\n${YELLOW}Creating Azure Container Registry: ${ACR_NAME}${NC}"
az acr create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${ACR_NAME} \
    --sku Basic \
    --admin-enabled true

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name ${ACR_NAME} --query loginServer -o tsv)
echo -e "${GREEN}ACR Login Server: ${ACR_LOGIN_SERVER}${NC}"

# Build and push Docker image
echo -e "\n${YELLOW}Building Docker image...${NC}"
az acr build \
    --registry ${ACR_NAME} \
    --image spiderweb-ml:latest \
    --image spiderweb-ml:$(date +%Y%m%d-%H%M%S) \
    --file Dockerfile \
    .

# Deploy using Bicep
echo -e "\n${YELLOW}Deploying infrastructure with Bicep...${NC}"
az deployment group create \
    --resource-group ${RESOURCE_GROUP} \
    --template-file bicep/main.bicep \
    --parameters \
        appName=${APP_NAME} \
        environment=${ENVIRONMENT} \
        containerRegistryName=${ACR_NAME} \
        containerImage="${ACR_LOGIN_SERVER}/spiderweb-ml:latest"

# Get the Container App FQDN
CONTAINER_APP_NAME="${APP_NAME}-app-${ENVIRONMENT}"
APP_URL=$(az containerapp show \
    --name ${CONTAINER_APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --query properties.configuration.ingress.fqdn -o tsv)

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Application URL: https://${APP_URL}${NC}"
echo -e "${GREEN}Health Check: https://${APP_URL}/health${NC}"
echo -e "${GREEN}API Status: https://${APP_URL}/v1/ai/status${NC}"
echo -e "\n${YELLOW}Important: Configure API keys as secrets:${NC}"
echo -e "az containerapp secret set \\"
echo -e "  --name ${CONTAINER_APP_NAME} \\"
echo -e "  --resource-group ${RESOURCE_GROUP} \\"
echo -e "  --secrets openai-api-key=<your-key> anthropic-api-key=<your-key>"
echo -e "${GREEN}================================================${NC}"
