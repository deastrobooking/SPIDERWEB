// Azure Bicep template for SPIDERWEB ML-as-a-Service Platform
// Deploy with: az deployment group create --resource-group <rg> --template-file bicep/main.bicep

@description('Name of the application')
param appName string = 'spiderweb-ml'

@description('Azure region for resources')
param location string = resourceGroup().location

@description('Environment name (dev, staging, prod)')
@allowed([
  'dev'
  'staging'
  'prod'
])
param environment string = 'dev'

@description('Container Registry name')
param containerRegistryName string = '${replace(appName, '-', '')}acr'

@description('Log Analytics Workspace name')
param logAnalyticsWorkspaceName string = '${appName}-logs-${environment}'

@description('Container Apps Environment name')
param containerAppsEnvironmentName string = '${appName}-env-${environment}'

@description('Container App name')
param containerAppName string = '${appName}-app-${environment}'

@description('Container image and tag')
param containerImage string = '${containerRegistryName}.azurecr.io/spiderweb-ml:latest'

@description('Minimum number of replicas')
@minValue(0)
@maxValue(25)
param minReplicas int = 1

@description('Maximum number of replicas')
@minValue(1)
@maxValue(25)
param maxReplicas int = 10

@description('CPU cores (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)')
param cpuCores string = '1.0'

@description('Memory in Gi (0.5Gi, 1.0Gi, 1.5Gi, 2.0Gi, 2.5Gi, 3.0Gi, 3.5Gi, 4.0Gi)')
param memorySize string = '2Gi'

// Container Registry
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: containerRegistryName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// Log Analytics Workspace
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsWorkspaceName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// Container Apps Environment
resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerAppsEnvironmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// Container App
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: containerAppName
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 5000
        transport: 'http'
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          username: containerRegistry.listCredentials().username
          passwordSecretRef: 'registry-password'
        }
      ]
      secrets: [
        {
          name: 'registry-password'
          value: containerRegistry.listCredentials().passwords[0].value
        }
        {
          name: 'openai-api-key'
          value: '' // Set via Key Vault or CLI
        }
        {
          name: 'anthropic-api-key'
          value: '' // Set via Key Vault or CLI
        }
        {
          name: 'perplexity-api-key'
          value: '' // Set via Key Vault or CLI
        }
        {
          name: 'gemini-api-key'
          value: '' // Set via Key Vault or CLI
        }
        {
          name: 'xai-api-key'
          value: '' // Set via Key Vault or CLI
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'spiderweb-ml'
          image: containerImage
          env: [
            {
              name: 'OPENAI_API_KEY'
              secretRef: 'openai-api-key'
            }
            {
              name: 'ANTHROPIC_API_KEY'
              secretRef: 'anthropic-api-key'
            }
            {
              name: 'PERPLEXITY_API_KEY'
              secretRef: 'perplexity-api-key'
            }
            {
              name: 'GEMINI_API_KEY'
              secretRef: 'gemini-api-key'
            }
            {
              name: 'XAI_API_KEY'
              secretRef: 'xai-api-key'
            }
            {
              name: 'FLASK_ENV'
              value: 'production'
            }
            {
              name: 'PYTHONUNBUFFERED'
              value: '1'
            }
          ]
          resources: {
            cpu: json(cpuCores)
            memory: memorySize
          }
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 5000
              }
              initialDelaySeconds: 10
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: 5000
              }
              initialDelaySeconds: 5
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '50'
              }
            }
          }
        ]
      }
    }
  }
}

// Outputs
output containerRegistryLoginServer string = containerRegistry.properties.loginServer
output containerAppFQDN string = containerApp.properties.configuration.ingress.fqdn
output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
