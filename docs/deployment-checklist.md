# Azure Deployment Checklist for SPIDERWEB

Complete this checklist before deploying SPIDERWEB to Azure.

## ✅ Pre-Deployment Checklist

### Azure Setup
- [ ] Azure subscription is active and accessible
- [ ] Azure CLI installed and logged in (`az login`)
- [ ] Resource group created or selected
- [ ] Sufficient quota for Container Apps/App Service
- [ ] Cost alerts configured (optional but recommended)

### API Keys
- [ ] OpenAI API key obtained and tested
- [ ] Anthropic API key obtained and tested
- [ ] Perplexity API key obtained (optional)
- [ ] Google Gemini API key obtained (optional)
- [ ] xAI Grok API key obtained (optional)
- [ ] Separate keys for production environment (recommended)

### Code Repository
- [ ] Code is committed to Git repository
- [ ] Repository is pushed to GitHub/Azure DevOps
- [ ] All sensitive data removed from code
- [ ] `.gitignore` configured properly
- [ ] README.md is up to date

### Docker
- [ ] Docker installed locally for testing
- [ ] Dockerfile tested locally (`docker build -t spiderweb-ml:test .`)
- [ ] Container runs successfully locally (`docker run -p 5000:5000 spiderweb-ml:test`)
- [ ] Health endpoint accessible (`curl http://localhost:5000/health`)

## 🚀 Deployment Checklist

### Option A: Automated Deployment Script

- [ ] Review `deploy-azure.sh` configuration
- [ ] Update resource names if needed
- [ ] Make script executable (`chmod +x deploy-azure.sh`)
- [ ] Run deployment script (`./deploy-azure.sh`)
- [ ] Note the deployment output (URLs, resource names)
- [ ] Configure API keys as secrets
- [ ] Verify health endpoint
- [ ] Test API endpoints

### Option B: Manual Deployment

#### 1. Resource Group
- [ ] Create resource group
  ```bash
  az group create --name spiderweb-ml-rg --location eastus
  ```

#### 2. Container Registry
- [ ] Create ACR
  ```bash
  az acr create --name spiderwebmlacr --resource-group spiderweb-ml-rg --sku Basic --admin-enabled true
  ```
- [ ] Note ACR login server
- [ ] Verify ACR credentials

#### 3. Build and Push Image
- [ ] Build Docker image
  ```bash
  az acr build --registry spiderwebmlacr --image spiderweb-ml:latest --file Dockerfile .
  ```
- [ ] Verify image in ACR
  ```bash
  az acr repository list --name spiderwebmlacr
  ```

#### 4. Deploy Infrastructure
- [ ] Review Bicep template (`bicep/main.bicep`)
- [ ] Deploy with Bicep
  ```bash
  az deployment group create --resource-group spiderweb-ml-rg --template-file bicep/main.bicep
  ```
- [ ] Verify deployment succeeded
- [ ] Note Container App URL

#### 5. Configure Secrets
- [ ] Set OpenAI API key
- [ ] Set Anthropic API key
- [ ] Set other API keys (optional)
- [ ] Restart container app
- [ ] Verify secrets are loaded

### Option C: CI/CD with GitHub Actions

#### 1. GitHub Setup
- [ ] Repository is on GitHub
- [ ] Create Azure service principal
  ```bash
  az ad sp create-for-rbac --name "github-actions-spiderweb" --role contributor --scopes /subscriptions/<id>/resourceGroups/spiderweb-ml-rg --sdk-auth
  ```
- [ ] Add `AZURE_CREDENTIALS` to GitHub Secrets
- [ ] Add all API key secrets to GitHub Secrets
- [ ] Create GitHub environments (dev, staging, prod)
- [ ] Configure environment protection rules

#### 2. Workflow Setup
- [ ] Review `.github/workflows/azure-deploy.yml`
- [ ] Update resource names if needed
- [ ] Verify workflow syntax
- [ ] Enable GitHub Actions for repository

#### 3. Initial Deployment
- [ ] Run `deploy-azure.sh` to create initial infrastructure
- [ ] Push code to trigger first workflow
- [ ] Monitor workflow execution
- [ ] Verify deployment in Azure

## 🔍 Post-Deployment Verification

### Health Checks
- [ ] Access application URL
- [ ] Verify health endpoint returns 200 OK
  ```bash
  curl https://your-app.azurecontainerapps.io/health
  ```
- [ ] Check service status endpoint
  ```bash
  curl https://your-app.azurecontainerapps.io/v1/ai/status
  ```

### API Testing
- [ ] Test synthetic data generation endpoint
  ```bash
  curl -X POST https://your-app.azurecontainerapps.io/v1/ai/synthetic-data \
    -H "Content-Type: application/json" \
    -d '{"dataset_type":"classification","num_samples":10}'
  ```
- [ ] Test model analysis endpoint
- [ ] Test enhancement pipeline
- [ ] Verify all configured AI services are working

### Web Dashboard
- [ ] Access dashboard at root URL
- [ ] Verify service status cards display correctly
- [ ] Test each button/feature in dashboard
- [ ] Check browser console for errors

### Logging and Monitoring
- [ ] View application logs
  ```bash
  az containerapp logs show --name spiderweb-ml-app-prod --resource-group spiderweb-ml-rg --follow
  ```
- [ ] Check for any errors or warnings
- [ ] Verify log analytics workspace is receiving data
- [ ] Set up alerts (optional)

### Performance
- [ ] Check container startup time
- [ ] Measure API response times
- [ ] Verify auto-scaling works (if configured)
- [ ] Monitor memory and CPU usage

## 🔐 Security Checklist

### Secrets Management
- [ ] All API keys stored as secrets (not in code)
- [ ] No secrets in environment variables visible in portal
- [ ] Consider using Azure Key Vault for production
- [ ] Rotate keys regularly (establish schedule)

### Network Security
- [ ] HTTPS enabled (automatic with Container Apps)
- [ ] Consider enabling firewall rules
- [ ] Review ingress settings
- [ ] Configure CORS if needed

### Access Control
- [ ] Review Azure RBAC roles
- [ ] Limit admin access
- [ ] Enable managed identity (recommended)
- [ ] Configure authentication (optional)

### Compliance
- [ ] Review data residency requirements
- [ ] Check API provider terms of service
- [ ] Document data handling procedures
- [ ] Review privacy policy

## 📊 Monitoring Setup

### Azure Monitor
- [ ] Enable Application Insights (optional)
- [ ] Configure custom metrics
- [ ] Set up availability tests
- [ ] Create alert rules

### Cost Management
- [ ] Enable cost alerts
- [ ] Set up budgets
- [ ] Review pricing calculator estimates
- [ ] Monitor daily spending

### Alerts
- [ ] Alert on failed health checks
- [ ] Alert on high error rates
- [ ] Alert on high latency
- [ ] Alert on cost thresholds
- [ ] Configure notification channels (email, SMS, Slack)

## 🔄 Maintenance Planning

### Regular Tasks
- [ ] Schedule regular security updates
- [ ] Plan for key rotation
- [ ] Review and update dependencies
- [ ] Monitor for CVEs
- [ ] Review and optimize costs

### Backup and Disaster Recovery
- [ ] Document rollback procedure
- [ ] Test rollback process
- [ ] Export configuration (Bicep/ARM templates)
- [ ] Document recovery procedures
- [ ] Set up geo-redundancy (if needed)

### Scaling Strategy
- [ ] Document expected traffic patterns
- [ ] Configure auto-scaling rules
- [ ] Plan for traffic spikes
- [ ] Test scaling behavior
- [ ] Monitor scaling metrics

## 📝 Documentation

### Deployment Documentation
- [ ] Document actual resource names used
- [ ] Note all URLs and endpoints
- [ ] Record configuration decisions
- [ ] Document any customizations made
- [ ] Update runbook with deployment details

### Team Knowledge
- [ ] Share deployment credentials (securely)
- [ ] Train team on deployment process
- [ ] Document troubleshooting steps
- [ ] Create incident response plan
- [ ] Establish on-call rotation (if applicable)

## ✨ Optional Enhancements

### Custom Domain
- [ ] Purchase/configure domain name
- [ ] Add custom domain to Container App
- [ ] Configure DNS records
- [ ] Enable managed SSL certificate
- [ ] Test domain resolution

### CDN Integration
- [ ] Set up Azure CDN (if needed)
- [ ] Configure caching rules
- [ ] Test CDN performance
- [ ] Configure purge rules

### Advanced Monitoring
- [ ] Set up distributed tracing
- [ ] Configure custom dashboards
- [ ] Enable query performance insights
- [ ] Set up synthetic monitoring

### Additional Services
- [ ] Consider Azure API Management
- [ ] Evaluate Azure Front Door
- [ ] Review Azure Cache for Redis
- [ ] Assess need for Azure Storage

## 🎉 Go-Live Checklist

Final checks before announcing availability:

- [ ] All critical features tested in production
- [ ] Performance meets requirements
- [ ] Security review completed
- [ ] Monitoring and alerts configured
- [ ] Documentation complete
- [ ] Team trained on operations
- [ ] Support procedures established
- [ ] Rollback plan tested
- [ ] Stakeholders notified
- [ ] Marketing materials ready (if applicable)

## 📞 Support and Resources

### Azure Support
- Documentation: https://docs.microsoft.com/azure
- Support Portal: https://portal.azure.com
- Community Forums: https://docs.microsoft.com/answers

### AI Provider Support
- OpenAI: https://help.openai.com
- Anthropic: https://support.anthropic.com
- Perplexity: https://docs.perplexity.ai
- Google Gemini: https://ai.google.dev/docs
- xAI: https://docs.x.ai

### Emergency Contacts
- [ ] Document Azure subscription admin contact
- [ ] Document DevOps lead contact
- [ ] Document on-call engineer contact
- [ ] Document escalation procedures

---

## Deployment Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Developer | | | |
| DevOps Lead | | | |
| Security Review | | | |
| Product Owner | | | |

**Deployment Date**: _________________

**Production URL**: _________________

**Notes**: 
_________________________________________
_________________________________________
_________________________________________

---

**Last Updated**: October 2025
