# ML Training API Service - Deployment Guide

## Overview

This guide covers deploying the ML-as-a-Service platform in various environments, from local development to production-scale deployments with full AI service integration.

## Prerequisites

### System Requirements
- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- 20GB disk space for models and dependencies
- Network access for AI service APIs

### Required Dependencies
```bash
pip install flask requests anthropic openai
```

### Optional Components
- Rust toolchain for native framework development
- GPU drivers for CUDA acceleration (future releases)
- Docker for containerized deployment

## Local Development Setup

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd ml-training-api

# Install dependencies
pip install -r requirements.txt

# Configure AI services (optional)
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key

# Start development server
python3 demo_server.py
```

### Verify Installation
```bash
# Check service health
curl http://localhost:5000/health

# Test AI service status
curl http://localhost:5000/v1/ai/status
```

## Production Deployment

### Environment Configuration

Create production environment file:
```bash
# production.env
FLASK_ENV=production
FLASK_DEBUG=False
HOST=0.0.0.0
PORT=5000

# AI Service Configuration
OPENAI_API_KEY=your_production_openai_key
ANTHROPIC_API_KEY=your_production_anthropic_key
PERPLEXITY_API_KEY=your_perplexity_key
GEMINI_API_KEY=your_gemini_key
XAI_API_KEY=your_grok_key

# Performance Tuning
MAX_WORKERS=4
REQUEST_TIMEOUT=300
RATE_LIMIT_ENABLED=true
```

### Production Server Setup

Using Gunicorn (recommended):
```bash
# Install production server
pip install gunicorn

# Start production server
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 300 demo_server:app
```

Using uWSGI:
```bash
# Install uWSGI
pip install uwsgi

# Start with uWSGI
uwsgi --http :5000 --module demo_server:app --processes 4 --threads 2
```

### Load Balancer Configuration

Nginx configuration example:
```nginx
upstream ml_api {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    listen 80;
    server_name ml-api.yourdomain.com;

    location / {
        proxy_pass http://ml_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    location /health {
        proxy_pass http://ml_api;
        access_log off;
    }
}
```

## Cloud Deployment Options

### Docker Containerization

Dockerfile:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "300", "demo_server:app"]
```

Build and run:
```bash
# Build image
docker build -t ml-training-api .

# Run container
docker run -d \
  --name ml-api \
  -p 5000:5000 \
  -e OPENAI_API_KEY=your_key \
  -e ANTHROPIC_API_KEY=your_key \
  ml-training-api
```

### Kubernetes Deployment

deployment.yaml:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-training-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-training-api
  template:
    metadata:
      labels:
        app: ml-training-api
    spec:
      containers:
      - name: ml-api
        image: ml-training-api:latest
        ports:
        - containerPort: 5000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-service-keys
              key: openai-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-service-keys
              key: anthropic-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-training-api-service
spec:
  selector:
    app: ml-training-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

### AWS Deployment

Using AWS ECS with Fargate:
```json
{
  "family": "ml-training-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "ml-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/ml-training-api:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "FLASK_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:ml-api-keys"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ml-training-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

## Security Configuration

### API Key Management
Store API keys securely using environment-specific solutions:
- Local: Environment variables
- Docker: Docker secrets
- Kubernetes: Kubernetes secrets
- AWS: AWS Secrets Manager or Parameter Store
- Azure: Azure Key Vault
- GCP: Google Secret Manager

### Rate Limiting
Implement rate limiting for production:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/v1/ai/enhance')
@limiter.limit("5 per minute")
def enhance_model():
    # Implementation
```

### HTTPS Configuration
Always use HTTPS in production:
```nginx
server {
    listen 443 ssl http2;
    server_name ml-api.yourdomain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://ml_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto https;
    }
}
```

## Monitoring and Logging

### Application Monitoring
```python
import logging
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('ml_api_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('ml_api_request_duration_seconds', 'Request latency')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler('ml_api.log'),
        logging.StreamHandler()
    ]
)
```

### Health Check Endpoint
Enhanced health check with dependency verification:
```python
@app.route('/health')
def health_check():
    checks = {
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'status': 'healthy'
    }
    
    # Check AI service connectivity
    try:
        ai_status = orchestrator.get_service_status()
        checks['ai_services'] = ai_status
    except Exception as e:
        checks['status'] = 'degraded'
        checks['ai_services'] = {'error': str(e)}
    
    status_code = 200 if checks['status'] == 'healthy' else 503
    return jsonify(checks), status_code
```

## Performance Optimization

### Caching Strategy
```python
from flask_caching import Cache

cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0'
})

@app.route('/v1/ai/status')
@cache.cached(timeout=300)  # Cache for 5 minutes
def get_ai_status():
    return orchestrator.get_service_status()
```

### Async Processing
For long-running tasks:
```python
from celery import Celery

celery = Celery(app.name, broker='redis://localhost:6379/0')

@celery.task
def process_model_enhancement(model_data):
    # Long-running AI enhancement task
    return orchestrator.comprehensive_enhancement(model_data)

@app.route('/v1/ai/enhance/async', methods=['POST'])
def enhance_model_async():
    task = process_model_enhancement.delay(request.json)
    return jsonify({'task_id': task.id, 'status': 'processing'})
```

## Scaling Considerations

### Horizontal Scaling
- Deploy multiple instances behind load balancer
- Use Redis for session storage and caching
- Implement database connection pooling
- Consider async task queues for heavy workloads

### Vertical Scaling
- Monitor memory usage during AI service calls
- Optimize Python memory usage with appropriate garbage collection
- Consider memory-mapped files for large model data

### Auto-scaling Configuration
AWS Auto Scaling Group example:
```json
{
  "AutoScalingGroupName": "ml-api-asg",
  "MinSize": 2,
  "MaxSize": 10,
  "DesiredCapacity": 3,
  "TargetGroupARNs": ["arn:aws:elasticloadbalancing:..."],
  "HealthCheckType": "ELB",
  "HealthCheckGracePeriod": 300
}
```

## Backup and Recovery

### Data Backup
- Regular backups of configuration and model metadata
- API key rotation procedures
- Log archival strategy

### Disaster Recovery
- Multi-region deployment for high availability
- Database replication and failover procedures
- AI service failover and degraded mode operation

## Cost Optimization

### AI Service Usage Monitoring
```python
class CostTracker:
    def __init__(self):
        self.usage_metrics = {}
    
    def track_api_call(self, provider, endpoint, tokens_used):
        key = f"{provider}_{endpoint}"
        if key not in self.usage_metrics:
            self.usage_metrics[key] = {'calls': 0, 'tokens': 0}
        
        self.usage_metrics[key]['calls'] += 1
        self.usage_metrics[key]['tokens'] += tokens_used
    
    def get_cost_estimate(self):
        # Calculate estimated costs based on provider pricing
        pass
```

### Optimization Strategies
- Cache frequent AI service responses
- Implement intelligent request batching
- Use cheaper models for simple tasks
- Monitor and alert on usage thresholds

This deployment guide provides comprehensive coverage for running the ML Training API Service in production environments with proper security, monitoring, and scaling considerations.