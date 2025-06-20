# Getting Started with ML Training API Service

## Quick Setup

### 1. Start the Service
```bash
python3 demo_server.py
```

The service starts on `http://localhost:5000` with a web dashboard for testing.

### 2. Verify Installation
Open your browser to `http://localhost:5000` or test via command line:
```bash
curl http://localhost:5000/health
```

You should see a health status response indicating the service is running.

### 3. Check Service Capabilities
```bash
curl http://localhost:5000/v1/ai/status
```

This shows which AI services are configured and available.

## Basic Usage Examples

### Generate Synthetic Training Data
```bash
curl -X POST http://localhost:5000/v1/ai/synthetic-data \
  -H "Content-Type: application/json" \
  -d '{
    "model_description": "Classification model for customer segmentation",
    "existing_data": "Customer demographics and purchase history",
    "target_count": 1000
  }'
```

### Analyze Model Performance
```bash
curl -X POST http://localhost:5000/v1/ai/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "model_description": "Neural network for image classification",
    "data_sample": "Training dataset with 10,000 images",
    "performance_metrics": {"accuracy": 0.85, "loss": 0.42}
  }'
```

### Comprehensive Model Enhancement
```bash
curl -X POST http://localhost:5000/v1/ai/enhance \
  -H "Content-Type: application/json" \
  -d '{
    "model_description": "Text classification transformer",
    "training_data": "News articles dataset",
    "performance_metrics": {"accuracy": 0.78, "f1_score": 0.76},
    "config": {"optimization_focus": "accuracy"}
  }'
```

## Enhanced Functionality with AI Services

### Configure AI Service Keys
For full functionality, set environment variables:
```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
```

Additional services (optional):
```bash
export PERPLEXITY_API_KEY=your_perplexity_key
export GEMINI_API_KEY=your_gemini_key
export XAI_API_KEY=your_grok_key
```

### Service Capabilities by Provider
- **OpenAI**: Synthetic data generation, optimization strategies
- **Anthropic**: Advanced reasoning, architectural analysis
- **Perplexity**: Research integration, benchmarking
- **Gemini**: Multimodal analysis, code optimization
- **Grok**: Creative problem-solving, innovative designs

## Python Integration

### Using Requests Library
```python
import requests
import json

# Check service status
response = requests.get('http://localhost:5000/v1/ai/status')
status = response.json()
print(f"Configured services: {status['configured_services']}")

# Generate synthetic data
data_request = {
    "model_description": "Fraud detection model",
    "existing_data": "Transaction records with features",
    "target_count": 5000
}

response = requests.post(
    'http://localhost:5000/v1/ai/synthetic-data',
    json=data_request
)

if response.status_code == 200:
    result = response.json()
    print(f"Generated {result['synthetic_data']['generated_samples']} samples")
else:
    print(f"Error: {response.json()}")
```

### Simple SDK Example
```python
class MLTrainingAPI:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def get_service_status(self):
        response = requests.get(f"{self.base_url}/v1/ai/status")
        return response.json()
    
    def enhance_model(self, model_desc, training_data, metrics, config=None):
        payload = {
            "model_description": model_desc,
            "training_data": training_data,
            "performance_metrics": metrics,
            "config": config or {}
        }
        response = requests.post(f"{self.base_url}/v1/ai/enhance", json=payload)
        return response.json()

# Usage
api = MLTrainingAPI()
result = api.enhance_model(
    model_desc="Image classifier CNN",
    training_data="CIFAR-10 dataset",
    metrics={"accuracy": 0.82, "loss": 0.45}
)
```

## Web Dashboard Features

### Interactive Testing
Navigate to `http://localhost:5000` for:
- Service status monitoring
- Interactive API endpoint testing
- Real-time response viewing
- Configuration guidance

### API Explorer
The dashboard provides:
- Pre-filled example requests
- Response format documentation
- Error handling examples
- Performance metrics

## Common Use Cases

### 1. Data Augmentation
Use synthetic data generation to expand training datasets:
```json
{
  "model_description": "Sentiment analysis model",
  "existing_data": "Product reviews dataset, 5000 samples",
  "target_count": 15000,
  "data_format": "text",
  "quality_requirements": {
    "diversity": "high",
    "realism": "production_ready"
  }
}
```

### 2. Model Optimization
Get AI-powered recommendations for model improvement:
```json
{
  "model_description": "Convolutional neural network for medical imaging",
  "architecture": {
    "layers": ["conv2d", "maxpool", "conv2d", "dense"],
    "parameters": "2.3M"
  },
  "performance_metrics": {
    "accuracy": 0.89,
    "precision": 0.87,
    "recall": 0.91
  }
}
```

### 3. Research Integration
Leverage real-time research insights:
```json
{
  "model_description": "Transformer for language understanding",
  "domain": "natural_language_processing",
  "config": {
    "optimization_focus": "performance",
    "research_integration": true
  }
}
```

## Error Handling

### Common Response Patterns
```python
def handle_api_response(response):
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 400:
        error = response.json()
        print(f"Validation error: {error['error']['message']}")
    elif response.status_code == 429:
        print("Rate limit exceeded. Please retry later.")
    elif response.status_code == 503:
        print("Service temporarily unavailable.")
    else:
        print(f"Unexpected error: {response.status_code}")
    
    return None
```

### Retry Logic
```python
import time
import random

def api_call_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = func()
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Exponential backoff for rate limits
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                break
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)
    
    return None
```

## Performance Tips

### Optimize Request Size
- Keep model descriptions concise but informative
- Limit data samples to representative examples
- Use appropriate target counts for synthetic data

### Batch Processing
```python
def process_multiple_models(models):
    results = []
    for model in models:
        result = api.analyze_model(
            model['description'],
            model['data_sample'],
            model['metrics']
        )
        results.append(result)
        time.sleep(0.1)  # Respect rate limits
    return results
```

### Caching Strategy
```python
import functools
import time

@functools.lru_cache(maxsize=100)
def cached_service_status():
    response = requests.get('http://localhost:5000/v1/ai/status')
    return response.json()

# Cache expires after 5 minutes
def get_cached_status():
    return cached_service_status()
```

## Next Steps

### Production Deployment
- Review [Deployment Guide](DEPLOYMENT_GUIDE.md) for production setup
- Configure proper authentication and rate limiting
- Set up monitoring and logging

### Advanced Features
- Explore [API Documentation](API_DOCUMENTATION.md) for complete endpoint reference
- Review [Architecture Overview](ARCHITECTURE_OVERVIEW.md) for system design
- Check [Advanced ML Architecture](../ADVANCED_ML_ARCHITECTURE.md) for framework development

### Contributing
- Study the codebase structure and patterns
- Run the test suite: `python test_ai_services.py`
- Review contribution guidelines in project documentation

The ML Training API Service provides a powerful foundation for AI-enhanced model development with comprehensive documentation and examples to get you started quickly.