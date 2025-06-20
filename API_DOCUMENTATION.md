# ML Training API Service Documentation

## Overview

The ML-as-a-Service platform provides comprehensive API endpoints for AI-enhanced model development, training, and optimization. The service integrates five major AI providers to deliver sophisticated machine learning capabilities through a unified interface.

## Base URL
```
http://localhost:5000
```

## Authentication

API keys are configured via environment variables. The service operates with graceful degradation - endpoints work with partial configuration.

```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
export PERPLEXITY_API_KEY=your_perplexity_key
export GEMINI_API_KEY=your_gemini_key
export XAI_API_KEY=your_grok_key
```

## Core API Endpoints

### Health Check
**GET** `/health`

Returns service health status and configuration.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-20T18:48:00Z",
  "version": "1.0.0",
  "services": {
    "ai_orchestrator": "operational",
    "database": "operational"
  }
}
```

### Service Status
**GET** `/v1/ai/status`

Returns detailed status of all AI service integrations.

**Response:**
```json
{
  "service_status": {
    "openai": {
      "status": "configured",
      "capabilities": ["synthetic_data", "optimization", "code_generation"]
    },
    "anthropic": {
      "status": "configured",
      "capabilities": ["reasoning", "analysis", "architecture_review"]
    },
    "perplexity": {
      "status": "not_configured",
      "capabilities": ["research", "benchmarking", "industry_insights"]
    },
    "gemini": {
      "status": "not_configured",
      "capabilities": ["multimodal", "code_optimization", "performance_analysis"]
    },
    "grok": {
      "status": "not_configured",
      "capabilities": ["creativity", "innovation", "architecture_design"]
    }
  },
  "overall_health": "partial_configuration",
  "configured_services": 2,
  "total_services": 5
}
```

## AI Enhancement Endpoints

### Synthetic Data Generation
**POST** `/v1/ai/synthetic-data`

Generate synthetic training data using OpenAI's advanced capabilities.

**Request Body:**
```json
{
  "model_description": "Binary classification model for fraud detection",
  "existing_data": "Transaction data with features: amount, merchant, time, location",
  "target_count": 10000,
  "data_format": "tabular",
  "quality_requirements": {
    "diversity": "high",
    "realism": "production_ready",
    "balance": "stratified"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "provider": "openai",
  "synthetic_data": {
    "generated_samples": 10000,
    "format": "structured",
    "quality_metrics": {
      "diversity_score": 0.92,
      "realism_score": 0.88,
      "balance_ratio": 0.85
    },
    "data_schema": {
      "features": ["amount", "merchant_category", "time_of_day", "location_risk"],
      "target": "is_fraudulent"
    }
  },
  "recommendations": [
    "Consider augmenting with adversarial examples for robustness",
    "Validate synthetic data distribution against production metrics"
  ],
  "generation_time_seconds": 45.2
}
```

### Model Analysis
**POST** `/v1/ai/analyze`

Perform comprehensive model analysis using Anthropic's reasoning capabilities.

**Request Body:**
```json
{
  "model_description": "Convolutional Neural Network for image classification",
  "architecture": {
    "layers": [
      {"type": "conv2d", "filters": 32, "kernel_size": 3},
      {"type": "maxpool", "pool_size": 2},
      {"type": "conv2d", "filters": 64, "kernel_size": 3},
      {"type": "dense", "units": 128},
      {"type": "output", "units": 10, "activation": "softmax"}
    ]
  },
  "data_sample": "CIFAR-10 dataset, 32x32 RGB images, 10 classes",
  "performance_metrics": {
    "accuracy": 0.847,
    "loss": 0.423,
    "training_time": "2.3 hours",
    "inference_latency": "12ms"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "provider": "anthropic",
  "analysis": {
    "architecture_assessment": {
      "strengths": [
        "Appropriate conv layer progression",
        "Reasonable filter sizes for 32x32 inputs",
        "Good balance between complexity and performance"
      ],
      "weaknesses": [
        "Missing batch normalization layers",
        "No dropout for regularization",
        "Could benefit from residual connections"
      ],
      "overall_score": 0.73
    },
    "performance_analysis": {
      "accuracy_assessment": "Good for baseline CNN, room for improvement",
      "convergence_quality": "Stable training curve indicated",
      "overfitting_risk": "Moderate - recommend validation monitoring"
    },
    "optimization_suggestions": [
      {
        "category": "architecture",
        "suggestion": "Add batch normalization after each conv layer",
        "expected_improvement": "5-8% accuracy gain"
      },
      {
        "category": "regularization",
        "suggestion": "Implement dropout (0.3-0.5) before dense layers",
        "expected_improvement": "Reduced overfitting, better generalization"
      },
      {
        "category": "data_augmentation",
        "suggestion": "Apply random horizontal flips and rotations",
        "expected_improvement": "3-5% accuracy improvement"
      }
    ]
  },
  "analysis_time_seconds": 28.7
}
```

### Comprehensive Model Enhancement
**POST** `/v1/ai/enhance`

Full model enhancement pipeline using multiple AI providers.

**Request Body:**
```json
{
  "model_description": "Transformer model for natural language processing",
  "training_data": "Text classification dataset, 50K samples, 5 categories",
  "performance_metrics": {
    "accuracy": 0.823,
    "f1_score": 0.801,
    "training_loss": 0.387,
    "validation_loss": 0.421
  },
  "config": {
    "optimization_focus": "accuracy",
    "performance_constraints": {
      "max_inference_latency": "100ms",
      "max_model_size": "500MB"
    },
    "enhancement_scope": ["architecture", "training", "data"]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "enhancement_summary": {
    "providers_used": ["openai", "anthropic", "perplexity"],
    "total_processing_time": 142.3,
    "confidence_score": 0.91
  },
  "enhancements": {
    "synthetic_data": {
      "provider": "openai",
      "generated_samples": 15000,
      "quality_score": 0.89,
      "augmentation_strategies": ["paraphrasing", "style_transfer", "domain_adaptation"]
    },
    "architecture_optimization": {
      "provider": "anthropic",
      "recommendations": [
        {
          "component": "attention_mechanism",
          "modification": "Multi-head attention with 8 heads",
          "rationale": "Improved feature representation for text classification"
        },
        {
          "component": "positional_encoding",
          "modification": "Learned positional embeddings",
          "rationale": "Better handling of variable-length sequences"
        }
      ],
      "expected_improvements": {
        "accuracy": "+0.047",
        "f1_score": "+0.052"
      }
    },
    "research_insights": {
      "provider": "perplexity",
      "latest_techniques": [
        "Layer-wise learning rate decay for transformer fine-tuning",
        "Gradient accumulation with adaptive batch sizing",
        "Curriculum learning for text classification tasks"
      ],
      "benchmarking_data": {
        "sota_accuracy": 0.934,
        "industry_average": 0.807,
        "improvement_potential": "High"
      }
    }
  },
  "implementation_roadmap": [
    {
      "phase": "data_enhancement",
      "timeline": "1-2 days",
      "tasks": ["Integrate synthetic data", "Implement augmentation pipeline"]
    },
    {
      "phase": "architecture_updates",
      "timeline": "3-5 days",
      "tasks": ["Modify attention mechanism", "Update positional encoding"]
    },
    {
      "phase": "training_optimization",
      "timeline": "2-3 days", 
      "tasks": ["Implement curriculum learning", "Optimize learning rate schedule"]
    }
  ]
}
```

## Model Management Endpoints

### List Models
**GET** `/v1/models`

Returns list of available models and their status.

**Response:**
```json
{
  "models": [
    {
      "id": "model_001",
      "name": "Fraud Detection CNN",
      "type": "classification",
      "status": "training",
      "accuracy": 0.847,
      "created_at": "2025-06-20T10:30:00Z"
    },
    {
      "id": "model_002", 
      "name": "Text Classifier Transformer",
      "type": "nlp_classification",
      "status": "completed",
      "accuracy": 0.823,
      "created_at": "2025-06-19T14:15:00Z"
    }
  ],
  "total_models": 2,
  "active_training": 1
}
```

### Start Training
**POST** `/v1/models/train`

Initiate AI-enhanced model training with multi-provider optimization.

**Request Body:**
```json
{
  "model_config": {
    "name": "Enhanced Image Classifier",
    "type": "computer_vision",
    "architecture": "resnet50",
    "dataset": "custom_images",
    "target_accuracy": 0.92
  },
  "training_config": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "ai_enhancement": true,
    "providers": ["openai", "anthropic", "perplexity"]
  }
}
```

**Response:**
```json
{
  "status": "training_started",
  "model_id": "model_003",
  "training_job_id": "job_789",
  "estimated_completion": "2025-06-21T08:30:00Z",
  "ai_enhancements_scheduled": [
    "synthetic_data_generation",
    "architecture_optimization", 
    "hyperparameter_tuning"
  ],
  "monitoring_url": "/v1/models/model_003/status"
}
```

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Model description is required",
    "details": {
      "field": "model_description",
      "received": null,
      "expected": "string"
    }
  },
  "timestamp": "2025-06-20T18:48:00Z",
  "request_id": "req_12345"
}
```

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `MISSING_API_KEY` | Required AI service not configured | Set appropriate environment variable |
| `RATE_LIMIT_EXCEEDED` | API rate limit reached | Implement exponential backoff |
| `VALIDATION_ERROR` | Invalid request parameters | Check request body format |
| `SERVICE_UNAVAILABLE` | AI provider temporarily unavailable | Retry with different provider |
| `PROCESSING_TIMEOUT` | Request processing timeout | Reduce request complexity |

## Rate Limits

| Endpoint | Rate Limit | Window |
|----------|------------|--------|
| `/health` | 100 requests | 1 minute |
| `/v1/ai/status` | 60 requests | 1 minute |
| `/v1/ai/synthetic-data` | 10 requests | 1 minute |
| `/v1/ai/analyze` | 20 requests | 1 minute |
| `/v1/ai/enhance` | 5 requests | 1 minute |

## SDK Examples

### Python SDK
```python
import requests
import json

class MLTrainingAPI:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def get_service_status(self):
        response = requests.get(f"{self.base_url}/v1/ai/status")
        return response.json()
    
    def generate_synthetic_data(self, model_desc, existing_data, target_count):
        payload = {
            "model_description": model_desc,
            "existing_data": existing_data,
            "target_count": target_count
        }
        response = requests.post(
            f"{self.base_url}/v1/ai/synthetic-data",
            json=payload
        )
        return response.json()
    
    def analyze_model(self, model_desc, data_sample, metrics):
        payload = {
            "model_description": model_desc,
            "data_sample": data_sample,
            "performance_metrics": metrics
        }
        response = requests.post(
            f"{self.base_url}/v1/ai/analyze",
            json=payload
        )
        return response.json()

# Usage
api = MLTrainingAPI()
status = api.get_service_status()
print(f"Services configured: {status['configured_services']}")
```

### cURL Examples
```bash
# Check all services status
curl -X GET http://localhost:5000/v1/ai/status

# Generate synthetic data
curl -X POST http://localhost:5000/v1/ai/synthetic-data \
  -H "Content-Type: application/json" \
  -d '{
    "model_description": "Binary classifier for email spam detection",
    "existing_data": "Email dataset with subject, body, sender features",
    "target_count": 5000
  }'

# Analyze model performance
curl -X POST http://localhost:5000/v1/ai/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "model_description": "LSTM for time series forecasting",
    "data_sample": "Stock price data, 5-year history",
    "performance_metrics": {"mse": 0.023, "mae": 0.14}
  }'
```

## Best Practices

### Request Optimization
- Include detailed model descriptions for better AI analysis
- Provide representative data samples for accurate recommendations
- Use appropriate target counts for synthetic data generation
- Specify performance constraints clearly

### Error Recovery
- Implement exponential backoff for rate-limited requests
- Handle partial service availability gracefully
- Cache successful responses when appropriate
- Monitor service status before making requests

### Performance Monitoring
- Track response times for optimization
- Monitor AI service usage and costs
- Implement request timeouts (30-60 seconds recommended)
- Log errors for debugging and improvement

The ML Training API Service provides unprecedented capabilities for AI-enhanced model development, combining multiple AI providers into a unified, production-ready platform for advanced machine learning workflows.