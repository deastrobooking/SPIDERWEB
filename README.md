[![Deploy to Azure Container Apps](https://github.com/deastrobooking/SPIDERWEB/actions/workflows/azure-deploy.yml/badge.svg)](https://github.com/deastrobooking/SPIDERWEB/actions/workflows/azure-deploy.yml)

# SPIDERWEB: ML-as-a-Service Platform

A production-ready, hybrid machine learning platform combining native Rust ML framework capabilities with multi-provider AI service orchestration. SPIDERWEB provides a unified API for model training, enhancement, and deployment with seamless integration across OpenAI, Anthropic, Perplexity, Gemini, and Grok.

## 🚀 Overview

SPIDERWEB is designed for enterprise-grade ML workloads that require both high-performance native computation and advanced AI enhancement capabilities. The platform operates in three modes:

- **Full Mode**: All AI services configured for complete enhancement pipeline
- **Partial Mode**: Graceful degradation with available services
- **Demo Mode**: Functional demonstration with mock responses

## ✨ Key Features

### AI Service Orchestration
- **Multi-Provider Integration**: OpenAI, Anthropic, Perplexity, Gemini, and Grok
- **Intelligent Routing**: Automatic service selection based on task requirements
- **Fallback Handling**: Graceful degradation when services are unavailable
- **Real-time Research**: Integration with Perplexity for current information

### Native Rust ML Framework
- **Tensor Operations**: High-performance numerical computing with ndarray
- **Neural Networks**: Complete implementation of layers (Linear, Conv, RNN, Transformer)
- **Optimizers**: SGD, Adam, AdamW, RMSprop, AdaGrad
- **Autograd**: Automatic differentiation for gradient computation
- **Distributed Training**: Multi-node training support

### API Endpoints
- **Health Monitoring**: `/health` - Service health checks
- **Service Status**: `/v1/ai/status` - AI provider availability
- **Synthetic Data**: `/v1/ai/synthetic-data` - Generate training datasets
- **Model Analysis**: `/v1/ai/analyze` - Deep model interpretation
- **Enhancement Pipeline**: `/v1/ai/enhance` - Complete model optimization

### Web Dashboard
Interactive HTML/JavaScript dashboard for:
- Real-time service status monitoring
- API endpoint testing
- Model enhancement workflows
- Synthetic data generation

## 📋 Prerequisites

- **Python 3.11+** (for demo server)
- **Rust 1.70+** (for native ML framework)
- **AI Service API Keys** (optional, operates in demo mode without them):
  - OpenAI API Key
  - Anthropic API Key
  - Perplexity API Key
  - Google Gemini API Key
  - xAI Grok API Key

## 🔧 Quick Start

### 1. Configure Environment Variables

```bash
# Required for full functionality (optional for demo mode)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export PERPLEXITY_API_KEY="pplx-..."
export GEMINI_API_KEY="..."
export XAI_API_KEY="xai-..."
```

### 2. Install Python Dependencies

```bash
# Using uv (recommended)
uv pip install -r pyproject.toml

# Or using pip
pip install anthropic flask openai requests
```

### 3. Start the Demo Server

```bash
python3 demo_server.py
```

The server will start on `http://localhost:5000`

### 4. Access the Dashboard

Open your browser to `http://localhost:5000` to access the interactive web dashboard.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Dashboard (HTML/JS)                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              Flask API Server (demo_server.py)               │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │  OpenAI  │Anthropic │Perplexity│  Gemini  │   Grok   │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│           Native Rust ML Framework (src/)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tensor Ops │ Neural Nets │ Optimizers │ Autograd  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📚 API Documentation

### Health Check
```bash
GET /health
```

### Service Status
```bash
GET /v1/ai/status
```

Response:
```json
{
  "services": {
    "openai": {"available": true, "configured": true},
    "anthropic": {"available": true, "configured": true},
    "perplexity": {"available": true, "configured": true},
    "gemini": {"available": false, "configured": false},
    "grok": {"available": false, "configured": false}
  },
  "mode": "partial",
  "capabilities": ["synthetic_data", "analysis", "research"]
}
```

### Generate Synthetic Data
```bash
POST /v1/ai/synthetic-data
Content-Type: application/json

{
  "dataset_type": "classification",
  "num_samples": 100,
  "features": ["age", "income", "credit_score"],
  "target": "loan_approved"
}
```

### Analyze Model
```bash
POST /v1/ai/analyze
Content-Type: application/json

{
  "model_type": "neural_network",
  "architecture": "3-layer MLP",
  "performance_metrics": {"accuracy": 0.85, "loss": 0.15}
}
```

### Enhancement Pipeline
```bash
POST /v1/ai/enhance
Content-Type: application/json

{
  "model_description": "Image classification CNN",
  "current_performance": {"accuracy": 0.78},
  "enhancement_goals": ["improve accuracy", "reduce overfitting"]
}
```

## 📁 Project Structure

```
SPIDERWEB/
├── src/                          # Rust ML framework source
│   ├── lib.rs                    # Main library entry point
│   ├── tensor.rs                 # Tensor operations
│   ├── nn/                       # Neural network layers
│   │   ├── linear.rs             # Fully connected layers
│   │   ├── conv.rs               # Convolutional layers
│   │   ├── transformer.rs        # Transformer architecture
│   │   └── ...
│   ├── optim/                    # Optimizers
│   │   ├── adam.rs               # Adam optimizer
│   │   ├── sgd.rs                # SGD optimizer
│   │   └── ...
│   ├── ai_services/              # External AI integrations
│   │   ├── openai.rs             # OpenAI integration
│   │   ├── anthropic.rs          # Anthropic integration
│   │   ├── orchestrator.rs       # Service orchestration
│   │   └── ...
│   └── api/                      # REST API handlers
├── demo_server.py                # Python Flask demo server
├── examples/                     # Example scripts
│   ├── ai_service_demo.py        # AI service integration demo
│   └── ...
├── docs/                         # Comprehensive documentation
│   ├── README.md                 # Documentation index
│   ├── getting-started.md        # Setup guide
│   ├── api-reference.md          # API documentation
│   └── ...
├── Cargo.toml                    # Rust dependencies
├── pyproject.toml                # Python dependencies
└── README.md                     # This file
```

## 🔐 Security

- API keys are loaded from environment variables only
- No hardcoded credentials in source code
- Graceful degradation when services are unavailable
- Input validation on all API endpoints
- Rate limiting support (configure as needed)

## 🚀 Deployment

### Docker
```bash
# Build Docker image
docker build -t spiderweb-ml .

# Run container
docker run -p 5000:5000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  spiderweb-ml
```

### Azure
See [docs/deployment-azure.md](docs/deployment-azure.md) for detailed Azure deployment instructions.

## 🧪 Development

### Running Tests
```bash
# Python tests
python3 test_ai_services.py

# Rust tests
cargo test
```

### Building Native Library
```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release
```

## 📊 Performance

- **Native Rust Operations**: 10-100x faster than Python for numerical computing
- **Parallel Processing**: Automatic multi-core utilization with Rayon
- **Memory Efficient**: Zero-copy operations where possible
- **GPU Ready**: CUDA/Metal backend support (optional)

## 🛣️ Roadmap

- [ ] Complete CUDA/Metal GPU acceleration
- [ ] WebAssembly compilation for browser deployment
- [ ] Kubernetes deployment manifests
- [ ] Advanced model compression techniques
- [ ] Federated learning support
- [ ] Extended LLM fine-tuning capabilities

## 📖 Documentation

Full documentation is available in the [docs/](docs/) directory:

- [Getting Started Guide](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Core Concepts](docs/core-concepts.md)
- [Framework Summary](docs/framework-summary.md)
- [AI Services Summary](docs/ai-services-summary.md)
- [Contributing Guide](docs/contributing.md)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/contributing.md) for guidelines.

## 📄 License

This project is available under the terms specified in the project documentation.

## 🙋 Support

For questions, issues, or feature requests, please refer to the documentation in the `docs/` directory or open an issue on the repository.

---

**Built with ❤️ using Rust and Python**
