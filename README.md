# SPIDERWEB: ML-as-a-Service Platform

[![Deploy to Azure Container Apps](https://github.com/deastrobooking/SPIDERWEB/actions/workflows/azure-deploy.yml/badge.svg)](https://github.com/deastrobooking/SPIDERWEB/actions/workflows/azure-deploy.yml)

A production-ready, hybrid machine learning platform combining native Rust ML framework capabilities with multi-provider AI service orchestration. SPIDERWEB provides a unified API for model training, enhancement, and deployment with seamless integration across OpenAI, Anthropic, Perplexity, Gemini, and Grok.

## 🚀 Quick Start

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

## 🌟 Key Features

### Multi-Provider AI Integration
- **OpenAI**: Synthetic data generation, optimization strategies, embeddings
- **Anthropic**: Advanced reasoning, architectural analysis, debugging
- **Perplexity**: Research integration, benchmarking, industry insights
- **Gemini**: Multimodal analysis, code optimization, deployment planning
- **Grok**: Creative problem-solving, innovative architecture design

### Native Rust ML Framework (In Development)
- High-performance tensor operations with automatic differentiation
- Complete neural network layer implementations (Linear, Conv, RNN, Transformer)
- State-of-the-art optimizers (SGD, Adam, AdamW, RMSprop)
- GPU acceleration support (CUDA, Vulkan)
- PyTorch/TensorFlow feature parity with Rust safety guarantees

### Production-Ready API
- RESTful endpoints for all ML operations
- Comprehensive error handling and validation
- Rate limiting and authentication ready
- Interactive web dashboard for testing

## 📖 Documentation

### Getting Started
- **[Installation & Setup](docs/getting-started.md)** - Complete setup guide with examples
- **[API Documentation](docs/api-documentation.md)** - Detailed endpoint documentation
- **[Examples](docs/examples.md)** - Code samples and tutorials

### Architecture & Design
- **[System Architecture](docs/advanced-architecture.md)** - Technical design and components
- **[AI Services Integration](docs/ai-services-summary.md)** - Multi-provider orchestration
- **[Framework Summary](docs/framework-summary.md)** - Native Rust ML framework overview

### Development
- **[Contributing Guide](docs/contributing.md)** - How to contribute to the project
- **[Performance Guide](docs/performance.md)** - Optimization strategies
- **[Roadmap](docs/roadmap.md)** - Future development plans

## 📡 API Endpoints

### AI Enhancement Services
```
GET  /health                    # Health check
GET  /v1/ai/status             # AI service configuration status
POST /v1/ai/enhance            # Comprehensive model enhancement
POST /v1/ai/synthetic-data     # Generate synthetic training data
POST /v1/ai/analyze            # Advanced model analysis
```

### Model Training & Inference
```
POST /v1/models/train          # Start training job
GET  /v1/models                # List available models
GET  /v1/models/{id}/status    # Check training status
POST /v1/models/predict        # Run inference
```

## 💡 Usage Examples

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

**Built with ❤️ using Rust 🦀 and Python 🐍**