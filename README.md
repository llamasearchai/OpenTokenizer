# MLTokenizer

<p align="center">
  <img src="docs/assets/logo.png" alt="MLTokenizer Logo" width="200"/>
</p>

MLTokenizer is a comprehensive tokenization system for machine learning models, with a focus on performance, cross-model compatibility, and extensibility.

## Features

- **Multiple Tokenization Algorithms**: BPE, WordPiece, Unigram, SentencePiece, and Character implementations
- **Model Compatibility**: Pre-built adapters for popular models (BERT, GPT, T5, LLaMA)
- **Performance Optimization**: Rust-powered performance-critical components
- **Visualization**: Interactive Tokenization Laboratory with Tauri
- **Extensibility**: Modular architecture for easy addition of new tokenizers
- **Cross-language**: Python API with optional Rust components
- **Production-ready**: Comprehensive testing, robust error handling, and API services

## Installation

### Using pip

```bash
pip install mltokenizer
```

### From Source

```bash
# Clone the repository
git clone https://github.com/mltokenizer/mltokenizer.git
cd mltokenizer

# Install with pip
pip install -e .

# Build Rust extensions (optional, requires Rust toolchain)
pip install maturin
maturin develop --release
```

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build and run individual services
docker build -t mltokenizer .
docker run -p 8000:8000 mltokenizer
```

## Quick Start

### Basic Usage

```python
from mltokenizer.algorithms.bpe import BPETokenizer
from mltokenizer.normalization.normalizers import SequenceNormalizer

# Create a BPE tokenizer
tokenizer = BPETokenizer(
    vocab_size=10000,
    normalizer=SequenceNormalizer.default()
)

# Train the tokenizer
with open("training_data.txt", "r") as f:
    texts = [line.strip() for line in f]
tokenizer.train(texts=texts)

# Tokenize a text
result = tokenizer.encode("Hello, world!", return_tokens=True)
print(result.tokens)  # List of tokens
print(result.input_ids)  # List of token IDs

# Decode tokens back to text
decoded = tokenizer.decode(result.input_ids)
print(decoded)  # "Hello, world!"

# Save the tokenizer
tokenizer.save("./my_tokenizer")

# Load the tokenizer
loaded_tokenizer = BPETokenizer.load("./my_tokenizer")
```

### Using Pre-trained Models

```python
from mltokenizer.models.bert import BertTokenizer
from mltokenizer.models.gpt import GPTTokenizer

# Load BERT tokenizer
bert_tokenizer = BertTokenizer.from_huggingface("bert-base-uncased")

# Load GPT tokenizer
gpt_tokenizer = GPTTokenizer.from_huggingface("gpt2")

# Tokenize with both and compare
bert_result = bert_tokenizer.encode("Hello, world!", return_tokens=True)
gpt_result = gpt_tokenizer.encode("Hello, world!", return_tokens=True)

print(f"BERT tokens: {bert_result.tokens}")
print(f"GPT tokens: {gpt_result.tokens}")
```

### Command Line Interface

```bash
# Train a tokenizer
tokenize train-tokenizer \
    --tokenizer-type bpe \
    --input-file data/train.txt \
    --output-dir ./my_tokenizer \
    --vocab-size 20000

# Load a pretrained tokenizer
tokenize load-pretrained --model-name bert-base-uncased

# Tokenize text
tokenize tokenize "Hello, world!" --tokenizer-id bert_base_uncased

# Analyze token distribution
tokenize analyze-distribution \
    --input-file data/corpus.txt \
    --tokenizer-id bert_base_uncased \
    --output-file analysis.json

# Start the tokenization server
tokenize serve --port 8000
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────────────────┐   │
│  │ Tokenization│    │  Encoding   │    │    Normalization      │   │
│  │    Core     │───▶│  Management │───▶│        Engine         │   │
│  └─────────────┘    └─────────────┘    └───────────────────────┘   │
│         │                  │                      │                 │
│         ▼                  ▼                      ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────────────────┐   │
│  │  Algorithm  │    │  Vocabulary │    │    Pre/Post-Process   │   │
│  │   Registry  │    │  Management │    │        Pipeline       │   │
│  └─────────────┘    └─────────────┘    └───────────────────────┘   │
│                                                    │                │
│                                                    ▼                │
│  ┌─────────────────────────────┐    ┌───────────────────────────┐  │
│  │   Performance Monitoring    │◀───│     Orchestration Hub     │  │
│  └─────────────────────────────┘    └───────────────────────────┘  │
│                  │                             ▲                    │
│                  ▼                             │                    │
│  ┌─────────────────────────────┐    ┌───────────────────────────┐  │
│  │       Model Adapters        │    │   Tokenization Laboratory │  │
│  └─────────────────────────────┘    └───────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

The MLTokenizer system is built on several key components:

1. **Tokenization Core**: Base classes and interfaces for all tokenizers
2. **Algorithm Registry**: Implementation of various tokenization algorithms
3. **Encoding Management**: Converts between tokens and token IDs
4. **Normalization Engine**: Text normalization components
5. **Preprocessing Pipeline**: Text preprocessing components
6. **Performance Monitoring**: Metrics collection and optimization
7. **Model Adapters**: Integration with popular ML models
8. **Tokenization Laboratory**: Interactive UI for exploring tokenization

## Tokenization Laboratory

The Tokenization Laboratory provides an interactive interface for experimenting with different tokenizers, visualizing token distributions, and analyzing tokenization efficiency.

```bash
# Start the tokenization lab 
cd tokenization_lab
npm run tauri dev
```

<p align="center">
  <img src="docs/assets/tokenization_lab_screenshot.png" alt="Tokenization Lab" width="800"/>
</p>

## API Documentation

The API documentation is available at `/docs` when the server is running:

```bash
tokenize serve
# Then navigate to http://localhost:8000/docs
```

For detailed API documentation, see our [API Reference](docs/api/tokenizers.md).

## Extending the System

### Adding a New Tokenization Algorithm

1. Create a new class that inherits from `BaseTokenizer`
2. Implement the required methods: `train()`, `encode()`, `decode()`, `save()`, and `load()`
3. Register the tokenizer with the registry

### Adding a New Model Adapter

1. Create a new module in the `models` directory
2. Implement adapter methods for loading and configuring the tokenizer
3. Add factory methods for different initialization strategies

## Performance Optimization

For performance-critical applications, the system offers several optimization options:

1. **Rust Extensions**: Critical paths are implemented in Rust for maximum performance
2. **Caching**: Token merges and encodings are cached to avoid redundant work
3. **Batching**: Use batch methods for processing multiple texts
4. **Parallelization**: Training and batch processing leverage multiple cores

## Cross-Model Compatibility

The system includes tools for analyzing and enhancing cross-model compatibility:

```python
from mltokenizer.utils.compatibility import compare_tokenization, transfer_tokenization

# Compare tokenization between models
comparison = compare_tokenization(
    text="This is a test",
    tokenizer_a=bert_tokenizer,
    tokenizer_b=gpt_tokenizer
)

# Transfer tokenization from one model to another
transferred = transfer_tokenization(
    tokens=bert_result.tokens,
    source_tokenizer=bert_tokenizer,
    target_tokenizer=gpt_tokenizer
)
```

## Testing

Comprehensive tests are included for all components:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/python/unit
pytest tests/python/integration
pytest tests/python/performance
pytest tests/python/multilingual

# Run with coverage report
pytest --cov=mltokenizer
```

## License

MIT

## Citation

If you use MLTokenizer in your research, please cite:

```bibtex
@software{mltokenizer2023,
  author = {MLTokenizer Team},
  title = {MLTokenizer: A Comprehensive Tokenization System},
  year = {2023},
  url = {https://github.com/mltokenizer/mltokenizer}
}
```

## Contributing

Contributions are welcome! Please check the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

# OpenTokenizer: ML Tokenization System

A comprehensive ML Tokenization System migrated from the older `OpenTokenizer` structure. Includes a Python library, API, interactive lab (Tauri), CLI, and robust testing.

## Features
- **Python Library**: Tokenization algorithms, preprocessing, normalization, and performance utilities.
- **Rust Integration**: High-performance components for critical paths.
- **Interactive Lab**: Tauri-based app for experimenting with tokenizers.
- **Comprehensive Testing**: Unit, integration, and performance tests.

## Setup
```bash
# Clone the repository
git clone https://github.com/llamasearchai/OpenTokenizer.git
cd OpenTokenizer

# Install dependencies
pip install -e .

# Build Rust components
cd mltokenizer-rs
cargo build --release
```

## Usage
```python
from mltokenizer import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.train(["sample text"])
tokens = tokenizer.encode("sample text")
print(tokens)
```

## Logo
![OpenTokenizer Logo](opentokenizer.svg)

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. 