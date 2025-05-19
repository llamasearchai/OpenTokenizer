# MLTokenizer API Documentation

## Overview

MLTokenizer is a comprehensive tokenization system for machine learning models, designed to provide high-performance, versatile text tokenization for natural language processing tasks. This library offers several tokenization algorithms, normalization options, and preprocessing capabilities in a modular, extensible architecture.

## Core Architecture

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
8. **Orchestration Hub**: Central coordination of tokenization workflow

## Tokenization Algorithms

MLTokenizer supports several tokenization algorithms:

### BPE (Byte-Pair Encoding)

BPE tokenizer learns to merge frequent pairs of characters or subwords into new tokens.

```python
from mltokenizer.algorithms.bpe import BPETokenizer
from mltokenizer.normalization.normalizers import LowercaseNormalizer

# Create and train a BPE tokenizer
tokenizer = BPETokenizer(
    vocab_size=1000,
    normalizer=LowercaseNormalizer()
)

# Train on texts
tokenizer.train(texts=["This is a sample text", "Another example"])

# Encode text
encoded = tokenizer.encode("This is a test")
print(encoded.input_ids)  # List of token IDs

# Decode back to text
decoded = tokenizer.decode(encoded.input_ids)
print(decoded)  # "this is a test"
```

### WordPiece

WordPiece tokenizer splits words into subwords, using the `##` prefix to mark subword pieces.

```python
from mltokenizer.algorithms.wordpiece import WordpieceTokenizer
from mltokenizer.encoding.special_tokens import SpecialTokens

# Create BERT-style tokenizer
special_tokens = SpecialTokens(
    pad_token="[PAD]",
    unk_token="[UNK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

tokenizer = WordpieceTokenizer(
    vocab_size=30000,
    special_tokens=special_tokens,
    wordpiece_prefix="##"
)

# Train and use
tokenizer.train(texts=["Example text for WordPiece tokenization"])
result = tokenizer.encode("WordPiece splits words", return_tokens=True)
print(result.tokens)  # ["word", "##piece", "splits", "words"]
```

### Character Tokenizer

Character-level tokenizer splits text into individual characters.

```python
from mltokenizer.algorithms.character import CharacterTokenizer

# Create and train a character tokenizer
tokenizer = CharacterTokenizer()
tokenizer.train(["Sample text"])

# Encode text
result = tokenizer.encode("Hello", return_tokens=True)
print(result.tokens)  # ["H", "e", "l", "l", "o"]
```

## Normalizers

Normalizers are used to preprocess text before tokenization.

```python
from mltokenizer.normalization.normalizers import (
    ComposeNormalizer,
    LowercaseNormalizer,
    StripNormalizer,
    WhitespaceNormalizer
)

# Create a composed normalizer
normalizer = ComposeNormalizer([
    LowercaseNormalizer(),  # Convert to lowercase
    StripNormalizer(),      # Remove leading/trailing whitespace
    WhitespaceNormalizer()  # Normalize internal whitespace
])

# Use the normalizer
normalized_text = normalizer.normalize("  HELLO  WORLD  ")
print(normalized_text)  # "hello world"
```

## Special Tokens

Special tokens are used to mark specific positions or have special meaning in tokenized sequences.

```python
from mltokenizer.encoding.special_tokens import SpecialTokens

# Create special tokens handler for GPT-style models
special_tokens = SpecialTokens(
    pad_token="<pad>",
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>"
)

# Create special tokens handler for BERT-style models
bert_tokens = SpecialTokens(
    pad_token="[PAD]",
    unk_token="[UNK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# Get special tokens
all_tokens = special_tokens.all_special_tokens
print(all_tokens)  # ["<pad>", "<unk>", "<s>", "</s>"]
```

## Preprocessing Pipeline

The preprocessing pipeline applies a series of processors to text.

```python
from mltokenizer.preprocessing.pipeline import PreprocessingPipeline
from mltokenizer.preprocessing.text_cleaning import (
    RemoveHTMLTagsProcessor,
    RemoveURLsProcessor,
    RemoveEmailsProcessor
)

# Create a preprocessing pipeline
pipeline = PreprocessingPipeline([
    RemoveHTMLTagsProcessor(),
    RemoveURLsProcessor(),
    RemoveEmailsProcessor()
])

# Process text
text = "<p>Visit our <b>website</b> at https://example.com or email info@example.com</p>"
processed = pipeline.process(text)
print(processed)  # "Visit our website at  or email "
```

## Encoder

The encoder handles the mapping between tokens and their integer IDs.

```python
from mltokenizer.encoding.encoder import Encoder

# Create an encoder from a vocabulary
vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
encoder = Encoder(vocab)

# Encode tokens to IDs
ids = encoder.encode(["hello", "world", "unknown"])
print(ids)  # [2, 3, 1]  (1 is the [UNK] token ID)

# Decode IDs back to tokens
tokens = encoder.decode([2, 3, 1])
print(tokens)  # ["hello", "world", "[UNK]"]
```

## Performance Monitoring

Monitor tokenization performance and collect metrics.

```python
from mltokenizer.performance.metrics import MetricsTracker, track_time

# Create a metrics tracker
tracker = MetricsTracker()

# Track phases of tokenization
tracker.start_phase("normalization")
# ... Normalization code ...
tracker.end_phase()

tracker.start_phase("tokenization")
# ... Tokenization code ...
tracker.end_phase()

# Get metrics
metrics = tracker.get_metrics()
print(f"Total time: {metrics.total_time_ms}ms")
print(f"Tokens per second: {metrics.tokens_per_second}")

# Use as a decorator
class Tokenizer:
    @track_time("encoding")
    def encode(self, text):
        # ... Encoding code ...
        return result
```

## Model Adapters

Integration with popular ML model formats.

```python
from mltokenizer.models.bert import BertTokenizer
from mltokenizer.models.gpt import GPTTokenizer

# Load a BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load a GPT tokenizer
gpt_tokenizer = GPTTokenizer.from_pretrained("gpt2")

# Use the model-specific tokenizers
bert_output = bert_tokenizer.encode("This is a test", return_tokens=True)
gpt_output = gpt_tokenizer.encode("This is a test", return_tokens=True)

print(bert_output.tokens)
print(gpt_output.tokens)
```

## Saving and Loading Tokenizers

All tokenizers support saving and loading.

```python
from mltokenizer.algorithms.bpe import BPETokenizer

# Create and train a tokenizer
tokenizer = BPETokenizer(vocab_size=1000)
tokenizer.train(["Sample texts for training"])

# Save the tokenizer
tokenizer.save("./my_tokenizer")

# Load the tokenizer
loaded_tokenizer = BPETokenizer.load("./my_tokenizer")

# Verify it works the same
original = tokenizer.encode("Test")
loaded = loaded_tokenizer.encode("Test")
assert original.input_ids == loaded.input_ids
```

## API Server

Run the tokenization API server for network-based access.

```python
from mltokenizer.server.api import start_server

# Start the API server
start_server(host="0.0.0.0", port=8000)
```

## CLI Interface

Use the command-line interface for tokenization tasks.

```bash
# Train a tokenizer
tokenize train-tokenizer --tokenizer-type bpe --input-file data.txt --output-dir ./my_tokenizer

# Load a pretrained tokenizer
tokenize load-pretrained --model-name bert-base-uncased

# Tokenize text
tokenize tokenize "Hello, world!" --tokenizer-id bert_base_uncased

# Analyze token distribution
tokenize analyze-distribution --input-file corpus.txt --tokenizer-id bert_base_uncased

# Start the tokenization server
tokenize serve --port 8000
```

## Performance Considerations

MLTokenizer is designed for high performance:

1. **Batch Processing**: Use `encode_batch` for processing multiple texts efficiently
2. **Rust Extensions**: Critical components are implemented in Rust for speed
3. **Caching**: Token merges and encodings are cached when possible
4. **Parallelization**: Training and batch processing leverage multicore CPUs

```python
# Batch processing for better performance
texts = ["First text", "Second text", "Third text", "..."]
batch_output = tokenizer.encode_batch(texts)

# Check metrics
if hasattr(tokenizer, "metrics_tracker"):
    metrics = tokenizer.metrics_tracker.get_metrics()
    print(f"Processed at {metrics.tokens_per_second} tokens/second")
```

## Integration Examples

### Complete Tokenization Pipeline

```python
from mltokenizer.algorithms.bpe import BPETokenizer
from mltokenizer.normalization.normalizers import SequenceNormalizer
from mltokenizer.preprocessing.pipeline import PreprocessingPipeline
from mltokenizer.preprocessing.text_cleaning import RemoveHTMLTagsProcessor
from mltokenizer.encoding.special_tokens import SpecialTokens

# Create components
normalizer = SequenceNormalizer.default()
preprocessor = PreprocessingPipeline([RemoveHTMLTagsProcessor()])
special_tokens = SpecialTokens(bos_token="<s>", eos_token="</s>")

# Create tokenizer
tokenizer = BPETokenizer(
    vocab_size=1000,
    normalizer=normalizer,
    preprocessor=preprocessor,
    special_tokens=special_tokens
)

# Train tokenizer
tokenizer.train(["Sample text for training", "Another example"])

# Process text
html_text = "<p>This is a <b>test</b> sentence</p>"
result = tokenizer.encode(html_text, add_special_tokens=True, return_tokens=True)

# Examine result
print("Input IDs:", result.input_ids)
print("Tokens:", result.tokens)
print("Attention Mask:", result.attention_mask)

# Decode back to text
decoded = tokenizer.decode(result.input_ids)
print("Decoded Text:", decoded)
```

## Advanced Topics

### Custom Tokenizers

Create custom tokenizers by extending `BaseTokenizer`:

```python
from mltokenizer.core.base_tokenizer import BaseTokenizer, TokenizerType, TokenizedOutput

class CustomTokenizer(BaseTokenizer):
    def __init__(self, vocab_size=1000, **kwargs):
        super().__init__(
            tokenizer_type=TokenizerType.CUSTOM,
            vocab_size=vocab_size,
            **kwargs
        )
        
    def train(self, texts, **kwargs):
        # Custom training logic
        pass
        
    def encode(self, text, text_pair=None, add_special_tokens=True, return_tokens=False):
        # Custom encoding logic
        pass
        
    def decode(self, token_ids, skip_special_tokens=True):
        # Custom decoding logic
        pass
        
    def save(self, path):
        # Custom save logic
        pass
        
    @classmethod
    def load(cls, path):
        # Custom load logic
        pass
```

### Cross-Model Compatibility

Techniques for ensuring tokenization is compatible across models:

```python
# Load tokenizers from different model families
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
gpt_tokenizer = GPTTokenizer.from_pretrained("gpt2")

# Compare tokenizations
text = "This is a sample text for comparison"
bert_tokens = bert_tokenizer.encode(text, return_tokens=True).tokens
gpt_tokens = gpt_tokenizer.encode(text, return_tokens=True).tokens

print("BERT tokens:", bert_tokens)
print("GPT tokens:", gpt_tokens)

# Calculate overlap metrics
common_tokens = set(bert_tokens).intersection(set(gpt_tokens))
print(f"Common tokens: {len(common_tokens)}/{len(bert_tokens)}")
```

### Tokenizer Registry

Register and retrieve tokenizers:

```python
from mltokenizer.core.registry import TokenizerRegistry

# Create a registry
registry = TokenizerRegistry()

# Register tokenizers
registry.register_tokenizer("my_bpe", bpe_tokenizer)
registry.register_tokenizer("my_wordpiece", wordpiece_tokenizer)

# Get a tokenizer by ID
tokenizer = registry.get_tokenizer("my_bpe")

# List all registered tokenizers
all_tokenizers = registry.list_tokenizers()
```

## Best Practices

1. **Choose the Right Algorithm**:
   - BPE: Best for general-purpose tokenization, flexible with unseen text
   - WordPiece: Good for BERT-like models, handles morphologically rich languages well
   - Character: Simple but verbose, useful for some multilingual scenarios

2. **Training Dataset Selection**:
   - Use a corpus representative of the target domain
   - Include diverse vocabulary and writing styles
   - Ensure sufficient size (typically millions of tokens)

3. **Vocabulary Size**:
   - Typical ranges: 10,000-50,000 tokens
   - Too small: Many out-of-vocabulary words
   - Too large: Inefficient, may overfit to training data

4. **Special Tokens**:
   - Match the special tokens used by your target model
   - Be consistent with token IDs across training/inference

5. **Performance Optimization**:
   - Use batch processing for multiple texts
   - Consider caching results for repeated tokenization
   - Monitor metrics for bottlenecks

## Troubleshooting

Common issues and solutions:

1. **Slow Tokenization**:
   - Ensure you're using batch processing for multiple texts
   - Check if normalizers or preprocessors are causing the slowdown
   - Consider using simpler normalization for performance-critical applications

2. **Memory Issues**:
   - Large vocabularies consume more memory
   - Consider reducing vocabulary size or using a more efficient algorithm
   - Use streaming processing for very large inputs

3. **Poor Tokenization Quality**:
   - Ensure training data matches your target domain
   - Check that normalization is appropriate for your language
   - Increase vocabulary size to handle specialized terminology

4. **Incompatible Tokenization**:
   - Ensure you're using the same version of the tokenizer for training and inference
   - Check that special tokens are consistently applied
   - Save and load tokenizers using the provided methods, not manual configuration 