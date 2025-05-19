import time
import gc
import random
from typing import Dict, List

import numpy as np
import pytest

from mltokenizer.algorithms.bpe import BPETokenizer
from mltokenizer.algorithms.wordpiece import WordpieceTokenizer
from mltokenizer.algorithms.character import CharacterTokenizer
from mltokenizer.core.base_tokenizer import BaseTokenizer
from mltokenizer.normalization.normalizers import LowercaseNormalizer


@pytest.fixture
def performance_texts() -> List[str]:
    """Generate a set of texts for performance testing."""
    # Fixed set of sample sentences
    sample_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "This is a sample sentence for tokenization benchmarking.",
        "Machine learning models process text as numerical data.",
        "Tokenization is the process of breaking text into smaller pieces called tokens.",
        "Natural language processing systems rely on effective tokenization strategies.",
        "Performance testing measures how efficiently a system completes its tasks.",
        "Throughput refers to the amount of data processed in a given time period.",
        "Memory usage is critical for applications running in constrained environments.",
        "Python provides several tools for measuring and profiling code performance.",
        "Benchmarking helps compare different algorithms or implementations.",
    ]
    
    # Generate a larger test set by repeating and combining these sentences
    texts = []
    for _ in range(100):  # Generate 100 paragraphs
        paragraph_length = random.randint(3, 10)  # 3 to 10 sentences per paragraph
        paragraph = " ".join(random.sample(sample_sentences, paragraph_length))
        texts.append(paragraph)
    
    return texts


@pytest.fixture
def trained_tokenizers(performance_texts) -> Dict[str, BaseTokenizer]:
    """Create and train a set of tokenizers for performance testing."""
    # Select a subset of texts for training
    train_texts = performance_texts[:20]
    
    # Configure tokenizers
    bpe = BPETokenizer(
        vocab_size=1000,
        normalizer=LowercaseNormalizer()
    )
    
    wordpiece = WordpieceTokenizer(
        vocab_size=1000,
        normalizer=LowercaseNormalizer()
    )
    
    character = CharacterTokenizer(
        normalizer=LowercaseNormalizer()
    )
    
    # Train each tokenizer
    bpe.train(train_texts, min_frequency=1)
    wordpiece.train(train_texts, min_frequency=1)
    character.train(train_texts)
    
    return {
        "bpe": bpe,
        "wordpiece": wordpiece,
        "character": character
    }


def measure_memory():
    """Measure current memory usage."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Memory in MB
    except ImportError:
        # Fallback if psutil is not available
        return 0


def benchmark_tokenizer(tokenizer: BaseTokenizer, texts: List[str], batch_size: int = 32) -> Dict:
    """Benchmark a tokenizer on a set of texts."""
    # Force garbage collection before measurement
    gc.collect()
    
    # Measure initial memory
    initial_memory = measure_memory()
    
    # Prepare batches
    num_batches = (len(texts) + batch_size - 1) // batch_size  # Ceiling division
    batches = [texts[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    
    # Measure single text tokenization
    single_times = []
    for _ in range(10):  # Take average of 10 runs
        sample_text = random.choice(texts)
        start_time = time.time()
        tokenizer.encode(sample_text)
        end_time = time.time()
        single_times.append(end_time - start_time)
    
    # Measure batch tokenization
    batch_times = []
    for batch in batches:
        start_time = time.time()
        tokenizer.encode_batch(batch)
        end_time = time.time()
        batch_times.append(end_time - start_time)
    
    # Measure final memory after tokenization
    final_memory = measure_memory()
    
    # Calculate performance metrics
    avg_single_time = np.mean(single_times)
    avg_batch_time = np.mean(batch_times)
    texts_per_second_single = 1 / avg_single_time
    texts_per_second_batch = len(texts) / sum(batch_times)
    memory_usage = final_memory - initial_memory
    
    # Calculate approximate token throughput
    # This is an approximation - actually counting tokens would be more accurate
    token_count = 0
    for sample_text in texts[:5]:  # Use a small sample for estimation
        token_count += len(tokenizer.encode(sample_text).input_ids)
    avg_tokens_per_text = token_count / 5
    tokens_per_second = texts_per_second_batch * avg_tokens_per_text
    
    return {
        "avg_single_time_ms": avg_single_time * 1000,
        "avg_batch_time_ms": avg_batch_time * 1000,
        "texts_per_second_single": texts_per_second_single,
        "texts_per_second_batch": texts_per_second_batch,
        "tokens_per_second": tokens_per_second,
        "memory_usage_mb": memory_usage,
        "avg_tokens_per_text": avg_tokens_per_text
    }


@pytest.mark.benchmark
def test_tokenizer_single_text_performance(trained_tokenizers, performance_texts):
    """Test single text tokenization performance."""
    results = {}
    
    for name, tokenizer in trained_tokenizers.items():
        # Test single text performance
        single_text = performance_texts[0]
        
        # Warm-up run
        tokenizer.encode(single_text)
        
        # Measure performance
        times = []
        for _ in range(100):  # 100 iterations for stable measurement
            start_time = time.time()
            tokenizer.encode(single_text)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        results[name] = avg_time
    
    # Print results
    for name, avg_time in results.items():
        print(f"{name} tokenizer: {avg_time:.2f}ms per single text")
    
    # No specific assertions, this is primarily for reporting
    # But we can check that all tokenizers completed
    assert len(results) == len(trained_tokenizers)


@pytest.mark.benchmark
def test_tokenizer_batch_performance(trained_tokenizers, performance_texts):
    """Test batch tokenization performance."""
    batch_size = 32
    results = {}
    
    for name, tokenizer in trained_tokenizers.items():
        # Warm-up run
        tokenizer.encode_batch(performance_texts[:batch_size])
        
        # Measure performance
        start_time = time.time()
        tokenizer.encode_batch(performance_texts)
        end_time = time.time()
        
        total_time = end_time - start_time
        texts_per_second = len(performance_texts) / total_time
        results[name] = texts_per_second
    
    # Print results
    for name, throughput in results.items():
        print(f"{name} tokenizer: {throughput:.2f} texts/second (batch mode)")
    
    # No specific assertions, this is primarily for reporting
    assert len(results) == len(trained_tokenizers)


@pytest.mark.benchmark
def test_tokenizer_comprehensive_benchmark(trained_tokenizers, performance_texts):
    """Run a comprehensive benchmark of all tokenizers."""
    results = {}
    
    for name, tokenizer in trained_tokenizers.items():
        print(f"\nBenchmarking {name} tokenizer...")
        metrics = benchmark_tokenizer(tokenizer, performance_texts)
        results[name] = metrics
        
        # Print results in a readable format
        print(f"  Single text time: {metrics['avg_single_time_ms']:.2f}ms")
        print(f"  Batch processing time: {metrics['avg_batch_time_ms']:.2f}ms per batch")
        print(f"  Throughput (single): {metrics['texts_per_second_single']:.2f} texts/sec")
        print(f"  Throughput (batch): {metrics['texts_per_second_batch']:.2f} texts/sec")
        print(f"  Token throughput: {metrics['tokens_per_second']:.2f} tokens/sec")
        print(f"  Average tokens per text: {metrics['avg_tokens_per_text']:.2f}")
        if metrics['memory_usage_mb'] > 0:
            print(f"  Memory usage: {metrics['memory_usage_mb']:.2f}MB")
    
    # Compare relative performance
    fastest_tokenizer = min(results.items(), key=lambda x: x[1]['avg_batch_time_ms'])[0]
    print(f"\nFastest tokenizer: {fastest_tokenizer}")
    
    # Calculate relative speeds
    baseline = results[fastest_tokenizer]['texts_per_second_batch']
    for name, metrics in results.items():
        relative_speed = metrics['texts_per_second_batch'] / baseline
        print(f"{name} is {relative_speed:.2f}x the speed of the fastest ({fastest_tokenizer})")
    
    # No specific assertions, this is primarily for reporting
    assert len(results) == len(trained_tokenizers)


@pytest.mark.benchmark
def test_tokenizer_memory_usage(trained_tokenizers, performance_texts):
    """Test memory usage of different tokenizers."""
    if measure_memory() == 0:
        pytest.skip("psutil not available for memory measurement")
    
    results = {}
    
    for name, tokenizer in trained_tokenizers.items():
        # Force garbage collection
        gc.collect()
        
        # Measure baseline memory
        baseline_memory = measure_memory()
        
        # Process a large batch of texts
        tokenizer.encode_batch(performance_texts)
        
        # Measure memory after processing
        final_memory = measure_memory()
        
        # Calculate memory usage
        memory_usage = final_memory - baseline_memory
        results[name] = memory_usage
    
    # Print results
    for name, memory in results.items():
        print(f"{name} tokenizer memory usage: {memory:.2f}MB")
    
    # No specific assertions, this is primarily for reporting
    assert len(results) == len(trained_tokenizers)


@pytest.mark.benchmark
def test_tokenizer_scaling_performance(trained_tokenizers):
    """Test how tokenizer performance scales with input size."""
    # Select a single tokenizer for this test
    tokenizer = trained_tokenizers["bpe"]
    
    # Generate texts of increasing length
    base_text = "The quick brown fox jumps over the lazy dog. "
    text_sizes = [1, 10, 100, 1000]  # Number of repetitions
    
    results = []
    
    for size in text_sizes:
        text = base_text * size
        
        # Warm-up
        tokenizer.encode(text)
        
        # Measure performance
        start_time = time.time()
        output = tokenizer.encode(text)
        end_time = time.time()
        
        token_count = len(output.input_ids)
        processing_time = end_time - start_time
        characters = len(text)
        
        results.append({
            "text_size": size,
            "character_count": characters,
            "token_count": token_count,
            "processing_time_ms": processing_time * 1000,
            "tokens_per_second": token_count / processing_time,
            "characters_per_second": characters / processing_time
        })
    
    # Print results
    print("\nTokenizer scaling performance:")
    for result in results:
        print(f"Size: {result['text_size']} repeats ({result['character_count']} chars) → "
              f"{result['processing_time_ms']:.2f}ms, "
              f"{result['tokens_per_second']:.2f} tokens/sec, "
              f"{result['characters_per_second']:.2f} chars/sec")
    
    # Check if performance scales somewhat linearly
    # This is a very rough check - in practice many factors can cause non-linearity
    if len(results) >= 2:
        time_ratios = []
        size_ratios = []
        
        for i in range(1, len(results)):
            time_ratio = results[i]['processing_time_ms'] / results[i-1]['processing_time_ms']
            size_ratio = results[i]['character_count'] / results[i-1]['character_count']
            time_ratios.append(time_ratio)
            size_ratios.append(size_ratio)
        
        print("\nScaling factors (time ratio / size ratio, closer to 1.0 is more linear):")
        for i, (t_ratio, s_ratio) in enumerate(zip(time_ratios, size_ratios)):
            scaling_factor = t_ratio / s_ratio
            print(f"Size {text_sizes[i]} → {text_sizes[i+1]}: {scaling_factor:.2f}")


if __name__ == "__main__":
    pytest.main() 