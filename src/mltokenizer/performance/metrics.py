import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np


@dataclass
class TokenizationMetrics:
    """Metrics for tokenization performance."""
    
    # Tokenization counts
    total_texts: int = 0
    total_tokens: int = 0
    
    # Character statistics
    chars_per_token_mean: float = 0.0
    chars_per_token_std: float = 0.0
    
    # Timing metrics (in milliseconds)
    normalization_time_ms: float = 0.0
    tokenization_time_ms: float = 0.0
    encoding_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Tokens per second
    tokens_per_second: float = 0.0
    texts_per_second: float = 0.0
    
    # Memory usage
    peak_memory_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to a dictionary."""
        return {
            "total_texts": self.total_texts,
            "total_tokens": self.total_tokens,
            "chars_per_token_mean": self.chars_per_token_mean,
            "chars_per_token_std": self.chars_per_token_std,
            "normalization_time_ms": self.normalization_time_ms,
            "tokenization_time_ms": self.tokenization_time_ms,
            "encoding_time_ms": self.encoding_time_ms,
            "total_time_ms": self.total_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "texts_per_second": self.texts_per_second,
            "peak_memory_mb": self.peak_memory_mb,
        }


class MetricsTracker:
    """Utility for tracking tokenization metrics."""
    
    def __init__(self):
        """Initialize the metrics tracker."""
        self.metrics = TokenizationMetrics()
        self.timing_stack = []
        self.current_phase = None
    
    def start_phase(self, phase: str) -> None:
        """Start tracking time for a phase.
        
        Args:
            phase: Name of the phase
        """
        self.current_phase = phase
        self.timing_stack.append((phase, time.time()))
    
    def end_phase(self) -> float:
        """End tracking time for the current phase.
        
        Returns:
            Duration of the phase in milliseconds
        """
        if not self.timing_stack:
            return 0.0
        
        phase, start_time = self.timing_stack.pop()
        duration_ms = (time.time() - start_time) * 1000
        
        # Update the appropriate metric
        if phase == "normalization":
            self.metrics.normalization_time_ms += duration_ms
        elif phase == "tokenization":
            self.metrics.tokenization_time_ms += duration_ms
        elif phase == "encoding":
            self.metrics.encoding_time_ms += duration_ms
        
        self.current_phase = self.timing_stack[-1][0] if self.timing_stack else None
        
        return duration_ms
    
    def track_encoding(
        self, 
        num_texts: int, 
        tokens: List[List[str]], 
        original_texts: List[str]
    ) -> None:
        """Track metrics for an encoding operation.
        
        Args:
            num_texts: Number of texts processed
            tokens: List of token lists
            original_texts: Original texts
        """
        self.metrics.total_texts += num_texts
        
        # Count tokens
        total_tokens = sum(len(t) for t in tokens)
        self.metrics.total_tokens += total_tokens
        
        # Calculate characters per token
        chars_per_token = []
        for text, token_list in zip(original_texts, tokens):
            if token_list:
                chars_per_token.append(len(text) / len(token_list))
        
        if chars_per_token:
            self.metrics.chars_per_token_mean = np.mean(chars_per_token)
            self.metrics.chars_per_token_std = np.std(chars_per_token)
        
        # Calculate throughput
        if self.metrics.total_time_ms > 0:
            self.metrics.tokens_per_second = (self.metrics.total_tokens / self.metrics.total_time_ms) * 1000
            self.metrics.texts_per_second = (self.metrics.total_texts / self.metrics.total_time_ms) * 1000
        
        # Estimate memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.metrics.peak_memory_mb = memory_info.rss / (1024 * 1024)
        except ImportError:
            pass
    
    def get_metrics(self) -> TokenizationMetrics:
        """Get the current metrics.
        
        Returns:
            Current metrics
        """
        # Update total time
        self.metrics.total_time_ms = (
            self.metrics.normalization_time_ms + 
            self.metrics.tokenization_time_ms + 
            self.metrics.encoding_time_ms
        )
        
        return self.metrics


def track_time(phase: str) -> Callable:
    """Decorator to track time spent in a function.
    
    Args:
        phase: Name of the phase
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if metrics tracker exists
            if not hasattr(self, "metrics_tracker"):
                self.metrics_tracker = MetricsTracker()
            
            self.metrics_tracker.start_phase(phase)
            result = func(self, *args, **kwargs)
            self.metrics_tracker.end_phase()
            
            return result
        return wrapper
    return decorator 