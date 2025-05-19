import time
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from mltokenizer.performance.metrics import (
    TokenizationMetrics,
    MetricsTracker,
    track_time
)


def test_tokenization_metrics_initialization():
    """Test initialization of TokenizationMetrics."""
    metrics = TokenizationMetrics()
    
    # Check default values
    assert metrics.total_texts == 0
    assert metrics.total_tokens == 0
    assert metrics.chars_per_token_mean == 0.0
    assert metrics.chars_per_token_std == 0.0
    assert metrics.normalization_time_ms == 0.0
    assert metrics.tokenization_time_ms == 0.0
    assert metrics.encoding_time_ms == 0.0
    assert metrics.total_time_ms == 0.0
    assert metrics.tokens_per_second == 0.0
    assert metrics.texts_per_second == 0.0
    assert metrics.peak_memory_mb == 0.0


def test_tokenization_metrics_to_dict():
    """Test conversion of TokenizationMetrics to dictionary."""
    metrics = TokenizationMetrics(
        total_texts=10,
        total_tokens=100,
        chars_per_token_mean=3.5,
        chars_per_token_std=1.2,
        normalization_time_ms=50.0,
        tokenization_time_ms=150.0,
        encoding_time_ms=100.0,
        total_time_ms=300.0,
        tokens_per_second=333.33,
        texts_per_second=33.33,
        peak_memory_mb=50.0
    )
    
    metrics_dict = metrics.to_dict()
    
    # Check dictionary values
    assert metrics_dict["total_texts"] == 10
    assert metrics_dict["total_tokens"] == 100
    assert metrics_dict["chars_per_token_mean"] == 3.5
    assert metrics_dict["chars_per_token_std"] == 1.2
    assert metrics_dict["normalization_time_ms"] == 50.0
    assert metrics_dict["tokenization_time_ms"] == 150.0
    assert metrics_dict["encoding_time_ms"] == 100.0
    assert metrics_dict["total_time_ms"] == 300.0
    assert metrics_dict["tokens_per_second"] == 333.33
    assert metrics_dict["texts_per_second"] == 33.33
    assert metrics_dict["peak_memory_mb"] == 50.0


def test_metrics_tracker_initialization():
    """Test initialization of MetricsTracker."""
    tracker = MetricsTracker()
    
    # Check initial state
    assert tracker.metrics is not None
    assert isinstance(tracker.metrics, TokenizationMetrics)
    assert tracker.timing_stack == []
    assert tracker.current_phase is None


def test_metrics_tracker_start_phase():
    """Test starting a tracking phase."""
    tracker = MetricsTracker()
    
    # Start a phase
    tracker.start_phase("normalization")
    
    # Check state
    assert tracker.current_phase == "normalization"
    assert len(tracker.timing_stack) == 1
    assert tracker.timing_stack[0][0] == "normalization"
    
    # Start another phase
    tracker.start_phase("tokenization")
    
    # Check updated state
    assert tracker.current_phase == "tokenization"
    assert len(tracker.timing_stack) == 2
    assert tracker.timing_stack[1][0] == "tokenization"


def test_metrics_tracker_end_phase():
    """Test ending a tracking phase."""
    tracker = MetricsTracker()
    
    # Start and end a normalization phase
    tracker.start_phase("normalization")
    time.sleep(0.01)  # Small delay for measurable duration
    duration = tracker.end_phase()
    
    # Check duration and metrics
    assert duration > 0
    assert tracker.metrics.normalization_time_ms > 0
    assert tracker.current_phase is None
    
    # Start and end a tokenization phase
    tracker.start_phase("tokenization")
    time.sleep(0.01)  # Small delay for measurable duration
    duration = tracker.end_phase()
    
    # Check duration and metrics
    assert duration > 0
    assert tracker.metrics.tokenization_time_ms > 0
    
    # Start and end an encoding phase
    tracker.start_phase("encoding")
    time.sleep(0.01)  # Small delay for measurable duration
    duration = tracker.end_phase()
    
    # Check duration and metrics
    assert duration > 0
    assert tracker.metrics.encoding_time_ms > 0


def test_metrics_tracker_track_encoding():
    """Test tracking encoding metrics."""
    tracker = MetricsTracker()
    
    # Track an encoding operation
    num_texts = 3
    tokens = [["hello", "world"], ["test", "token", "izer"], ["another", "example"]]
    original_texts = ["hello world", "test tokenizer", "another example"]
    
    # Set some timing metrics
    tracker.metrics.total_time_ms = 100.0
    
    tracker.track_encoding(num_texts, tokens, original_texts)
    
    # Check updated metrics
    assert tracker.metrics.total_texts == num_texts
    assert tracker.metrics.total_tokens == 7  # Total tokens across all sequences
    assert tracker.metrics.chars_per_token_mean > 0
    assert tracker.metrics.chars_per_token_std >= 0
    assert tracker.metrics.tokens_per_second > 0
    assert tracker.metrics.texts_per_second > 0


def test_metrics_tracker_get_metrics():
    """Test getting metrics from the tracker."""
    tracker = MetricsTracker()
    
    # Set some timing metrics
    tracker.metrics.normalization_time_ms = 50.0
    tracker.metrics.tokenization_time_ms = 150.0
    tracker.metrics.encoding_time_ms = 100.0
    
    # Get the metrics
    metrics = tracker.get_metrics()
    
    # Check total time calculation
    assert metrics.total_time_ms == 300.0


class TestClass:
    """Test class for track_time decorator."""
    
    def __init__(self):
        self.metrics_tracker = None
    
    @track_time("normalization")
    def normalize(self, text):
        """Mock normalization function."""
        time.sleep(0.01)  # Small delay for measurable duration
        return text.lower()
    
    @track_time("tokenization")
    def tokenize(self, text):
        """Mock tokenization function."""
        time.sleep(0.01)  # Small delay for measurable duration
        return text.split()


def test_track_time_decorator():
    """Test the track_time decorator."""
    test_obj = TestClass()
    
    # Call decorated methods
    result = test_obj.normalize("HELLO WORLD")
    assert result == "hello world"
    assert test_obj.metrics_tracker is not None
    assert test_obj.metrics_tracker.metrics.normalization_time_ms > 0
    
    # Call another decorated method
    tokens = test_obj.tokenize("hello world")
    assert tokens == ["hello", "world"]
    assert test_obj.metrics_tracker.metrics.tokenization_time_ms > 0


def test_track_time_decorator_no_tracker():
    """Test the track_time decorator creates a tracker if none exists."""
    test_obj = TestClass()
    test_obj.metrics_tracker = None
    
    # Call decorated method
    test_obj.normalize("HELLO WORLD")
    
    # Check that a tracker was created
    assert test_obj.metrics_tracker is not None
    assert isinstance(test_obj.metrics_tracker, MetricsTracker)


if __name__ == "__main__":
    pytest.main() 