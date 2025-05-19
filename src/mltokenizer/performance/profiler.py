import cProfile
import pstats
import io
import time
from functools import wraps
from typing import Callable, Any, Optional

from loguru import logger

def profile_function(sort_by: str = 'cumulative', top_n: int = 10) -> Callable:
    """A decorator to profile a function using cProfile.

    Args:
        sort_by: How to sort the profiling results (e.g., 'cumulative', 'tottime').
        top_n: Number of top results to print.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            profiler.disable()
            
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
            ps.print_stats(top_n)
            
            logger.info(f"Profiling results for {func.__name__} (executed in {end_time - start_time:.4f}s):\n{s.getvalue()}")
            return result
        return wrapper
    return decorator

class CodeProfiler:
    """Context manager for profiling blocks of code."""
    def __init__(self, sort_by: str = 'cumulative', top_n: int = 10, name: Optional[str] = None):
        self.sort_by = sort_by
        self.top_n = top_n
        self.profiler = cProfile.Profile()
        self.name = name or "UnnamedProfile"
        self.start_time = 0.0

    def __enter__(self):
        logger.info(f"Starting profiler for section: {self.name}")
        self.profiler.enable()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.disable()
        end_time = time.perf_counter()
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(self.sort_by)
        ps.print_stats(self.top_n)
        logger.info(f"Profiling results for section '{self.name}' (executed in {end_time - self.start_time:.4f}s):\n{s.getvalue()}")
        # Return False to propagate exceptions if any occurred
        return False 

# Example usage:
# @profile_function(top_n=5)
# def my_slow_function():
#     # ... do something slow ...
#     pass

# with CodeProfiler(name="my_critical_block", top_n=3):
#     # ... critical code section ...
#     pass

logger.info("Code profiler module loaded. Use @profile_function decorator or CodeProfiler context manager.") 