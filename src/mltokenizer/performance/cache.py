import functools
from typing import Dict, Any, Callable, Optional, TypeVar
from collections import OrderedDict

from loguru import logger

_R = TypeVar('_R') # Return type for cached functions

# Simple LRU Cache implementation
class LRUCache:
    """A simple Least Recently Used (LRU) cache."""
    def __init__(self, max_size: int = 128):
        if max_size <= 0:
            raise ValueError("LRUCache max_size must be positive")
        self.cache: OrderedDict[Any, Any] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[Any]:
        if key not in self.cache:
            self.misses += 1
            return None
        else:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

    def put(self, key: Any, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False) # Pop the oldest item
    
    def clear(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("LRUCache cleared.")

    def __contains__(self, key: Any) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)
    
    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": len(self.cache), "max_size": self.max_size}

# Global cache instances (can be configured or made per-tokenizer)
TOKENIZATION_CACHE = LRUCache(max_size=1024)       # For caching (text -> tokens) results
ENCODING_CACHE = LRUCache(max_size=2048)          # For caching (tokens -> ids) or (text -> ids)
VOCAB_LOOKUP_CACHE = LRUCache(max_size=4096)      # For caching (token -> id) and (id -> token)

def lru_cache_decorator(cache_instance: LRUCache) -> Callable[..., Callable[..., _R]]:
    """Decorator to apply a specific LRUCache instance to a function."""
    def decorator(func: Callable[..., _R]) -> Callable[..., _R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> _R:
            # Create a cache key from args and kwargs
            # This needs to be robust and handle unhashable types if necessary.
            # For simplicity, using a tuple of args and sorted kwargs items.
            # Consider using `inspect.signature` for more robust key generation.
            key_parts = list(args)
            if kwargs:
                for item in sorted(kwargs.items()):
                    key_parts.append(item)
            cache_key = tuple(key_parts)
            
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            result = func(*args, **kwargs)
            cache_instance.put(cache_key, result)
            return result
        
        # Attach cache controls to the wrapper
        wrapper.cache = cache_instance # type: ignore
        wrapper.cache_clear = cache_instance.clear # type: ignore
        wrapper.cache_stats = cache_instance.stats # type: ignore
        return wrapper
    return decorator

# Example usage:
# @lru_cache_decorator(TOKENIZATION_CACHE)
# def expensive_tokenize_operation(text: str, options: Any) -> List[str]:
#    # ... expensive operation ...
#    pass

logger.info("Caching module loaded. Includes LRUCache and decorators.") 