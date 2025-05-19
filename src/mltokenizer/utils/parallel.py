# src/mltokenizer/utils/parallel.py
import multiprocessing as mp
from typing import Callable, List, Any
from tqdm import tqdm

def parallel_process(data: List[Any], func: Callable, n_jobs: int = -1, use_kwargs: bool = False, front_num: int = 3) -> List[Any]:
    """Placeholder for parallel processing utility.
    Actual implementation would depend on the specific needs and could use multiprocessing, joblib, etc.
    This is a simplified version based on typical BPE training usage.
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    if n_jobs == 1:
        # Single process execution (useful for debugging)
        results = []
        for item in tqdm(data, desc=f"Processing with {func.__name__}"):
            if use_kwargs:
                results.append(func(**item))
            else:
                results.append(func(item))
        return results

    # Multi-process execution
    # This is a very basic multiprocessing pool example. 
    # Error handling, chunking, shared memory considerations might be needed for robustness.
    results = [] # Ensure results is initialized
    try:
        with mp.Pool(processes=n_jobs) as pool:
            if use_kwargs:
                # This is a simplified approach. For kwargs with starmap, data items should be dicts.
                # And func should be wrapped or starmap_async used more carefully.
                # For now, assuming if use_kwargs is true, func can handle dict unpacking.
                results = list(tqdm(pool.imap(func, data), total=len(data), desc=f"Processing with {func.__name__} (parallel)"))
            else:
                results = list(tqdm(pool.imap(func, data), total=len(data), desc=f"Processing with {func.__name__} (parallel)"))
    except Exception as e:
        # Fallback or error logging
        print(f"Parallel processing failed: {e}. Falling back to serial execution.")
        results = [func(item) for item in tqdm(data)] # Basic serial fallback
        
    return results

__all__ = ["parallel_process"] 