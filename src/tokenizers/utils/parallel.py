import multiprocessing as mp
from functools import partial
from typing import Any, Callable, Iterable, List, Optional, TypeVar

from tqdm import tqdm


T = TypeVar('T')
R = TypeVar('R')


def _process_chunk(func: Callable[[T], R], chunk: List[T]) -> List[R]:
    """Process a chunk of items with the given function.
    
    Args:
        func: Function to apply to each item
        chunk: List of items to process
        
    Returns:
        List of results
    """
    return [func(item) for item in chunk]


def parallel_process(
    items: Iterable[T],
    func: Callable[[T], R],
    n_jobs: int = -1,
    chunk_size: Optional[int] = None,
    show_progress: bool = True
) -> List[R]:
    """Process items in parallel.
    
    Args:
        items: Iterable of items to process
        func: Function to apply to each item
        n_jobs: Number of worker processes (-1 for all cores)
        chunk_size: Number of items per worker chunk
        show_progress: Whether to show a progress bar
        
    Returns:
        List of results
    """
    # Convert items to list if it's not already
    items_list = list(items)
    total = len(items_list)
    
    # Determine number of workers
    if n_jobs <= 0:
        n_jobs = mp.cpu_count()
    
    # Use appropriate chunk size if not specified
    if chunk_size is None:
        chunk_size = max(1, total // (n_jobs * 10))
    
    # Create chunks
    chunks = [items_list[i:i + chunk_size] for i in range(0, total, chunk_size)]
    
    # Process in parallel
    results = []
    with mp.Pool(n_jobs) as pool:
        process_func = partial(_process_chunk, func)
        
        if show_progress:
            for result in tqdm(pool.imap(process_func, chunks), total=len(chunks)):
                results.extend(result)
        else:
            for result in pool.imap(process_func, chunks):
                results.extend(result)
    
    return results