# This file will contain the PyO3 bindings for the Rust extensions.
# Example:
# from .mltokenizer_rs import rust_function_example

# def call_rust_function():
# return rust_function_example() 

from ..mltokenizer_rs import sum_as_string

def call_sum_as_string(a: int, b: int) -> str:
    """Calls the Rust 'sum_as_string' function."""
    return sum_as_string(a, b) 