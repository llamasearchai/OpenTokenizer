# src/mltokenizer/rust_extensions/__init__.py
# This directory is intended for Rust extensions (e.g., using PyO3).
# For now, this __init__.py makes it a Python package.

from loguru import logger
from .bindings import call_sum_as_string # Import from our bindings file

# Attempt to import the Rust extension module
# The actual name 'mltokenizer_rs' is based on Cargo.toml's [lib].name
try:
    # The bindings file should handle the direct import from mltokenizer_rs
    # For example, bindings.py would have: from ..mltokenizer_rs import sum_as_string
    # And then call_sum_as_string would use that.
    # We just need to ensure the maturin build places mltokenizer_rs where python can find it.
    
    # To verify the rust module itself is loadable, we can try a direct import here for checking
    import mltokenizer_rs as native_module_check 

    HAS_RUST_EXTENSIONS = True
    logger.info("Successfully imported Rust extensions (mltokenizer_rs).")

    # Define __all__ to control `from mltokenizer.rust_extensions import *`
    __all__ = [
        "HAS_RUST_EXTENSIONS",
        "call_sum_as_string", # Expose our wrapped function
        # Add other specific imported names here if you chose that route
        # "example_rust_function",
        # "ExampleRustClass"
    ]

except ImportError as e:
    logger.warning(
        f"Rust extensions (mltokenizer_rs) not found or import failed: {e}. "
        f"Performance-critical operations might use Python fallbacks or be unavailable. "
        f"Ensure Rust extensions are compiled (e.g., via 'poetry run maturin develop')."
    )
    HAS_RUST_EXTENSIONS = False
    
    # Create a placeholder if Rust extensions are critical and you want to avoid AttributeErrors
    class _MockRustFunction:
        def __init__(self, name):
            self._name = name

        def __call__(self, *args, **kwargs):
            logger.error(
                f"Rust function or attribute '{self._name}' accessed, but Rust extensions are not available. "
                f"Called with args: {args}, kwargs: {kwargs}"
            )
            raise NotImplementedError(
                f"Functionality '{self._name}' requires compiled Rust extensions (mltokenizer_rs) which are not loaded."
            )
    
    # Mock the functions we expect from bindings.py
    _mocked_call_sum_as_string = _MockRustFunction("call_sum_as_string")

    # To allow `from . import call_sum_as_string` to work even if rust fails,
    # we can assign the mock directly here if we want to keep the import structure clean at call sites.
    # However, the current bindings.py directly imports from mltokenizer_rs, so if that fails, 
    # the import `from .bindings import call_sum_as_string` at the top of this file would fail.
    # A more robust way for mocking would be for bindings.py itself to handle the ImportError from mltokenizer_rs
    # and define a mock version of sum_as_string there.

    # For now, let's assume if the top import `from .bindings import call_sum_as_string` succeeded,
    # but the `import mltokenizer_rs` fails, something is inconsistent.
    # The primary role of this __init__ is to set HAS_RUST_EXTENSIONS and potentially export.

    # If `from .bindings import call_sum_as_string` itself fails due to mltokenizer_rs not being there,
    # this part of the code won't even be reached with `call_sum_as_string` in its current form.
    # The ImportError from the top of the file would be the one that's raised.

    # A simpler mock for __all__ when import fails:
    call_sum_as_string = _mocked_call_sum_as_string # Make the mock available for direct import
    __all__ = ["HAS_RUST_EXTENSIONS", "call_sum_as_string"]

# Example usage (within mltokenizer, other modules could do this):
# from . import HAS_RUST_EXTENSIONS, call_sum_as_string
# if HAS_RUST_EXTENSIONS:
#     result = call_sum_as_string(1,2)
# else:
#     # Potentially call the mock, which will raise NotImplementedError
#     # Or have an explicit Python fallback
#     try:
#         result = call_sum_as_string(1,2)
#     except NotImplementedError:
#         logger.info("Using Python fallback for sum_as_string")
#         result = str(1 + 2) # python fallback example
#     pass 