import json
import pickle
from pathlib import Path
from typing import Any, Union, Literal, Optional

from loguru import logger

# Define supported serialization formats
SerializationFormat = Literal["json", "pickle", "text"]

def save_data(
    data: Any, 
    file_path: Union[str, Path],
    format: SerializationFormat = "json", 
    encoding: str = "utf-8",
    **kwargs
) -> None:
    """Saves data to a file in the specified format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format == "json":
            with open(path, "w", encoding=encoding) as f:
                json.dump(data, f, **kwargs)
        elif format == "pickle":
            with open(path, "wb") as f: # Pickle requires binary mode
                pickle.dump(data, f, **kwargs)
        elif format == "text":
            with open(path, "w", encoding=encoding) as f:
                if isinstance(data, list):
                    for item in data:
                        f.write(str(item) + "\n")
                else:
                    f.write(str(data))
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
        logger.info(f"Data successfully saved to {path} in {format} format.")
    except Exception as e:
        logger.error(f"Failed to save data to {path} in {format} format: {e}")
        raise

def load_data(
    file_path: Union[str, Path], 
    format: SerializationFormat = "json", 
    encoding: str = "utf-8",
    **kwargs
) -> Any:
    """Loads data from a file in the specified format."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found for loading: {path}")
        raise FileNotFoundError(f"No such file: '{path}'")
    
    try:
        if format == "json":
            with open(path, "r", encoding=encoding) as f:
                return json.load(f, **kwargs)
        elif format == "pickle":
            with open(path, "rb") as f: # Pickle requires binary mode
                return pickle.load(f, **kwargs)
        elif format == "text":
            with open(path, "r", encoding=encoding) as f:
                # For text, typically return list of lines or full content
                # This behavior might need to be more specific based on use case
                if kwargs.get("return_lines", False):
                    return [line.strip("\n") for line in f.readlines()]
                return f.read()
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
    except Exception as e:
        logger.error(f"Failed to load data from {path} in {format} format: {e}")
        raise

# Tokenizer-specific serialization/deserialization will likely live within
# each tokenizer's `save` and `load` methods, which might use these utilities.

logger.info("Serialization module loaded. Provides save/load utilities.") 